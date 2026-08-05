//! Desktop control panel for a HavnAI GPU node.
//!
//! The Python runtime already knows how to diagnose itself (`client.doctor`)
//! and how to acquire weights (`client.fetch_models`), and both speak JSON.
//! This app is deliberately a thin shell over those: it discovers the install,
//! runs those tools, streams their output to the UI, and supervises the node
//! process. Keeping the logic in Python means the desktop app and a headless
//! SSH operator can never disagree about whether a node is healthy.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

/// Where the node lives on disk, and which pieces of it are present.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InstallState {
    pub installed: bool,
    pub havnai_home: String,
    pub runtime_dir: String,
    pub python: String,
    pub version: String,
    pub has_runtime: bool,
    pub has_venv: bool,
    pub service_installed: bool,
    pub platform: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeConfig {
    pub server_url: String,
    pub join_token: String,
    pub wallet: String,
    pub node_name: String,
    pub creator_mode: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct CommandOutput {
    pub success: bool,
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// A line of streamed output from a long-running child process.
#[derive(Debug, Serialize, Clone)]
struct StreamLine {
    stream: String,
    line: String,
}

#[derive(Debug, Serialize, Clone)]
struct StreamDone {
    success: bool,
    code: i32,
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn havnai_home() -> PathBuf {
    std::env::var_os("HAVNAI_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".havnai"))
}

fn runtime_dir() -> PathBuf {
    havnai_home().join("current")
}

/// The interpreter that owns the node's dependencies.
fn venv_python() -> PathBuf {
    let base = havnai_home().join("venv");
    let candidate = if cfg!(windows) {
        base.join("Scripts").join("python.exe")
    } else {
        base.join("bin").join("python")
    };
    if candidate.exists() {
        candidate
    } else {
        PathBuf::from("python3")
    }
}

fn env_file() -> PathBuf {
    havnai_home().join(".env")
}

fn service_file() -> PathBuf {
    if cfg!(target_os = "macos") {
        home_dir()
            .join("Library")
            .join("LaunchAgents")
            .join("com.havnai.node.plist")
    } else {
        home_dir()
            .join(".config")
            .join("systemd")
            .join("user")
            .join("havnai-node.service")
    }
}

// ---------------------------------------------------------------------------
// Configuration file handling
// ---------------------------------------------------------------------------

fn parse_env_file(path: &Path) -> HashMap<String, String> {
    let mut values = HashMap::new();
    let Ok(contents) = std::fs::read_to_string(path) else {
        return values;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            values.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    values
}

/// Rewrite the node's `.env`, preserving keys the app does not manage.
fn write_env_file(config: &NodeConfig) -> Result<(), String> {
    let path = env_file();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let mut values = parse_env_file(&path);
    values.insert("SERVER_URL".into(), config.server_url.clone());
    values.insert("HAVNAI_SERVER_URL".into(), config.server_url.clone());
    values.insert("WALLET".into(), config.wallet.clone());
    values.insert("NODE_NAME".into(), config.node_name.clone());
    values.insert(
        "CREATOR_MODE".into(),
        if config.creator_mode { "true" } else { "false" }.into(),
    );
    if config.join_token.is_empty() {
        values.remove("JOIN_TOKEN");
        values.remove("HAVNAI_NODE_TOKEN");
    } else {
        values.insert("JOIN_TOKEN".into(), config.join_token.clone());
        values.insert("HAVNAI_NODE_TOKEN".into(), config.join_token.clone());
    }

    let mut keys: Vec<&String> = values.keys().collect();
    keys.sort();
    let body: String = keys
        .iter()
        .map(|key| format!("{}={}\n", key, values[*key]))
        .collect();

    std::fs::write(&path, body).map_err(|e| e.to_string())?;

    // The file holds the join token; keep it owner-only.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Build the environment a node subprocess should inherit.
fn node_env() -> HashMap<String, String> {
    let mut env = parse_env_file(&env_file());
    env.insert(
        "HAVNAI_HOME".into(),
        havnai_home().to_string_lossy().to_string(),
    );
    // Unbuffered so streamed progress reaches the UI as it happens rather than
    // arriving in one lump when the child exits.
    env.insert("PYTHONUNBUFFERED".into(), "1".into());
    env.insert("NO_COLOR".into(), "1".into());
    env
}

// ---------------------------------------------------------------------------
// Process helpers
// ---------------------------------------------------------------------------

fn run_capture(program: &str, args: &[&str], cwd: Option<PathBuf>) -> CommandOutput {
    let mut command = Command::new(program);
    command.args(args);
    for (key, value) in node_env() {
        command.env(key, value);
    }
    if let Some(dir) = cwd {
        if dir.exists() {
            command.current_dir(dir);
        }
    }

    match command.output() {
        Ok(output) => CommandOutput {
            success: output.status.success(),
            code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        },
        Err(err) => CommandOutput {
            success: false,
            code: -1,
            stdout: String::new(),
            stderr: format!("failed to run {program}: {err}"),
        },
    }
}

/// Forward one of a child's output pipes to the UI, a line at a time.
///
/// Generic over the reader because `ChildStdout` and `ChildStderr` are distinct
/// types, and both pipes have to be drained concurrently: a child that fills
/// whichever pipe we are not reading would block forever.
fn spawn_reader<R>(
    handle: R,
    app: AppHandle,
    event: String,
    label: &'static str,
) -> std::thread::JoinHandle<()>
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        for line in BufReader::new(handle).lines().map_while(Result::ok) {
            let _ = app.emit(
                &event,
                StreamLine {
                    stream: label.to_string(),
                    line,
                },
            );
        }
    })
}

/// Run a child process, emitting each output line to the UI as it arrives.
///
/// `event` receives `{stream, line}` payloads; `{event}-done` receives the exit
/// status. Long installs and multi-gigabyte downloads are the whole reason this
/// exists - the operator needs to see movement, not a frozen window.
fn run_streaming(
    app: AppHandle,
    event: String,
    program: String,
    args: Vec<String>,
    cwd: Option<PathBuf>,
) -> Result<(), String> {
    let mut command = Command::new(&program);
    command
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in node_env() {
        command.env(key, value);
    }
    if let Some(dir) = cwd {
        if dir.exists() {
            command.current_dir(dir);
        }
    }

    let mut child = command
        .spawn()
        .map_err(|err| format!("failed to start {program}: {err}"))?;

    let mut readers = Vec::new();
    if let Some(handle) = child.stdout.take() {
        readers.push(spawn_reader(handle, app.clone(), event.clone(), "stdout"));
    }
    if let Some(handle) = child.stderr.take() {
        readers.push(spawn_reader(handle, app.clone(), event.clone(), "stderr"));
    }

    std::thread::spawn(move || {
        let status = child.wait();
        for reader in readers {
            let _ = reader.join();
        }
        let (success, code) = match status {
            Ok(status) => (status.success(), status.code().unwrap_or(-1)),
            Err(_) => (false, -1),
        };
        let _ = app.emit(&format!("{event}-done"), StreamDone { success, code });
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
fn detect_install() -> InstallState {
    let home = havnai_home();
    let runtime = runtime_dir();
    let python = venv_python();

    // The runtime counts as present only if the entry point is actually there.
    let has_runtime = runtime.join("client").join("client.py").exists();
    // venv_python() falls back to bare "python3" when the venv is absent, so a
    // fallback interpreter does not count as an installed environment.
    let has_venv = python.exists() && python != Path::new("python3");
    let version = std::fs::read_to_string(home.join("VERSION"))
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|_| "unknown".into());

    InstallState {
        installed: has_runtime && has_venv,
        havnai_home: home.to_string_lossy().to_string(),
        runtime_dir: runtime.to_string_lossy().to_string(),
        python: python.to_string_lossy().to_string(),
        version,
        has_runtime,
        has_venv,
        service_installed: service_file().exists(),
        platform: std::env::consts::OS.to_string(),
    }
}

#[tauri::command]
fn load_config() -> NodeConfig {
    let values = parse_env_file(&env_file());
    let get = |key: &str| values.get(key).cloned().unwrap_or_default();

    let server_url = if get("SERVER_URL").is_empty() {
        "https://api.joinhavn.io".to_string()
    } else {
        get("SERVER_URL")
    };
    let node_name = if get("NODE_NAME").is_empty() {
        std::env::var("HOSTNAME").unwrap_or_else(|_| "havnai-node".into())
    } else {
        get("NODE_NAME")
    };

    NodeConfig {
        server_url,
        join_token: get("JOIN_TOKEN"),
        wallet: get("WALLET"),
        node_name,
        creator_mode: get("CREATOR_MODE").to_lowercase() != "false",
    }
}

#[tauri::command]
fn save_config(config: NodeConfig) -> Result<(), String> {
    write_env_file(&config)
}

/// Run the preflight diagnostics and return the parsed report.
#[tauri::command]
fn run_doctor(offline: bool) -> Result<serde_json::Value, String> {
    let python = venv_python();
    let mut args = vec!["-m", "client.doctor", "--json"];
    if offline {
        args.push("--offline");
    }

    let output = run_capture(
        &python.to_string_lossy(),
        &args,
        Some(runtime_dir()),
    );

    // doctor exits non-zero when checks fail, which is a normal result, not an
    // error - only unparseable output means we genuinely could not run it.
    serde_json::from_str(&output.stdout).map_err(|err| {
        let detail = if output.stderr.trim().is_empty() {
            output.stdout.trim().to_string()
        } else {
            output.stderr.trim().to_string()
        };
        if detail.is_empty() {
            format!("could not run diagnostics: {err}")
        } else {
            format!("could not run diagnostics: {detail}")
        }
    })
}

/// Install or repair the node by running the coordinator's installer.
#[tauri::command]
fn install_node(app: AppHandle, config: NodeConfig, skip_models: bool) -> Result<(), String> {
    if cfg!(windows) {
        return Err("Automated install is not supported on Windows yet. Use WSL2.".into());
    }

    let server = config.server_url.trim_end_matches('/').to_string();
    let mut installer = format!(
        "curl -fsSL {server}/installers/install-node.sh | bash -s -- --server {server}"
    );
    if !config.join_token.is_empty() {
        installer.push_str(&format!(" --token {}", shell_quote(&config.join_token)));
    }
    if !config.wallet.is_empty() {
        installer.push_str(&format!(" --wallet {}", shell_quote(&config.wallet)));
    }
    if !config.node_name.is_empty() {
        installer.push_str(&format!(" --name {}", shell_quote(&config.node_name)));
    }
    installer.push_str(if config.creator_mode {
        " --creator"
    } else {
        " --no-creator"
    });
    if skip_models {
        installer.push_str(" --skip-models");
    }

    run_streaming(
        app,
        "install-output".into(),
        "bash".into(),
        vec!["-lc".into(), installer],
        Some(home_dir()),
    )
}

/// Download outstanding model weights, streaming JSON progress to the UI.
#[tauri::command]
fn fetch_models(app: AppHandle, face_assets: bool) -> Result<(), String> {
    let python = venv_python();
    let mut args = vec![
        "-m".to_string(),
        "client.fetch_models".to_string(),
        "--json".to_string(),
    ];
    if face_assets {
        args.push("--face-assets".to_string());
    }

    run_streaming(
        app,
        "models-output".into(),
        python.to_string_lossy().to_string(),
        args,
        Some(runtime_dir()),
    )
}

/// List the download plan without transferring anything.
#[tauri::command]
fn model_plan() -> Result<Vec<serde_json::Value>, String> {
    let python = venv_python();
    let output = run_capture(
        &python.to_string_lossy(),
        &["-m", "client.fetch_models", "--dry-run", "--json"],
        Some(runtime_dir()),
    );

    if output.stdout.trim().is_empty() {
        return Err(if output.stderr.trim().is_empty() {
            "no response from the model planner".into()
        } else {
            output.stderr.trim().to_string()
        });
    }

    Ok(output
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value.get("event").and_then(|e| e.as_str()) == Some("plan"))
        .collect())
}

fn service_command(action: &str) -> CommandOutput {
    if cfg!(target_os = "macos") {
        let plist = service_file().to_string_lossy().to_string();
        let args: Vec<&str> = match action {
            "start" => vec!["load", "-w", &plist],
            "stop" => vec!["unload", "-w", &plist],
            _ => vec!["list", "com.havnai.node"],
        };
        run_capture("launchctl", &args, None)
    } else {
        let args: Vec<&str> = match action {
            "start" => vec!["--user", "start", "havnai-node"],
            "stop" => vec!["--user", "stop", "havnai-node"],
            "restart" => vec!["--user", "restart", "havnai-node"],
            _ => vec!["--user", "is-active", "havnai-node"],
        };
        run_capture("systemctl", &args, None)
    }
}

#[tauri::command]
fn node_control(action: String) -> Result<CommandOutput, String> {
    match action.as_str() {
        "start" | "stop" | "restart" | "status" => Ok(service_command(&action)),
        other => Err(format!("unknown action: {other}")),
    }
}

#[tauri::command]
fn node_status() -> String {
    let output = service_command("status");
    let text = format!("{}{}", output.stdout, output.stderr);
    let text = text.trim();

    if cfg!(target_os = "macos") {
        // launchctl list prints a row for a loaded agent and errors otherwise.
        return if output.success && !text.is_empty() {
            "running".into()
        } else {
            "stopped".into()
        };
    }

    // systemctl prints its verdict on the last line, but on a machine without a
    // session bus it writes an error there instead. Only known states are
    // reported; anything else is "unknown" rather than raw diagnostic text.
    match text.lines().last().unwrap_or("").trim() {
        "active" => "running".into(),
        "activating" | "reloading" => "starting".into(),
        "inactive" | "deactivating" => "stopped".into(),
        "failed" => "failed".into(),
        _ => "unknown".into(),
    }
}

/// Return the tail of the node log for the activity view.
#[tauri::command]
fn read_logs(lines: usize) -> Result<Vec<String>, String> {
    let path = havnai_home().join("logs").join("node.log");
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return Ok(vec![format!("No log file yet at {}", path.display())]);
    };
    let all: Vec<&str> = contents.lines().collect();
    let start = all.len().saturating_sub(lines.clamp(1, 5000));
    Ok(all[start..].iter().map(|line| line.to_string()).collect())
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    Command::new(opener)
        .arg(&path)
        .spawn()
        .map(|_| ())
        .map_err(|err| format!("could not open {path}: {err}"))
}

/// Single-quote a value for safe interpolation into the installer command.
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            detect_install,
            load_config,
            save_config,
            run_doctor,
            install_node,
            fetch_models,
            model_plan,
            node_control,
            node_status,
            read_logs,
            open_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the HavnAI node desktop app");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_quote_escapes_embedded_single_quotes() {
        assert_eq!(shell_quote("plain"), "'plain'");
        // A token containing a quote must not be able to break out of the
        // argument and inject a second command into the installer line.
        assert_eq!(shell_quote("a'b"), r"'a'\''b'");
        assert_eq!(shell_quote("'; rm -rf /; '"), r"''\''; rm -rf /; '\'''");
    }

    #[test]
    fn parse_env_file_ignores_comments_and_blanks() {
        let dir = std::env::temp_dir().join("havnai-env-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".env");
        std::fs::write(
            &path,
            "# comment\n\nSERVER_URL=https://example.test\nWALLET = 0xABC \nBROKEN\n",
        )
        .unwrap();

        let values = parse_env_file(&path);
        assert_eq!(
            values.get("SERVER_URL").map(String::as_str),
            Some("https://example.test")
        );
        assert_eq!(values.get("WALLET").map(String::as_str), Some("0xABC"));
        assert!(!values.contains_key("BROKEN"));
        let _ = std::fs::remove_file(&path);
    }
}
