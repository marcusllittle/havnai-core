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

/// The coordinator a fresh install points at. This is the address the website's
/// install command uses; the bare `api.` host does not answer.
const DEFAULT_SERVER_URL: &str = "https://joinhavn.io/api";

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

fn staging_dir() -> PathBuf {
    havnai_home().join(".staging")
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
    if cfg!(windows) {
        havnai_home().join("bin").join("havnai-node.cmd")
    } else if cfg!(target_os = "macos") {
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

/// Environment variables an AppImage runtime rewrites to point inside its own
/// bundle. Inherited by a child process they are actively harmful: a spawned
/// `python3` picks up the AppImage's `PYTHONHOME` and dies with
/// "No module named 'encodings'" before it runs a line of our code.
const APPIMAGE_HIJACKED_VARS: &[&str] = &[
    "PYTHONHOME",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PERLLIB",
    "XDG_DATA_DIRS",
    "GSETTINGS_SCHEMA_DIR",
    "GI_TYPELIB_PATH",
    "GST_PLUGIN_SYSTEM_PATH",
    "GST_PLUGIN_SYSTEM_PATH_1_0",
    "QT_PLUGIN_PATH",
];

/// Undo the AppImage runtime's environment rewriting for a child process.
///
/// AppRun saves whatever it overrode as `<VAR>_ORIG`, so restore from that when
/// present and otherwise drop the variable entirely. Outside an AppImage this
/// is a no-op, because none of these carry an `_ORIG` twin and the originals
/// are inherited unchanged.
fn restore_host_environment(command: &mut Command) {
    if std::env::var_os("APPDIR").is_none() {
        return;
    }
    for name in APPIMAGE_HIJACKED_VARS {
        match std::env::var_os(format!("{name}_ORIG")) {
            Some(original) => {
                command.env(name, original);
            }
            None => {
                command.env_remove(name);
            }
        }
    }
}

fn windows_pid_file() -> PathBuf {
    havnai_home().join("havnai-node.pid")
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
    // The node prints emoji; on Windows a piped stdout otherwise defaults to
    // cp1252 and the child dies with UnicodeEncodeError.
    env.insert("PYTHONUTF8".into(), "1".into());
    env
}

// ---------------------------------------------------------------------------
// Process helpers
// ---------------------------------------------------------------------------

/// Stop every child process from flashing up a console window on Windows.
///
/// The release build runs in the GUI subsystem, so each `python`, `pip` or
/// `curl` it spawns would otherwise open a console of its own.
fn hide_console(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = command;
}

fn run_capture(program: &str, args: &[&str], cwd: Option<PathBuf>) -> CommandOutput {
    let mut command = Command::new(program);
    command.args(args);
    hide_console(&mut command);
    restore_host_environment(&mut command);
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
        forward_lines(handle, |line| {
            let _ = app.emit(
                &event,
                StreamLine {
                    stream: label.to_string(),
                    line,
                },
            );
        });
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
    hide_console(&mut command);
    restore_host_environment(&mut command);
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
// Native Windows install
// ---------------------------------------------------------------------------

/// Where install progress goes: the app forwards it to the UI, tests collect it.
type InstallSink<'a> = &'a (dyn Fn(&str, String) + Sync);

/// PyPI's Windows torch wheels are CPU-only (the Linux ones bundle CUDA), so on
/// a machine with an NVIDIA driver torch has to come from PyTorch's own index.
const TORCH_CUDA_INDEX: &str = "https://download.pytorch.org/whl/cu128";

/// Requirements that cannot be part of the required set on Windows. triton
/// publishes no Windows wheels, xformers pins its own torch build and would
/// replace the CUDA one, and insightface ships only as source, so it needs
/// Microsoft's C++ Build Tools. Any of them would fail the whole install.
const WINDOWS_OPTIONAL_PACKAGES: &[&str] = &["triton", "xformers", "insightface"];

/// Optional packages worth attempting after the required set is in, with what
/// the operator loses if one does not install.
const WINDOWS_BEST_EFFORT: &[(&str, &str)] = &[(
    "insightface",
    "face swap stays unavailable until Microsoft C++ Build Tools are installed; image generation is unaffected",
)];

/// Python versions every required package publishes Windows wheels for.
const WINDOWS_PYTHON_CHECK: &str =
    "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)";

fn emit_install_done(app: &AppHandle, success: bool, code: i32) {
    let _ = app.emit("install-output-done", StreamDone { success, code });
}

fn run_plain_capture(program: &str, args: &[&str], cwd: Option<&Path>) -> CommandOutput {
    let mut command = Command::new(program);
    command.args(args);
    hide_console(&mut command);
    if let Some(dir) = cwd {
        command.current_dir(dir);
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

fn windows_python_command() -> Result<(String, Vec<String>, String), String> {
    // The py launcher is preferred because it finds installs that were never
    // added to PATH; bare `python` may be the Microsoft Store stub.
    let candidates: Vec<(&str, Vec<&str>)> = vec![
        ("py", vec!["-3.12"]),
        ("py", vec!["-3.11"]),
        ("py", vec!["-3.13"]),
        ("py", vec!["-3.10"]),
        ("python", vec![]),
        ("python3", vec![]),
    ];

    for (program, prefix) in candidates {
        let mut args = prefix.clone();
        args.extend(["-c", WINDOWS_PYTHON_CHECK]);
        if run_plain_capture(program, &args, None).success {
            let mut version_args = prefix.clone();
            version_args.extend(["-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"]);
            let version = run_plain_capture(program, &version_args, None)
                .stdout
                .trim()
                .to_string();
            return Ok((
                program.to_string(),
                prefix.into_iter().map(str::to_string).collect(),
                version,
            ));
        }
    }

    Err("Python 3.10 to 3.13 was not found. Install Python 3.12 from python.org, tick 'Add python.exe to PATH', then click Install node again.".into())
}

fn has_nvidia_gpu() -> bool {
    run_plain_capture("nvidia-smi", &["-L"], None).success
}

/// The normalised package name a requirements line installs, if any.
fn requirement_name(line: &str) -> Option<String> {
    let line = line.split('#').next().unwrap_or("").trim();
    if line.is_empty() || line.starts_with('-') {
        return None;
    }
    let end = line
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
        .unwrap_or(line.len());
    Some(line[..end].to_ascii_lowercase().replace('_', "-"))
}

/// The node's requirements with the Windows-optional packages removed.
fn windows_requirements(contents: &str) -> String {
    contents
        .lines()
        .filter(|line| {
            requirement_name(line)
                .map(|name| !WINDOWS_OPTIONAL_PACKAGES.contains(&name.as_str()))
                .unwrap_or(true)
        })
        .map(|line| format!("{line}\n"))
        .collect()
}

/// Forward a child's output pipe line by line, tolerating non-UTF-8 bytes.
///
/// Windows tools write in the console code page, and `BufRead::lines` stops at
/// the first invalid byte - which would stop draining the pipe and leave the
/// child blocked on a full buffer.
fn forward_lines<R: std::io::Read>(handle: R, mut forward: impl FnMut(String)) {
    let mut reader = BufReader::new(handle);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let line = String::from_utf8_lossy(&buf);
                let line = line.trim_end();
                if !line.trim().is_empty() {
                    forward(line.to_string());
                }
            }
        }
    }
}

/// Run one install step, streaming its output as it arrives.
///
/// Resolving and downloading CUDA torch takes minutes; buffering until exit
/// would leave the operator watching a console that looks frozen.
fn run_install_step(
    emit: InstallSink,
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    label: &str,
) -> Result<(), String> {
    emit("stdout", label.to_string());
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONUTF8", "1");
    hide_console(&mut command);
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }

    let step = label.trim();
    let mut child = command
        .spawn()
        .map_err(|err| format!("{step}: could not start {program}: {err}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    std::thread::scope(|scope| {
        if let Some(handle) = stdout {
            scope.spawn(move || forward_lines(handle, |line| emit("stdout", line)));
        }
        if let Some(handle) = stderr {
            scope.spawn(move || forward_lines(handle, |line| emit("stderr", line)));
        }
    });

    let status = child.wait().map_err(|err| format!("{step}: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{step} failed with exit code {}", status.code().unwrap_or(-1)))
    }
}

/// A `.cmd` launcher that runs a node module the way the Linux systemd unit does.
///
/// The unit loads `.env` with `EnvironmentFile=`, and the client depends on
/// that: its `SERVER_URL` comes from the process environment, falling back to a
/// hard-coded host rather than to `.env`. So the launcher exports every `.env`
/// line before starting Python. PYTHONUTF8 because the node logs emoji, which
/// the default cp1252 encoding cannot represent.
fn windows_launcher_body(home: &Path, runtime: &Path, python: &Path, module: &str) -> String {
    let env = home.join(".env");
    format!(
        "@echo off\r\n\
         set \"HAVNAI_HOME={home}\"\r\n\
         if exist \"{env}\" for /f \"usebackq eol=# tokens=1,* delims==\" %%A in (\"{env}\") do set \"%%A=%%B\"\r\n\
         set \"PYTHONUTF8=1\"\r\n\
         cd /d \"{runtime}\"\r\n\
         \"{python}\" -m {module} %*\r\n",
        home = home.display(),
        env = env.display(),
        runtime = runtime.display(),
        python = python.display(),
    )
}

fn write_windows_launcher(path: &Path, module: &str) -> Result<(), String> {
    let body = windows_launcher_body(&havnai_home(), &runtime_dir(), &venv_python(), module);
    std::fs::write(path, body).map_err(|err| err.to_string())
}

/// `pip install` with a wheel preferred over a newer source release: building
/// from source on Windows needs Microsoft's C++ Build Tools, which operators
/// generally do not have (albumentations' stringzilla dependency hit this).
fn pip_args(extra: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = ["-m", "pip", "install", "--prefer-binary", "--progress-bar", "off"]
        .into_iter()
        .map(String::from)
        .collect();
    args.extend(extra.iter().map(|arg| arg.to_string()));
    args
}

/// Install the node runtime natively under `%USERPROFILE%\.havnai`.
///
/// Mirrors what `install-node.sh` does on Linux and macOS: download the runtime
/// bundle, build a virtualenv, write `.env`, and create launchers.
fn install_windows_runtime(
    config: &NodeConfig,
    skip_models: bool,
    emit: InstallSink,
) -> Result<(), String> {
    let say = |line: String| emit("stdout", line);
    let path = |p: &Path| p.to_string_lossy().to_string();

    let server = config.server_url.trim_end_matches('/').to_string();
    let home = havnai_home();
    let staging = staging_dir();
    let staging_runtime = staging.join("runtime");
    let bundle_path = staging.join("runtime.tar.gz");
    let venv = home.join("venv");
    let bin = home.join("bin");

    say("[1/7] Preparing install folders".into());
    for dir in [
        home.clone(),
        bin.clone(),
        home.join("logs"),
        home.join("models").join("creator"),
        home.join("outputs"),
        home.join("loras"),
    ] {
        std::fs::create_dir_all(dir).map_err(|err| err.to_string())?;
    }
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging_runtime).map_err(|err| err.to_string())?;

    say("[2/7] Finding Python".into());
    let (python_program, python_prefix, python_version) = windows_python_command()?;
    say(format!("      Python {python_version} found"));

    say("[3/7] Downloading node runtime".into());
    run_install_step(
        emit,
        "curl.exe",
        &[
            "-fsSL".into(),
            "--retry".into(),
            "3".into(),
            format!("{server}/client/bundle.tar.gz"),
            "-o".into(),
            path(&bundle_path),
        ],
        None,
        "      Downloading runtime bundle",
    )?;
    let bytes = std::fs::metadata(&bundle_path)
        .map_err(|err| err.to_string())?
        .len();
    if bytes < 1024 {
        return Err(format!("runtime bundle looks truncated ({bytes} bytes)"));
    }
    say(format!("      runtime bundle: {bytes} bytes"));
    run_install_step(
        emit,
        "tar.exe",
        &[
            "-xzf".into(),
            path(&bundle_path),
            "-C".into(),
            path(&staging_runtime),
        ],
        None,
        "      Extracting runtime bundle",
    )?;
    for required in [
        "client/client.py",
        "client/doctor.py",
        "client/fetch_models.py",
        "client/requirements-node.txt",
    ] {
        if !staging_runtime.join(required).is_file() {
            return Err(format!("runtime bundle is incomplete: missing {required}"));
        }
    }

    say("[4/7] Installing Python environment".into());
    if venv.exists() {
        say("      Reusing existing Python environment".into());
    } else {
        let mut args = python_prefix.clone();
        args.extend(["-m".into(), "venv".into(), path(&venv)]);
        run_install_step(emit, &python_program, &args, None, "      Creating Python environment")?;
    }
    let venv_py = venv_python();
    if !venv_py.is_file() {
        return Err(format!("virtualenv is missing {}", venv_py.display()));
    }
    let py = path(&venv_py);
    run_install_step(
        emit,
        &py,
        &pip_args(&["--upgrade", "pip", "wheel"]),
        None,
        "      Updating pip",
    )?;
    if has_nvidia_gpu() {
        run_install_step(
            emit,
            &py,
            &pip_args(&["torch", "--index-url", TORCH_CUDA_INDEX]),
            None,
            "      Installing PyTorch with CUDA (about 3 GB, this takes a while)",
        )?;
    } else {
        emit(
            "stderr",
            "      No NVIDIA GPU detected - installing CPU-only PyTorch. The node needs an NVIDIA GPU to serve jobs.".into(),
        );
    }
    let requirements = std::fs::read_to_string(
        staging_runtime.join("client").join("requirements-node.txt"),
    )
    .map_err(|err| err.to_string())?;
    let windows_reqs = staging.join("requirements-windows.txt");
    std::fs::write(&windows_reqs, windows_requirements(&requirements))
        .map_err(|err| err.to_string())?;
    run_install_step(
        emit,
        &py,
        &pip_args(&["-r", &path(&windows_reqs)]),
        None,
        "      Installing node dependencies",
    )?;
    for (package, consequence) in WINDOWS_BEST_EFFORT {
        let step = format!("      Installing optional {package}");
        if run_install_step(emit, &py, &pip_args(&[package]), None, &step).is_err() {
            emit("stderr", format!("      {package} did not install: {consequence}."));
        }
    }

    say("[5/7] Activating runtime".into());
    // A running node holds files open under current\, and Windows refuses to
    // rename a directory while it does.
    if windows_node_command("status").success {
        say("      Stopping the running node first".into());
        let _ = windows_node_command("stop");
    }
    let runtime = runtime_dir();
    let previous = home.join("previous");
    if runtime.exists() {
        let _ = std::fs::remove_dir_all(&previous);
        std::fs::rename(&runtime, &previous).map_err(|err| {
            format!("could not replace the existing runtime (is a node still running?): {err}")
        })?;
    }
    std::fs::rename(&staging_runtime, &runtime).map_err(|err| err.to_string())?;
    if run_install_step(
        emit,
        "curl.exe",
        &[
            "-fsSL".into(),
            format!("{server}/client/version"),
            "-o".into(),
            path(&home.join("VERSION")),
        ],
        None,
        "      Fetching runtime version",
    )
    .is_err()
    {
        let _ = std::fs::write(home.join("VERSION"), "unknown");
    }

    say("[6/7] Writing configuration and launchers".into());
    write_env_file(config)?;
    let mut values = parse_env_file(&env_file());
    values.insert("HAVNAI_HOME".into(), path(&home));
    values.insert("HAVNAI_OUTPUTS_DIR".into(), path(&home.join("outputs")));
    let mut keys: Vec<&String> = values.keys().collect();
    keys.sort();
    let body: String = keys
        .iter()
        .map(|key| format!("{}={}\n", key, values[*key]))
        .collect();
    std::fs::write(env_file(), body).map_err(|err| err.to_string())?;
    write_windows_launcher(&bin.join("havnai-node.cmd"), "client.client")?;
    write_windows_launcher(&bin.join("havnai-doctor.cmd"), "client.doctor")?;
    write_windows_launcher(&bin.join("havnai-fetch-models.cmd"), "client.fetch_models")?;

    say("[7/7] Checking models and preflight".into());
    if skip_models {
        say("      Skipped model download".into());
    } else if config.creator_mode {
        if let Err(err) = run_install_step(
            emit,
            &py,
            &["-m".into(), "client.fetch_models".into(), "--face-assets".into()],
            Some(&runtime),
            "      Downloading model weights",
        ) {
            emit("stderr", format!("      {err}; retry from the Models tab."));
        }
    } else {
        say("      Worker mode selected; no model weights required".into());
    }
    // doctor exits non-zero when a check fails, which is a result to show, not
    // an install failure.
    if let Err(err) = run_install_step(
        emit,
        &py,
        &["-m".into(), "client.doctor".into()],
        Some(&runtime),
        "      Running preflight",
    ) {
        emit("stderr", format!("      {err}; see the Health tab."));
    }

    let _ = std::fs::remove_dir_all(&staging);
    say("Windows install complete. Use Start node when preflight is ready.".into());
    Ok(())
}

fn install_node_windows(app: AppHandle, config: NodeConfig, skip_models: bool) -> Result<(), String> {
    std::thread::spawn(move || {
        let emit = |stream: &str, line: String| {
            let _ = app.emit(
                "install-output",
                StreamLine {
                    stream: stream.into(),
                    line,
                },
            );
        };
        match install_windows_runtime(&config, skip_models, &emit) {
            Ok(()) => emit_install_done(&app, true, 0),
            Err(err) => {
                emit("stderr", err);
                emit_install_done(&app, false, 1);
            }
        }
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
        DEFAULT_SERVER_URL.to_string()
    } else {
        get("SERVER_URL")
    };
    let node_name = if get("NODE_NAME").is_empty() {
        std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "havnai-node".into())
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
        return install_node_windows(app, config, skip_models);
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
    if cfg!(windows) {
        windows_node_command(action)
    } else if cfg!(target_os = "macos") {
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

fn windows_node_command(action: &str) -> CommandOutput {
    let runner = service_file();
    let pid_file = windows_pid_file();

    match action {
        "start" | "restart" => {
            if action == "restart" {
                let _ = windows_node_command("stop");
            }
            if !runner.is_file() {
                return CommandOutput {
                    success: false,
                    code: 1,
                    stdout: String::new(),
                    stderr: "node runtime is not installed yet".into(),
                };
            }
            let script = format!(
                "$p = Start-Process -FilePath '{}' -WorkingDirectory '{}' -WindowStyle Hidden -PassThru; Set-Content -Path '{}' -Value $p.Id",
                runner.to_string_lossy().replace('\'', "''"),
                runtime_dir().to_string_lossy().replace('\'', "''"),
                pid_file.to_string_lossy().replace('\'', "''")
            );
            run_plain_capture(
                "powershell",
                &["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &script],
                None,
            )
        }
        "stop" => {
            let Ok(pid) = std::fs::read_to_string(&pid_file) else {
                return CommandOutput {
                    success: true,
                    code: 0,
                    stdout: "not running".into(),
                    stderr: String::new(),
                };
            };
            let pid = pid.trim();
            let output = run_plain_capture("taskkill", &["/PID", pid, "/T", "/F"], None);
            let _ = std::fs::remove_file(&pid_file);
            output
        }
        _ => {
            let Ok(pid) = std::fs::read_to_string(&pid_file) else {
                return CommandOutput {
                    success: false,
                    code: 1,
                    stdout: "stopped".into(),
                    stderr: String::new(),
                };
            };
            let pid = pid.trim();
            if pid.is_empty() {
                return CommandOutput {
                    success: false,
                    code: 1,
                    stdout: "stopped".into(),
                    stderr: String::new(),
                };
            }
            let filter = format!("PID eq {pid}");
            let output = run_plain_capture("tasklist", &["/FI", &filter], None);
            let running = output.success && output.stdout.lines().any(|line| line.contains(pid));
            CommandOutput {
                success: running,
                code: if running { 0 } else { 1 },
                stdout: if running { "running".into() } else { "stopped".into() },
                stderr: String::new(),
            }
        }
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

    if cfg!(windows) {
        return if output.success && text.contains("running") {
            "running".into()
        } else {
            "stopped".into()
        };
    }

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
    fn appimage_hijacked_python_vars_do_not_reach_children() {
        // Simulates the AppImage runtime: APPDIR set, PYTHONHOME pointing into
        // the bundle, and an _ORIG twin holding the host's real value.
        unsafe {
            std::env::set_var("APPDIR", "/tmp/appimage_extracted_test");
            std::env::set_var("PYTHONHOME", "/tmp/appimage_extracted_test/usr");
            std::env::set_var("LD_LIBRARY_PATH", "/tmp/appimage_extracted_test/usr/lib");
            std::env::set_var("LD_LIBRARY_PATH_ORIG", "/usr/lib/host");
        }

        let mut command = Command::new("python3");
        restore_host_environment(&mut command);

        // PYTHONHOME has no _ORIG twin, so it must be dropped outright;
        // LD_LIBRARY_PATH must be restored to the host value, not the bundle's.
        let overrides: Vec<(String, Option<String>)> = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().to_string(),
                    value.map(|v| v.to_string_lossy().to_string()),
                )
            })
            .collect();

        let python_home = overrides.iter().find(|(key, _)| key == "PYTHONHOME");
        assert_eq!(
            python_home.map(|(_, value)| value.clone()),
            Some(None),
            "PYTHONHOME must be removed for child processes"
        );

        let ld_path = overrides.iter().find(|(key, _)| key == "LD_LIBRARY_PATH");
        assert_eq!(
            ld_path.and_then(|(_, value)| value.clone()),
            Some("/usr/lib/host".to_string()),
            "LD_LIBRARY_PATH must be restored from its _ORIG twin"
        );

        unsafe {
            std::env::remove_var("APPDIR");
            std::env::remove_var("PYTHONHOME");
            std::env::remove_var("LD_LIBRARY_PATH");
            std::env::remove_var("LD_LIBRARY_PATH_ORIG");
        }
    }

    #[test]
    fn windows_requirements_drop_only_packages_without_windows_wheels() {
        let input = "# GPU\ntorch\nTriton>=2.0\nxformers ; sys_platform == 'linux'\n\
                     insightface==0.7.3\ninsightface_extra\n-r base.txt\nalbumentations<2\n";
        let output = windows_requirements(input);
        let kept: Vec<&str> = output.lines().collect();
        assert_eq!(
            kept,
            vec!["# GPU", "torch", "insightface_extra", "-r base.txt", "albumentations<2"]
        );
    }

    #[test]
    fn requirement_name_normalises_and_skips_options() {
        assert_eq!(requirement_name("Opencv_Python-Headless>=4"), Some("opencv-python-headless".into()));
        assert_eq!(requirement_name("diffusers>=0.32.0  # pipelines"), Some("diffusers".into()));
        assert_eq!(requirement_name("--index-url https://example.test"), None);
        assert_eq!(requirement_name("   # only a comment"), None);
    }

    #[test]
    fn forward_lines_survives_bytes_that_are_not_utf8() {
        // cp1252 output from a Windows tool: 0x92 is a curly apostrophe there
        // and invalid UTF-8. Reading must continue past it, not stop.
        let input: &[u8] = b"first\r\nit\x92s cp1252\n\nlast";
        let mut lines = Vec::new();
        forward_lines(input, |line| lines.push(line));
        assert_eq!(lines, vec!["first", "it\u{FFFD}s cp1252", "last"]);
    }

    /// The launcher must hand `.env` to the node the way systemd's
    /// EnvironmentFile does, or a Windows node ignores its configured
    /// coordinator. Runs the generated .cmd for real with a probe module.
    #[cfg(windows)]
    #[test]
    fn windows_launcher_exports_env_file_to_the_node() {
        let Ok((program, prefix, _)) = windows_python_command() else {
            eprintln!("skipping: no Python 3.10-3.13 on this machine");
            return;
        };
        let mut args = prefix;
        args.extend(["-c".into(), "import sys; print(sys.executable)".into()]);
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let python = PathBuf::from(run_plain_capture(&program, &arg_refs, None).stdout.trim());

        let home = std::env::temp_dir().join("havnai-launcher-test");
        let runtime = home.join("current");
        std::fs::create_dir_all(&runtime).unwrap();
        std::fs::write(
            home.join(".env"),
            "# comment\nSERVER_URL=https://joinhavn.io/api\nWALLET=0xabc\nJOIN_TOKEN=\n",
        )
        .unwrap();
        std::fs::write(
            runtime.join("envprobe.py"),
            "import os\nfor k in ('SERVER_URL', 'WALLET', 'PYTHONUTF8', 'HAVNAI_HOME'):\n    print(k, os.environ.get(k))\n",
        )
        .unwrap();
        let launcher = home.join("probe.cmd");
        std::fs::write(&launcher, windows_launcher_body(&home, &runtime, &python, "envprobe")).unwrap();

        let output = run_plain_capture("cmd", &["/c", &launcher.to_string_lossy()], None);
        assert!(output.success, "launcher failed: {}", output.stderr);
        assert!(output.stdout.contains("SERVER_URL https://joinhavn.io/api"), "{}", output.stdout);
        assert!(output.stdout.contains("WALLET 0xabc"), "{}", output.stdout);
        assert!(output.stdout.contains("PYTHONUTF8 1"), "{}", output.stdout);
        assert!(output.stdout.contains(&format!("HAVNAI_HOME {}", home.display())), "{}", output.stdout);
        let _ = std::fs::remove_dir_all(&home);
    }

    /// Runs the real native install against a live coordinator. Slow (it pulls
    /// CUDA torch), so it only runs on request:
    ///
    /// ```text
    /// set HAVNAI_E2E_HOME=C:\some\empty\dir
    /// cargo test windows_install_end_to_end -- --ignored --nocapture
    /// ```
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn windows_install_end_to_end() {
        let home = std::env::var("HAVNAI_E2E_HOME").expect("set HAVNAI_E2E_HOME");
        unsafe {
            std::env::set_var("HAVNAI_HOME", &home);
        }
        let config = NodeConfig {
            server_url: std::env::var("HAVNAI_E2E_SERVER")
                .unwrap_or_else(|_| DEFAULT_SERVER_URL.into()),
            join_token: String::new(),
            wallet: String::new(),
            node_name: "desktop-e2e".into(),
            creator_mode: false,
        };
        let emit = |stream: &str, line: String| println!("[{stream}] {line}");
        install_windows_runtime(&config, true, &emit).expect("install succeeds");

        let home = PathBuf::from(home);
        assert!(home.join("venv").join("Scripts").join("python.exe").is_file());
        assert!(home.join("current").join("client").join("client.py").is_file());
        assert!(home.join("bin").join("havnai-node.cmd").is_file());
        let env = parse_env_file(&home.join(".env"));
        assert_eq!(env.get("NODE_NAME").map(String::as_str), Some("desktop-e2e"));
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
