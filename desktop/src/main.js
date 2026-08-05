/**
 * HavnAI Node desktop control panel.
 *
 * All real work happens in the Python runtime; this layer invokes the Rust
 * commands, renders their results, and streams long-running output so the
 * operator can see an install or a multi-gigabyte download actually moving.
 */

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const FEATURE_LABELS = {
  core: "Core runtime",
  image: "Image generation",
  face_swap: "Face swap",
  video: "Video generation",
};

const CAPABILITY_ORDER = ["image", "face_swap", "video"];

const $ = (id) => document.getElementById(id);

/** Progress state per model name, so re-renders keep their bars. */
const modelProgress = new Map();
let statusTimer = null;
let logTimer = null;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("is-error", isError);
  el.classList.add("is-visible");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => el.classList.remove("is-visible"), 4000);
}

function humanBytes(bytes) {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

function appendConsole(el, text, isError = false) {
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  const span = document.createElement("span");
  if (isError) span.className = "err";
  span.textContent = `${text}\n`;
  el.appendChild(span);
  // Only follow the tail if the operator has not scrolled up to read something.
  if (atBottom) el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("is-active"));
    tab.classList.add("is-active");
    $(`panel-${tab.dataset.tab}`).classList.add("is-active");
    if (tab.dataset.tab === "logs") refreshLogs();
    if (tab.dataset.tab === "models") refreshPlan();
  });
});

// ---------------------------------------------------------------------------
// Install state + node status
// ---------------------------------------------------------------------------

async function refreshInstallState() {
  try {
    const state = await invoke("detect_install");
    const path = $("install-path");
    if (state.installed) {
      path.textContent = `${state.runtime_dir} · v${state.version} · ${state.platform}`;
    } else if (state.has_venv || state.has_runtime) {
      path.textContent = `${state.havnai_home} · incomplete install`;
    } else {
      path.textContent = `No node installed — open Setup and click Install node`;
    }
    if (state.platform === "windows") {
      $("setup-intro").textContent =
        "Enter the coordinator details, then install. On Windows, the app downloads the runtime, prepares Python, and creates Start / Stop controls automatically.";
    }
    return state;
  } catch (err) {
    $("install-path").textContent = `Could not inspect install: ${err}`;
    return null;
  }
}

async function refreshStatus() {
  try {
    const status = await invoke("node_status");
    const pill = $("node-status");
    pill.textContent = status;
    pill.className = `pill pill-${
      ["running", "starting", "stopped", "inactive", "failed"].includes(status)
        ? status
        : "unknown"
    }`;
    $("btn-start").disabled = status === "running";
    $("btn-stop").disabled = status !== "running";
  } catch {
    /* status polling is best-effort */
  }
}

$("btn-start").addEventListener("click", async () => {
  try {
    const result = await invoke("node_control", { action: "start" });
    toast(result.success ? "Node starting…" : `Could not start: ${result.stderr.trim()}`, !result.success);
    setTimeout(refreshStatus, 1200);
  } catch (err) {
    toast(String(err), true);
  }
});

$("btn-stop").addEventListener("click", async () => {
  try {
    const result = await invoke("node_control", { action: "stop" });
    toast(result.success ? "Node stopped." : `Could not stop: ${result.stderr.trim()}`, !result.success);
    setTimeout(refreshStatus, 1200);
  } catch (err) {
    toast(String(err), true);
  }
});

// ---------------------------------------------------------------------------
// Health / doctor
// ---------------------------------------------------------------------------

function renderCapabilities(report) {
  const grid = $("capability-grid");
  grid.innerHTML = "";
  CAPABILITY_ORDER.forEach((feature) => {
    const ready = Boolean(report.features?.[feature]);
    const card = document.createElement("div");
    card.className = `capability ${ready ? "is-ready" : "is-blocked"}`;
    card.innerHTML = `
      <div class="name"></div>
      <div class="state"></div>
    `;
    card.querySelector(".name").textContent = FEATURE_LABELS[feature];
    card.querySelector(".state").textContent = ready ? "Ready" : "Not ready";
    grid.appendChild(card);
  });
}

function renderChecks(report) {
  const container = $("doctor-results");
  container.innerHTML = "";

  const grouped = new Map();
  (report.checks || []).forEach((check) => {
    const feature = check.feature || "core";
    if (!grouped.has(feature)) grouped.set(feature, []);
    grouped.get(feature).push(check);
  });

  ["core", ...CAPABILITY_ORDER].forEach((feature) => {
    const checks = grouped.get(feature);
    if (!checks || !checks.length) return;

    const group = document.createElement("div");
    group.className = "check-group";
    const heading = document.createElement("h3");
    heading.textContent = FEATURE_LABELS[feature];
    group.appendChild(heading);

    checks.forEach((check) => {
      const row = document.createElement("div");
      row.className = "check";

      const badge = document.createElement("div");
      badge.className = `badge badge-${check.status}`;
      badge.textContent = check.status;

      const body = document.createElement("div");
      const label = document.createElement("div");
      label.className = "label";
      label.textContent = check.label;
      const detail = document.createElement("div");
      detail.className = "detail";
      detail.textContent = check.detail || "";
      body.append(label, detail);

      if (check.remedy && check.status !== "ok") {
        const remedy = document.createElement("div");
        remedy.className = "remedy";
        remedy.textContent = check.remedy;
        body.appendChild(remedy);
      }

      row.append(badge, body);
      group.appendChild(row);
    });

    container.appendChild(group);
  });
}

async function runDoctor() {
  const button = $("btn-doctor");
  button.disabled = true;
  button.textContent = "Checking…";
  $("doctor-results").innerHTML = '<p class="muted">Running diagnostics…</p>';

  try {
    const report = await invoke("run_doctor", { offline: false });
    renderCapabilities(report);
    renderChecks(report);
    if (!report.healthy) {
      const blocking = (report.checks || []).filter((c) => c.status === "fail").length;
      toast(`${blocking} blocking issue${blocking === 1 ? "" : "s"} found.`, true);
    }
  } catch (err) {
    $("capability-grid").innerHTML = "";
    $("doctor-results").innerHTML = "";
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent =
      `${err}\n\nIf no node is installed yet, use the Setup tab to install one.`;
    $("doctor-results").appendChild(message);
  } finally {
    button.disabled = false;
    button.textContent = "Re-run checks";
  }
}

$("btn-doctor").addEventListener("click", runDoctor);

// ---------------------------------------------------------------------------
// Configuration + install
// ---------------------------------------------------------------------------

async function loadConfig() {
  try {
    const config = await invoke("load_config");
    $("cfg-server").value = config.server_url;
    $("cfg-token").value = config.join_token;
    $("cfg-wallet").value = config.wallet;
    $("cfg-name").value = config.node_name;
    $("cfg-creator").checked = config.creator_mode;
  } catch (err) {
    toast(`Could not read configuration: ${err}`, true);
  }
}

function currentConfig() {
  return {
    serverUrl: $("cfg-server").value.trim() || "https://api.joinhavn.io",
    joinToken: $("cfg-token").value.trim(),
    wallet: $("cfg-wallet").value.trim(),
    nodeName: $("cfg-name").value.trim(),
    creatorMode: $("cfg-creator").checked,
  };
}

/** Rust expects snake_case field names on the NodeConfig struct. */
function toRustConfig(config) {
  return {
    server_url: config.serverUrl,
    join_token: config.joinToken,
    wallet: config.wallet,
    node_name: config.nodeName,
    creator_mode: config.creatorMode,
  };
}

$("config-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await invoke("save_config", { config: toRustConfig(currentConfig()) });
    toast("Configuration saved. Restart the node to apply.");
  } catch (err) {
    toast(`Could not save: ${err}`, true);
  }
});

$("btn-clear-install").addEventListener("click", () => {
  $("install-console").textContent = "";
});

$("btn-install").addEventListener("click", async () => {
  const button = $("btn-install");
  button.disabled = true;
  button.textContent = "Installing…";
  $("install-console").textContent = "";

  try {
    // Persist first so a mid-install crash does not lose the operator's input.
    await invoke("save_config", { config: toRustConfig(currentConfig()) });
    await invoke("install_node", {
      config: toRustConfig(currentConfig()),
      skipModels: $("install-skip-models").checked,
    });
    appendConsole($("install-console"), "Starting installer…");
  } catch (err) {
    appendConsole($("install-console"), String(err), true);
    toast(String(err), true);
    button.disabled = false;
    button.textContent = "Install node";
  }
});

listen("install-output", (event) => {
  const { stream, line } = event.payload;
  appendConsole($("install-console"), line, stream === "stderr");
});

listen("install-output-done", async (event) => {
  const { success, code } = event.payload;
  appendConsole(
    $("install-console"),
    success ? "\nInstall finished." : `\nInstaller exited with code ${code}.`,
    !success
  );
  $("btn-install").disabled = false;
  $("btn-install").textContent = "Install node";
  toast(success ? "Install complete." : "Install finished with issues — see output.", !success);
  await refreshInstallState();
  await runDoctor();
});

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

function renderModelRow(entry) {
  const name = entry.model;
  const row = document.createElement("div");
  row.className = "model";
  row.dataset.model = name;

  const top = document.createElement("div");
  top.className = "model-top";

  const left = document.createElement("div");
  const title = document.createElement("div");
  title.className = "name";
  title.textContent = name;
  const meta = document.createElement("div");
  meta.className = "meta";
  const size = entry.size_bytes ? ` · ${humanBytes(entry.size_bytes)}` : "";
  meta.textContent = `${entry.kind}${size}`;
  left.append(title, meta);

  const state = document.createElement("span");
  state.className = `state state-${entry.state}`;
  state.textContent =
    entry.state === "manual" ? "supply manually" : entry.state === "fetch" ? "missing" : entry.state;

  top.append(left, state);
  row.appendChild(top);

  const progress = document.createElement("div");
  progress.className = "progress";
  const bar = document.createElement("div");
  bar.className = "progress-bar";
  progress.appendChild(bar);
  row.appendChild(progress);

  const saved = modelProgress.get(name);
  if (saved !== undefined) bar.style.width = `${saved}%`;

  return row;
}

async function refreshPlan() {
  const list = $("model-list");
  try {
    const plan = await invoke("model_plan");
    list.innerHTML = "";
    if (!plan.length) {
      list.innerHTML = '<p class="muted">The coordinator manifest lists no models.</p>';
      return;
    }
    plan.forEach((entry) => list.appendChild(renderModelRow(entry)));
  } catch (err) {
    list.innerHTML = "";
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = `Could not load the download plan: ${err}`;
    list.appendChild(message);
  }
}

$("btn-plan").addEventListener("click", refreshPlan);

$("btn-fetch").addEventListener("click", async () => {
  const button = $("btn-fetch");
  button.disabled = true;
  button.textContent = "Downloading…";
  modelProgress.clear();
  try {
    await invoke("fetch_models", { faceAssets: $("fetch-face-assets").checked });
  } catch (err) {
    toast(String(err), true);
    button.disabled = false;
    button.textContent = "Download missing";
  }
});

function updateModelRow(name, updater) {
  const row = document.querySelector(`.model[data-model="${CSS.escape(name)}"]`);
  if (row) updater(row);
}

listen("models-output", (event) => {
  const { stream, line } = event.payload;
  if (stream === "stderr") return;

  let payload;
  try {
    payload = JSON.parse(line);
  } catch {
    return; // non-JSON chatter on stdout is not interesting here
  }

  if (payload.event === "progress" && payload.percent !== null) {
    modelProgress.set(payload.model, payload.percent);
    updateModelRow(payload.model, (row) => {
      row.querySelector(".progress-bar").style.width = `${payload.percent}%`;
    });
  }

  if (payload.event === "model_done") {
    const label =
      payload.status === "downloaded" || payload.status === "present"
        ? "present"
        : payload.status === "skipped"
          ? "manual"
          : "failed";
    updateModelRow(payload.model, (row) => {
      const state = row.querySelector(".state");
      state.className = `state state-${label}`;
      state.textContent = label === "manual" ? "supply manually" : label;
      if (label === "present") {
        row.querySelector(".progress-bar").style.width = "100%";
      }
    });
  }

  if (payload.event === "summary") {
    const parts = [`${payload.downloaded} downloaded`, `${payload.present} present`];
    if (payload.skipped) parts.push(`${payload.skipped} manual`);
    if (payload.failed) parts.push(`${payload.failed} failed`);
    toast(parts.join(" · "), Boolean(payload.failed));
  }
});

listen("models-output-done", async () => {
  $("btn-fetch").disabled = false;
  $("btn-fetch").textContent = "Download missing";
  await refreshPlan();
});

// ---------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------

async function refreshLogs() {
  try {
    const lines = await invoke("read_logs", { lines: 400 });
    const el = $("log-console");
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    el.textContent = lines.join("\n");
    if (atBottom) el.scrollTop = el.scrollHeight;
  } catch (err) {
    $("log-console").textContent = `Could not read logs: ${err}`;
  }
}

$("btn-logs").addEventListener("click", refreshLogs);

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  await refreshInstallState();
  await loadConfig();
  await refreshStatus();
  await runDoctor();
  await refreshPlan();

  statusTimer = setInterval(refreshStatus, 5000);
  logTimer = setInterval(() => {
    if ($("panel-logs").classList.contains("is-active") && $("logs-auto").checked) {
      refreshLogs();
    }
  }, 4000);
}

window.addEventListener("beforeunload", () => {
  clearInterval(statusTimer);
  clearInterval(logTimer);
});

boot();
