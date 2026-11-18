const PAGE_SIZE = 8;
const POLL_INTERVAL = 10000;
const $ = (id) => document.getElementById(id);
const previewPath = /^\/v2\/marketplace\/listings\/[1-9]\d*\/preview$/;
const apiBases = [...new Set([window.NEXT_PUBLIC_API_BASE_URL, window.NEXT_PUBLIC_HAVNAI_API_BASE, "", "/api"]
  .filter(base => typeof base === "string").map(base => base.replace(/\/+$/, "")))];
let preferredApiBase = null;
let feedBase = "";
let jobs = [];
let page = 0;
let refreshing = false;
let view = "public";
let accountRequest;
let accountRows = [];
let accountTotal = 0;
let accountMessage = "";
let publicListings = [];
let gallerySnapshot;
let renderedPage;
const dateFormat = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content != null) node.textContent = content;
  if (className) node.className = className;
  return node;
}

async function readJson(path) {
  const bases = [...new Set([preferredApiBase, ...apiBases])].filter(base => base !== null);
  for (const base of bases) {
    try {
      const response = await fetch(base + path, { cache: "no-store", signal: AbortSignal.timeout(10000) });
      if (response.status === 401 || response.status === 403) return { restricted: true };
      if (response.ok) return { data: await response.json(), base };
    } catch { /* Another configured API mount may be available. */ }
  }
  return {};
}

function attachApprovedPreview(card, path, base = preferredApiBase || "", alt = "Public listing preview") {
  if (typeof path !== "string" || !previewPath.test(path)) return;
  const img = document.createElement("img");
  img.alt = alt;
  img.width = 96;
  img.height = 96;
  img.hidden = true;
  img.addEventListener("load", () => {
    card.replaceChildren(img);
    img.hidden = false;
  }, { once: true });
  img.addEventListener("error", () => img.remove(), { once: true });
  card.append(img);
  img.src = base + path;
}

function renderGallery({ data, base }) {
  const gallery = $("pictureGallery");
  const listings = data?.listings?.filter(item => item.asset_type === "image" &&
    Number.isSafeInteger(item.id) && item.id > 0 && item.preview_url === `/v2/marketplace/listings/${item.id}/preview`);
  publicListings = (listings || []).map(item => ({ job_id: item.title || ("Listing #" + item.id), model: item.model,
    status: "Published", submitted_at: item.created_at, preview_url: item.preview_url, listingBase: base }));
  const snapshot = JSON.stringify([base, listings]);
  if (snapshot === gallerySnapshot && !gallery.querySelector(".preview:not(:has(img))")) return;
  gallerySnapshot = snapshot;
  if (!listings?.length) {
    gallery.replaceChildren(element("p", listings ? "No pictures have been published yet." : "Published pictures are temporarily unavailable.", "empty"));
    return;
  }
  gallery.replaceChildren(...listings.map(item => {
    const figure = element("figure");
    const frame = element("div", null, "preview");
    frame.append(element("span", "Preview unavailable"));
    attachApprovedPreview(frame, item.preview_url, base, item.title || "Published picture");
    figure.append(frame, element("figcaption", item.title || "Untitled picture"));
    return figure;
  }));
}

function number(value, digits = 0) {
  return value != null && Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits, minimumFractionDigits: digits
  }) : "—";
}

function localDate(value) {
  const date = new Date(typeof value === "number" ? value * 1000 : value);
  return value && Number.isFinite(date.getTime()) ? dateFormat.format(date) : "—";
}

function renderNetwork(data, rewards) {
  const summary = data.summary || {};
  const stats = data.job_summary || {};
  $("summaryOnline").textContent = number(summary.online_nodes);
  $("summaryTotal").textContent = `${number(summary.total_nodes)} total ${summary.total_nodes === 1 ? "node" : "nodes"}`;
  $("summaryUtil").textContent = `${number(summary.avg_utilization, 1)}%`;
  $("summaryActive").textContent = number(stats.active_jobs);
  $("summaryBacklog").textContent = `${number(stats.queued_jobs ?? summary.tasks_backlog)} queued`;
  $("metricJobsToday").textContent = number(stats.jobs_completed_today);
  $("summaryRewards").textContent = `${number(stats.total_distributed, 6)} HAI`;
  const nodes = (data.nodes || []).slice().sort((a, b) => String(a.node_id).localeCompare(String(b.node_id)));
  $("nodeList").replaceChildren(...nodes.map(node => {
    const row = element("li");
    const online = node.online ?? (node.status === "online");
    const state = element("span", online ? "Online" : "Offline", "node-state");
    state.dataset.state = online ? "online" : "offline";
    const detail = [node.role || "Worker", `${number(node.avg_utilization ?? node.gpu_utilization, 1)}% GPU`,
      `${number(rewards?.[node.node_id] ?? node.rewards, 6)} HAI`].join(" · ");
    row.append(element("span", node.node_name || node.node_id, "node-name"), state, element("small", detail, "node-detail"));
    return row;
  }));
  if (!nodes.length) $("nodeList").append(element("li", "No registered nodes.", "empty"));
}

function emptyRow(columns, message) {
  const row = element("tr");
  const cell = element("td", message);
  cell.colSpan = columns;
  row.append(cell);
  return row;
}

function publicFeed() {
  const time = job => typeof job.submitted_at === "number" ? job.submitted_at * 1000 : Date.parse(job.submitted_at) || 0;
  return [...jobs, ...publicListings].sort((a, b) => time(b) - time(a));
}

function renderJobs() {
  const personal = view === "private";
  const feed = personal ? accountRows : publicFeed();
  const total = personal ? accountTotal : feed.length;
  if (!personal) page = Math.min(page, Math.max(0, Math.ceil(total / PAGE_SIZE) - 1));
  const slice = personal ? feed : feed.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const fingerprint = JSON.stringify([view, slice, page, feedBase, accountMessage]);
  if (fingerprint !== renderedPage) {
    renderedPage = fingerprint;
    $("jobFeed").replaceChildren(...slice.map(job => {
      const row = element("tr");
      const preview = element("div", null, "job-preview");
      const cell = element("td");
      cell.append(preview);
      preview.append(element("span", personal ? "Preview unavailable" : "No public preview"));
      if (personal && typeof job.private_preview === "string" && /^\/api\/account-media\/[a-zA-Z0-9_-]+$/.test(job.private_preview)) {
        const img = element("img");
        img.alt = `Your picture for ${job.job_id}`;
        img.hidden = true;
        img.addEventListener("load", () => { preview.replaceChildren(img); img.hidden = false; }, { once: true });
        img.addEventListener("error", () => img.remove(), { once: true });
        preview.append(img);
        img.src = job.private_preview;
      } else if (!personal) {
        attachApprovedPreview(preview, job.preview_url, job.listingBase ?? feedBase);
      }
      const state = String(job.status || "Unknown").toLowerCase();
      const badge = element("span", state.charAt(0).toUpperCase() + state.slice(1), "job-state");
      badge.dataset.state = /^(success|succeeded|done)$/.test(state) ? "completed" : state;
      const statusCell = element("td");
      statusCell.append(badge);
      row.append(cell, element("td", job.job_id), element("td", job.model || "—"), statusCell,
        element("td", localDate(job.completed_at || job.submitted_at)));
      if (!personal) row.append(element("td", job.wallet || "—"), element("td", number(job.reward_hai ?? job.reward, 6)));
      return row;
    }));
    if (!slice.length) {
      const row = emptyRow(personal ? 5 : 7, personal ? accountMessage || "No account jobs on this page." : "No recent public jobs.");
      if (personal && accountMessage === "Sign in to see your pictures.") {
        const link = element("a", " Sign in ↗");
        link.href = "/sign-in?redirect_url=%2Fapi%2Fdashboard%3Fview%3Dmine";
        row.firstChild.append(link);
      }
      $("jobFeed").append(row);
    }
  }
  $("jobCount").textContent = slice.length ? `${page * PAGE_SIZE + 1}–${page * PAGE_SIZE + slice.length} of ${total}` : "";
  $("prevJobs").disabled = page === 0;
  $("nextJobs").disabled = (page + 1) * PAGE_SIZE >= total;
}

function resetAccount(message = "") {
  accountRequest?.abort();
  accountRequest = null;
  accountRows = [];
  accountTotal = 0;
  accountMessage = message;
  renderedPage = null;
}

async function loadAccount() {
  if (view !== "private" || document.hidden || accountRequest) return;
  const controller = new AbortController();
  accountRequest = controller;
  try {
    const response = await fetch("/api/grid/my-jobs?offset=" + page * PAGE_SIZE, {
      cache: "no-store", credentials: "same-origin", redirect: "error", signal: controller.signal,
    });
    if (!response.ok) throw new Error(response.status === 401 ? "Sign in to see your pictures." : "Your job history is temporarily unavailable.");
    const data = await response.json();
    if (!Array.isArray(data.jobs) || !Number.isSafeInteger(data.total)) throw new Error("Your job history is temporarily unavailable.");
    if (controller.signal.aborted) return;
    if (!data.jobs.length && page > 0 && page * PAGE_SIZE >= data.total) {
      page = Math.max(0, Math.ceil(data.total / PAGE_SIZE) - 1);
      accountRequest = null;
      return loadAccount();
    }
    accountRows = data.jobs;
    accountTotal = data.total;
    accountMessage = "";
  } catch (error) {
    if (controller.signal.aborted) return;
    accountRows = [];
    accountTotal = 0;
    accountMessage = error.message || "Your job history is temporarily unavailable.";
  } finally {
    if (accountRequest === controller) { accountRequest = null; renderJobs(); }
  }
}

function chooseView(next) {
  if (next === "private" && !window.location.pathname.startsWith("/api/")) {
    window.location.assign("https://joinhavn.io/api/dashboard?view=mine");
    return;
  }
  view = next;
  const url = new URL(window.location.href);
  if (next === "public") url.searchParams.delete("view");
  else url.searchParams.set("view", next === "private" ? "mine" : "pictures");
  window.history.replaceState(null, "", url);
  page = 0;
  resetAccount(next === "private" ? "Loading your pictures…" : "");
  for (const [id, name] of [["publicTab", "public"], ["picturesTab", "pictures"], ["privateTab", "private"]]) {
    $(id).setAttribute("aria-pressed", String(next === name));
  }
  $("privateHint").hidden = next !== "private";
  $("publishedPanel").hidden = next === "private";
  $("publishedPanel").classList.toggle("compact-gallery", next === "public");
  $("jobScroll").hidden = $("pager").hidden = next === "pictures";
  document.querySelectorAll(".public-column").forEach(cell => { cell.hidden = next === "private"; });
  renderJobs();
  if (next === "private") loadAccount();
}

function turnPage(step) {
  page += step;
  if (view === "private") resetAccount("Loading your pictures…");
  renderJobs();
  if (view === "private") loadAccount();
}

function renderModels(models) {
  $("modelCatalog").replaceChildren(...models.map(model => {
    const row = element("tr");
    const labels = value => Array.isArray(value) ? value.join(", ") : value;
    for (const value of [model.model, number(model.weight, 2), model.source, labels(model.nodes), labels(model.tags),
      model.size ? `${number(model.size / 1048576, 2)} MB` : "—"]) row.append(element("td", value || "—"));
    return row;
  }));
  if (!models.length) $("modelCatalog").append(emptyRow(6, "No models registered yet."));
}

async function fetchLogs() {
  const result = await readJson("/logs");
  const logs = result.data?.logs;
  $("logList").replaceChildren(...(logs || []).slice(-8).reverse().map(entry => {
    const item = element("li", entry.message);
    item.append(element("time", localDate(entry.timestamp)));
    return item;
  }));
  if (!logs?.length) $("logList").append(element("li", result.restricted ? "Activity logs are restricted to operators." :
    logs ? "No recent events." : "Activity logs are unavailable."));
}

async function fetchTelemetry() {
  if (refreshing) return;
  refreshing = true;
  $("refresh").disabled = true;
  try {
    const gallery = readJson("/v2/marketplace/listings?limit=6").then(renderGallery);
    const telemetry = Promise.all([readJson("/nodes"), readJson("/rewards")]).then(([nodes, rewards]) => {
      if (!Array.isArray(nodes.data?.nodes)) throw new Error("Grid unavailable. Retrying in 10 seconds.");
      preferredApiBase = feedBase = nodes.base;
      renderNetwork(nodes.data, rewards.data?.rewards);
      jobs = nodes.data.job_summary?.feed || [];
      renderJobs();
      renderModels(nodes.data.models_catalog || []);
      $("status").textContent = "Updated " + new Date().toLocaleTimeString();
      $("status").dataset.tone = "info";
    });
    const [, network] = await Promise.allSettled([gallery, telemetry]);
    renderJobs();
    if (network.status === "rejected") throw network.reason;
    if (document.querySelector("details.activity").open) await fetchLogs();
  } catch (error) {
    $("status").textContent = error.message || "Grid unavailable. Try refreshing.";
    $("status").dataset.tone = "error";
  } finally {
    refreshing = false;
    $("refresh").disabled = false;
  }
}

function startDashboard() {
  $("refresh").addEventListener("click", () => { fetchTelemetry(); loadAccount(); });
  $("publicTab").addEventListener("click", () => chooseView("public"));
  $("picturesTab").addEventListener("click", () => chooseView("pictures"));
  $("privateTab").addEventListener("click", () => chooseView("private"));
  $("prevJobs").addEventListener("click", () => turnPage(-1));
  $("nextJobs").addEventListener("click", () => turnPage(1));
  document.querySelector("details.activity").addEventListener("toggle", event => { if (event.target.open) fetchLogs(); });
  const conceal = () => {
    if (view === "private") { resetAccount("Loading your pictures…"); renderJobs(); }
  };
  document.addEventListener("visibilitychange", () => { if (document.hidden) conceal(); else loadAccount(); });
  window.addEventListener("pagehide", conceal);
  window.addEventListener("pageshow", loadAccount);
  const requested = new URLSearchParams(window.location.search).get("view");
  chooseView(requested === "mine" ? "private" : requested === "pictures" ? "pictures" : "public");
  fetchTelemetry();
  setInterval(() => { fetchTelemetry(); loadAccount(); }, POLL_INTERVAL);
}
startDashboard();
