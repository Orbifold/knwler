/* ===================================================================
   Knwler Desktop App – Frontend Logic
   =================================================================== */

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
let API_BASE = "";
let currentJobId = null;
let ws = null;
let selectedFile = null;

// Stage display names
const STAGE_LABELS = {
    loading:              "Loading document...",
    language_detection:   "Detecting language...",
    schema_discovery:     "Discovering schema...",
    chunking:             "Chunking text...",
    title_extraction:     "Extracting title...",
    summary_extraction:   "Generating summary...",
    rephrasing:           "Rephrasing chunks...",
    extraction:           "Extracting knowledge graph...",
    consolidation:        "Consolidating results...",
    community_analysis:   "Analyzing communities...",
};

// Cumulative weights for overall progress (must sum to 100)
const STAGE_ORDER = [
    "loading",
    "language_detection",
    "schema_discovery",
    "chunking",
    "title_extraction",
    "summary_extraction",
    "rephrasing",
    "extraction",
    "consolidation",
    "community_analysis",
];

const STAGE_WEIGHTS = {
    loading:              1,
    language_detection:   2,
    schema_discovery:     8,
    chunking:             1,
    title_extraction:     3,
    summary_extraction:   3,
    rephrasing:          17,
    extraction:          50,
    consolidation:        5,
    community_analysis:  10,
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    // Restore saved settings
    restoreSettings();

    // Bind UI events
    document.getElementById("btn-pick-file").addEventListener("click", pickFile);
    document.getElementById("file-input").addEventListener("change", onFileSelected);
    document.getElementById("setting-backend").addEventListener("change", onBackendChange);
    document.getElementById("btn-start").addEventListener("click", startExtraction);
    document.getElementById("btn-new").addEventListener("click", () => showView("upload"));
    document.getElementById("btn-export-json").addEventListener("click", exportJSON);
    document.getElementById("btn-export-html").addEventListener("click", exportHTML);

    // Drag & drop
    const dropZone = document.getElementById("drop-zone");
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", onFileDrop);

    // Check for Tauri environment
    if (window.__TAURI__) {
        const { listen } = window.__TAURI__.event;
        listen("backend-ready", (event) => {
            API_BASE = `http://127.0.0.1:${event.payload}`;
            console.log("Backend ready at", API_BASE);
        });
    } else {
        // Dev mode – connect to the local server (run: make server)
        API_BASE = localStorage.getItem("knwler-dev-api") || "http://127.0.0.1:8765";
        console.log("Dev mode – API:", API_BASE);
        waitForBackend();
    }
});

// ---------------------------------------------------------------------------
// View management
// ---------------------------------------------------------------------------
function showView(name) {
    document.querySelectorAll(".view").forEach((el) => el.classList.remove("active"));
    document.getElementById(`view-${name}`).classList.add("active");

    // Reset body padding for results view (full bleed)
    if (name === "results") {
        document.body.style.padding = "0";
    } else {
        document.body.style.padding = "2rem";
    }
}

// ---------------------------------------------------------------------------
// Backend connection (dev mode)
// ---------------------------------------------------------------------------
async function waitForBackend() {
    const startBtn = document.getElementById("btn-start");
    const subtitle = document.querySelector(".subtitle");
    const originalText = subtitle ? subtitle.textContent : "";

    async function check() {
        try {
            const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
            if (res.ok) {
                if (subtitle) subtitle.textContent = originalText;
                console.log("Backend connected");
                return true;
            }
        } catch (e) {
            // not ready yet
        }
        return false;
    }

    if (await check()) return;

    // Show waiting state
    if (subtitle) subtitle.textContent = "Waiting for backend... (run: make server)";
    startBtn.disabled = true;

    const interval = setInterval(async () => {
        if (await check()) {
            clearInterval(interval);
            // Re-enable start if a file is already selected
            if (selectedFile) startBtn.disabled = false;
        }
    }, 2000);
}

// ---------------------------------------------------------------------------
// File selection
// ---------------------------------------------------------------------------
async function pickFile() {
    if (window.__TAURI__) {
        const { open } = window.__TAURI__.dialog;
        const path = await open({
            filters: [{ name: "Documents", extensions: ["pdf", "txt", "md"] }],
            multiple: false,
        });
        if (path) {
            // Tauri returns a file path; we need to read it for upload
            const { readFile } = window.__TAURI__.fs;
            const contents = await readFile(path);
            const name = path.split("/").pop();
            selectedFile = new File([contents], name);
            showSelectedFile(name);
        }
    } else {
        document.getElementById("file-input").click();
    }
}

function onFileSelected(e) {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        showSelectedFile(file.name);
    }
}

function onFileDrop(e) {
    e.preventDefault();
    document.getElementById("drop-zone").classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) {
        selectedFile = file;
        showSelectedFile(file.name);
    }
}

function showSelectedFile(name) {
    const el = document.getElementById("selected-file");
    el.textContent = name;
    el.style.display = "block";
    document.getElementById("btn-start").disabled = false;
}

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------
function saveSettings() {
    const settings = {
        backend: document.getElementById("setting-backend").value,
        openaiKey: document.getElementById("setting-openai-key").value,
        openaiUrl: document.getElementById("setting-openai-url").value,
        extractionModel: document.getElementById("setting-extraction-model").value,
        discoveryModel: document.getElementById("setting-discovery-model").value,
        concurrent: document.getElementById("setting-concurrent").value,
        maxTokens: document.getElementById("setting-max-tokens").value,
        language: document.getElementById("setting-language").value,
        noDiscovery: document.getElementById("setting-no-discovery").checked,
    };
    localStorage.setItem("knwler-settings", JSON.stringify(settings));
}

function restoreSettings() {
    try {
        const raw = localStorage.getItem("knwler-settings");
        if (!raw) return;
        const s = JSON.parse(raw);
        if (s.backend) document.getElementById("setting-backend").value = s.backend;
        if (s.openaiKey) document.getElementById("setting-openai-key").value = s.openaiKey;
        if (s.openaiUrl) document.getElementById("setting-openai-url").value = s.openaiUrl;
        if (s.extractionModel) document.getElementById("setting-extraction-model").value = s.extractionModel;
        if (s.discoveryModel) document.getElementById("setting-discovery-model").value = s.discoveryModel;
        if (s.concurrent) document.getElementById("setting-concurrent").value = s.concurrent;
        if (s.maxTokens) document.getElementById("setting-max-tokens").value = s.maxTokens;
        if (s.language) document.getElementById("setting-language").value = s.language;
        if (s.noDiscovery) document.getElementById("setting-no-discovery").checked = s.noDiscovery;
        onBackendChange();
    } catch (e) {
        // ignore
    }
}

function onBackendChange() {
    const isOpenAI = document.getElementById("setting-backend").value === "openai";
    document.getElementById("openai-settings").style.display = isOpenAI ? "block" : "none";
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------
async function startExtraction() {
    if (!selectedFile || !API_BASE) return;

    saveSettings();

    const formData = new FormData();
    formData.append("file", selectedFile);

    const isOpenAI = document.getElementById("setting-backend").value === "openai";
    formData.append("use_openai", isOpenAI);
    if (isOpenAI) {
        formData.append("openai_api_key", document.getElementById("setting-openai-key").value);
        formData.append("openai_base_url", document.getElementById("setting-openai-url").value);
    }

    const extractionModel = document.getElementById("setting-extraction-model").value;
    const discoveryModel = document.getElementById("setting-discovery-model").value;
    if (extractionModel) formData.append("extraction_model", extractionModel);
    if (discoveryModel) formData.append("discovery_model", discoveryModel);

    formData.append("max_concurrent", document.getElementById("setting-concurrent").value);
    formData.append("max_tokens", document.getElementById("setting-max-tokens").value);
    formData.append("no_discovery", document.getElementById("setting-no-discovery").checked);

    const lang = document.getElementById("setting-language").value;
    if (lang) formData.append("language", lang);

    // Reset processing view
    document.getElementById("stage-label").textContent = "Starting...";
    document.getElementById("progress-bar").style.width = "0%";
    document.getElementById("progress-text").textContent = "0%";
    document.getElementById("log-output").innerHTML = "";
    showView("processing");

    try {
        const response = await fetch(`${API_BASE}/extract`, {
            method: "POST",
            body: formData,
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: response.statusText }));
            throw new Error(err.detail || err.error || "Server error");
        }
        const data = await response.json();
        currentJobId = data.job_id;
        connectWebSocket(data.job_id);
    } catch (err) {
        document.getElementById("stage-label").textContent = "Failed to start";
        addLog(`Error: ${err.message}`);
        if (err.message === "Failed to fetch" || err.message === "Load failed") {
            addLog("Backend not reachable. Run: make server");
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocket progress
// ---------------------------------------------------------------------------
function connectWebSocket(jobId) {
    const wsUrl = API_BASE.replace("http", "ws") + `/ws/jobs/${jobId}`;
    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "progress") {
            updateProgress(msg.stage, msg.current, msg.total);
        } else if (msg.type === "completed") {
            loadResults(jobId);
        } else if (msg.type === "error") {
            addLog(`Error: ${msg.message}`);
            document.getElementById("stage-label").textContent = "Failed";
        }
    };

    ws.onerror = () => {
        // Fallback to polling if WebSocket fails
        pollStatus(jobId);
    };

    ws.onclose = () => {
        ws = null;
    };
}

function pollStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/jobs/${jobId}/status`);
            const data = await res.json();
            if (data.progress) {
                updateProgress(data.progress.stage, data.progress.current, data.progress.total);
            }
            if (data.status === "completed") {
                clearInterval(interval);
                loadResults(jobId);
            } else if (data.status === "failed") {
                clearInterval(interval);
                addLog(`Error: ${data.error}`);
                document.getElementById("stage-label").textContent = "Failed";
            }
        } catch (e) {
            // Keep polling
        }
    }, 1000);
}

// ---------------------------------------------------------------------------
// Progress display
// ---------------------------------------------------------------------------
let completedStages = new Set();

function updateProgress(stage, current, total) {
    // Update stage label
    document.getElementById("stage-label").textContent =
        STAGE_LABELS[stage] || stage;

    // Calculate overall percentage based on stage weights
    let overallPercent = 0;

    // Add weights of fully completed stages
    for (const s of STAGE_ORDER) {
        if (s === stage) break;
        overallPercent += STAGE_WEIGHTS[s] || 0;
    }

    // Add partial weight of current stage
    const stageWeight = STAGE_WEIGHTS[stage] || 0;
    if (total > 0) {
        overallPercent += stageWeight * (current / total);
    }

    overallPercent = Math.min(100, Math.round(overallPercent));

    document.getElementById("progress-bar").style.width = overallPercent + "%";
    document.getElementById("progress-text").textContent = overallPercent + "%";

    // Log stage transitions
    if (!completedStages.has(stage) && current > 0) {
        completedStages.add(stage);
        const label = STAGE_LABELS[stage] || stage;
        if (total > 1) {
            addLog(`${label} (${current}/${total})`);
        } else {
            addLog(label);
        }
    }

    // Update log with current/total for multi-item stages
    if (total > 1) {
        updateLastLog(`${STAGE_LABELS[stage] || stage} (${current}/${total})`);
    }
}

function addLog(text) {
    const el = document.getElementById("log-output");
    const line = document.createElement("div");
    line.className = "log-line";
    line.textContent = text;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
}

function updateLastLog(text) {
    const el = document.getElementById("log-output");
    const lines = el.querySelectorAll(".log-line");
    if (lines.length > 0) {
        lines[lines.length - 1].textContent = text;
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
async function loadResults(jobId) {
    document.getElementById("stage-label").textContent = "Done!";
    document.getElementById("progress-bar").style.width = "100%";
    document.getElementById("progress-text").textContent = "100%";

    // Small delay so user sees 100%
    await new Promise((r) => setTimeout(r, 600));

    const iframe = document.getElementById("results-iframe");
    iframe.src = `${API_BASE}/jobs/${jobId}/report`;
    showView("results");

    // Reset for next run
    completedStages = new Set();
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
async function exportJSON() {
    if (!currentJobId) return;
    try {
        const res = await fetch(`${API_BASE}/jobs/${currentJobId}/result`);
        const data = await res.json();
        downloadBlob(
            JSON.stringify(data, null, 2),
            "knwler-results.json",
            "application/json"
        );
    } catch (e) {
        alert("Export failed: " + e.message);
    }
}

async function exportHTML() {
    if (!currentJobId) return;
    try {
        const res = await fetch(`${API_BASE}/jobs/${currentJobId}/report`);
        const html = await res.text();
        downloadBlob(html, "knwler-report.html", "text/html");
    } catch (e) {
        alert("Export failed: " + e.message);
    }
}

function downloadBlob(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
