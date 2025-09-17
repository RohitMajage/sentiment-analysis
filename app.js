// Entry point for the static demo app

// Globals for models
let bertClassifier = null; // Transformers.js pipeline (removed from UI)
let lstmModel = null;      // TensorFlow.js model
let bertModelId = "Xenova/bert-base-multilingual-uncased-sentiment";
let stopRequested = false;
let bertChart = null;
let lstmChart = null;
let currentOutputs = [];
let filteredOutputs = [];

// Language detection with fallbacks: langdetect-lite -> Unicode script heuristic
function detectLanguage(text) {
    const t = (text || "").trim();
    if (!t) return null;
    try {
        if (window.langdetect && typeof window.langdetect.detectOne === "function") {
            const result = window.langdetect.detectOne(t);
            if (result && result.lang) return result.lang;
        }
    } catch (_) {}
    // Heuristic by script
    const has = (re) => re.test(t);
    if (has(/[\u0600-\u06FF]/)) return "ar";      // Arabic
    // Devanagari block (Hindi/Marathi)
    if (has(/[\u0900-\u097F]/)) {
        // Marathi hints
        const mrHints = /\b(आहे|मला|तू|तुम्ही|छान|नाही|होय|कृपया|भारतात|मराठी)\b/;
        if (mrHints.test(t)) return "mr";
        return "hi";
    }
    // Kannada block
    if (has(/[\u0C80-\u0CFF]/)) return "kn";      // Kannada
    if (has(/[\u3040-\u30FF]/)) return "ja";      // Hiragana/Katakana
    if (has(/[\u4E00-\u9FFF]/)) return "zh";      // CJK Unified Ideographs
    if (has(/[\u0400-\u04FF]/)) return "ru";      // Cyrillic
    // Latin-based quick guesses by common words
    const lower = t.toLowerCase();
    if (/\b(bien|mal|producto|servicio|me encanta|no me gusta)\b/.test(lower)) return "es";
    if (/\b(bon|mauvais|mise à jour|j’aime|je n’aime pas)\b/.test(lower)) return "fr";
    if (/\b(gut|schlecht)\b/.test(lower)) return "de";
    if (/\b(bom|ruim|ótimo|serviço)\b/.test(lower)) return "pt";
    return "en"; // default
}

// Basic cleaning suitable for tweets
function cleanTweet(text) {
    return text
        .replace(/https?:\/\/\S+/g, "")
        .replace(/www\.\S+/g, "")
        .replace(/@[\w_]+/g, "")
        .replace(/#[\w_]+/g, "")
        .replace(/\s+/g, " ")
        .trim();
}

// Map score to label buckets if needed
function scoreToLabel(scoreObj) {
    // scoreObj: { label, score }
    if (!scoreObj || !scoreObj.label) return { label: "Neutral", score: 0.0 };
    const labelText = String(scoreObj.label).trim();
    const raw = labelText.toLowerCase();

    // Direct keywords
    if (raw.includes("positive") || raw.includes("pos")) return { label: "Positive", score: scoreObj.score };
    if (raw.includes("negative") || raw.includes("neg")) return { label: "Negative", score: scoreObj.score };
    if (raw.includes("neutral")) return { label: "Neutral", score: scoreObj.score };

    // LABEL_0/1/2 mapping (common for some models like CardiffNLP)
    if (/^label_\d+$/i.test(raw)) {
        const idx = parseInt(raw.split("_")[1], 10);
        if (idx === 0) return { label: "Negative", score: scoreObj.score };
        if (idx === 1) return { label: "Neutral", score: scoreObj.score };
        if (idx === 2) return { label: "Positive", score: scoreObj.score };
    }

    // Star ratings (e.g., nlptown/bert-base-multilingual-uncased-sentiment)
    const starMatch = raw.match(/([1-5])\s*star/);
    if (starMatch) {
        const n = parseInt(starMatch[1], 10);
        if (n <= 2) return { label: "Negative", score: scoreObj.score };
        if (n === 3) return { label: "Neutral", score: scoreObj.score };
        return { label: "Positive", score: scoreObj.score };
    }

    // Default
    return { label: "Neutral", score: scoreObj.score };
}

// Load BERT sentiment classifier (Transformers.js)
async function ensureBert() {
    if (bertClassifier) return bertClassifier;
    const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers@3.0.1");
    // Configure wasm assets path and caching to avoid load issues
    try {
        env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/@xenova/transformers@3.0.1/dist/";
        env.remotePathTemplate = "https://huggingface.co/{model}/resolve/main/{file}";
        env.allowRemoteModels = true;
        env.useBrowserCache = true;
    } catch (e) {
        console.warn("Transformers env config warning", e);
    }
    const tryLoad = async (modelId) => {
        try {
            setStatus(`Loading BERT: ${modelId} ...`);
            const p = await pipeline("sentiment-analysis", modelId, { quantized: true });
            setStatus(`BERT ready: ${modelId}`);
            return p;
        } catch (e) {
            console.warn(`Failed to load ${modelId}`, e);
            return null;
        }
    };
    // Attempt selected model first
    let p = await tryLoad(bertModelId);
    if (!p) {
        // Fallback to smaller, widely cached model
        const fallbackId = "nlptown/bert-base-multilingual-uncased-sentiment";
        p = await tryLoad(fallbackId);
        if (p) {
            bertModelId = fallbackId;
            const sel = document.getElementById("bertModel");
            if (sel) sel.value = fallbackId;
        }
    }
    if (!p) throw new Error("All BERT model loads failed");
    bertClassifier = p;
    return bertClassifier;
}

// Load LSTM tfjs model (pluggable)
async function ensureLstm() {
    if (lstmModel) return lstmModel;
    // Placeholder public demo model (English): TensorFlow.js toxicity is not sentiment; need a small sentiment model.
    // Provide a user-replaceable URL. If loading fails, we compute a heuristic lexicon-based score as fallback.
    const MODEL_URL = window.localStorage.getItem("lstm_model_url") || "";
    if (MODEL_URL) {
        try {
            lstmModel = await tf.loadLayersModel(MODEL_URL);
            return lstmModel;
        } catch (e) {
            console.warn("Failed to load LSTM model from", MODEL_URL, e);
        }
    }
    lstmModel = null; // keep null to trigger heuristic fallback
    return null;
}

// Very small heuristic fallback sentiment (multilingual-light via emojis and polarity words)
function heuristicSentiment(text) {
    const t = text.toLowerCase();
    const positiveHints = ["good", "great", "love", "awesome", "amazing", "😊", "😀", "❤️", "👍", "bien", "bueno", "bon", " ótimo", "bom", "gut", "schön", "حسن", "جميل", "अच्छा", "पसंद" ];
    const negativeHints = ["bad", "hate", "terrible", "awful", "worst", "😡", "😠", "💔", "👎", "mal", "malo", "mauvais", "ruim", "schlecht", "قبيح", "سيء", "खराब", "नफ़रत" ];
    let pos = 0, neg = 0;
    for (const w of positiveHints) if (t.includes(w)) pos++;
    for (const w of negativeHints) if (t.includes(w)) neg++;
    if (pos === 0 && neg === 0) return { label: "Neutral", score: 0.5 };
    if (pos >= neg) return { label: "Positive", score: Math.min(1, 0.5 + 0.1 * (pos - neg)) };
    return { label: "Negative", score: Math.min(1, 0.5 + 0.1 * (neg - pos)) };
}

// Tokenizer for LSTM model (placeholder: simple hashing-based)
function tokenizeForLstm(text, vocabSize = 10000, sequenceLength = 40) {
    // If using a real model, replace with its tokenizer. For demo fallback we won't use this.
    const tokens = cleanTweet(text).toLowerCase().split(/\s+/g).filter(Boolean);
    const ids = new Array(sequenceLength).fill(0);
    for (let i = 0; i < Math.min(tokens.length, sequenceLength); i++) {
        const h = [...tokens[i]].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7);
        ids[i] = (h % (vocabSize - 2)) + 1;
    }
    return ids;
}

async function predictWithLstm(text) {
    const model = await ensureLstm();
    if (!model) {
        return heuristicSentiment(text);
    }
    const input = tokenizeForLstm(text);
    const tensor = tf.tensor2d([input], [1, input.length]);
    const logits = model.predict(tensor);
    const data = await logits.data();
    tensor.dispose();
    if (logits.dispose) logits.dispose();
    // Assume 3-way output [neg, neu, pos]
    const arr = Array.from(data);
    const labels = ["Negative", "Neutral", "Positive"];
    let maxIdx = 0; let maxVal = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
    const sum = arr.reduce((a, b) => a + Math.exp(b), 0);
    const conf = Math.exp(arr[maxIdx]) / sum;
    return { label: labels[maxIdx] || "Neutral", score: conf };
}

async function predictWithBert(text) {
    const classifier = await ensureBert();
    const out = await classifier(text);
    const first = Array.isArray(out) ? out[0] : out;
    // Some pipelines may return an array for a single item; pick top-score
    const normalized = Array.isArray(first) ? first.reduce((a, b) => (b.score > a.score ? b : a), first[0]) : first;
    return scoreToLabel(normalized);
}

async function predictBatchWithBert(texts) {
    // Some models/backends are unstable with array batching. Run per-item for reliability.
    let classifier;
    try {
        classifier = await ensureBert();
    } catch (e) {
        console.warn("BERT unavailable", e);
        return new Array(texts.length).fill(null);
    }
    const results = [];
    for (let i = 0; i < texts.length; i++) {
        try {
            const out = await classifier(texts[i]);
            const first = Array.isArray(out) ? out[0] : out;
            const top = Array.isArray(first) ? first.reduce((a, b) => (b.score > a.score ? b : a), first[0]) : first;
            results.push(scoreToLabel(top));
        } catch (err) {
            console.warn("BERT inference failed for item", i, err);
            results.push(null);
        }
    }
    return results;
}

function setStatus(msg) {
    const s = document.getElementById("status");
    if (s) s.textContent = msg;
}

function setProgress(percent, visible) {
    const p = document.getElementById("progress");
    if (!p) return;
    if (visible === true) p.style.display = "inline-block";
    if (visible === false) p.style.display = "none";
    if (typeof percent === "number") p.value = Math.max(0, Math.min(100, percent));
}

function renderRows(rows) {
    const tbody = document.querySelector("#resultsTable tbody");
    tbody.innerHTML = "";
    for (const r of rows) {
        const tr = document.createElement("tr");
        const pill = (label) => `<span class="pill ${label.toLowerCase()}">${label}</span>`;
        tr.innerHTML = `
            <td>${r.idx + 1}</td>
            <td>${r.text.replace(/</g, "&lt;")}</td>
            <td>${r.lang || "-"}</td>
            <td>${r.lstm ? pill(r.lstm.label) : "-"}</td>
            <td>${r.lstm ? `<div class=\"conf-bar\"><span style=\"width:${Math.round(r.lstm.score*100)}%\"></span></div> ${(r.lstm.score).toFixed(3)}` : "-"}</td>
        `;
        tbody.appendChild(tr);
    }
}

function summarize(outputs) {
    const init = () => ({ Positive: 0, Neutral: 0, Negative: 0 });
    const sum = { lstm: init() };
    for (const r of outputs) {
        if (r.lstm && sum.lstm[r.lstm.label] !== undefined) sum.lstm[r.lstm.label]++;
    }
    return sum;
}

function renderSummary(outputs) {
    const sum = summarize(outputs);
    const lstmCounts = document.getElementById("lstmCounts");
    if (lstmCounts) lstmCounts.textContent = `Pos ${sum.lstm.Positive} · Neu ${sum.lstm.Neutral} · Neg ${sum.lstm.Negative}`;

    const labels = ["Positive", "Neutral", "Negative"];
    const lstmData = [sum.lstm.Positive, sum.lstm.Neutral, sum.lstm.Negative];

    const palette = {
        Positive: "#51cf66",
        Neutral: "#cdd7e3",
        Negative: "#ff6b6b"
    };

    const lc = document.getElementById("lstmChart");
    if (lc && window.Chart) {
        if (lstmChart) lstmChart.destroy();
        lstmChart = new Chart(lc, {
            type: "bar",
            data: { labels, datasets: [{ label: "LSTM", data: lstmData, backgroundColor: labels.map(l => palette[l]) }] },
            options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, ticks: { precision:0 } } } }
        });
    }
}

function parseCSVFile(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: (res) => resolve(res),
            error: (err) => reject(err)
        });
    });
}

async function handleAnalyze(rows, opts) {
    setStatus("Loading models...");
    if (opts.useBert) {
        try {
            await ensureBert();
        } catch (e) {
            setStatus("BERT failed to load. Running LSTM only...");
            opts.useBert = false;
        }
    }
    if (opts.useLstm) await ensureLstm();

    const outputs = [];
    const exportBtn = document.getElementById("exportBtn");
    if (exportBtn) exportBtn.disabled = true;
    setStatus("Running inference...");
    setProgress(0, true);
    stopRequested = false;
    const stopBtn = document.getElementById("stopBtn");
    if (stopBtn) stopBtn.disabled = false;

    // Prepare texts
    const prepared = [];
    const maxRowsEl = document.getElementById("maxRows");
    const maxRows = Math.max(1, parseInt(maxRowsEl && maxRowsEl.value ? maxRowsEl.value : "200", 10) || 200);
    for (let i = 0; i < rows.length && prepared.length < maxRows; i++) {
        const textRaw = rows[i].text || rows[i].tweet || rows[i].Tweet || rows[i].Text || "";
        const text = cleanTweet(String(textRaw));
        if (!text) continue;
        const lang = opts.langMode === "auto" ? detectLanguage(text) : null;
        prepared.push({ idx: prepared.length, text, lang });
    }

    // Batched BERT
    const batchSize = 8;
    for (let start = 0; start < prepared.length; start += batchSize) {
        if (stopRequested) break;
        const end = Math.min(start + batchSize, prepared.length);
        const slice = prepared.slice(start, end);
        const texts = slice.map(r => r.text);
        const bertResults = opts.useBert ? await predictBatchWithBert(texts) : new Array(slice.length).fill(null);
        // Parallel LSTM per item (lightweight)
        const lstmResults = await Promise.all(slice.map(r => opts.useLstm ? predictWithLstm(r.text) : Promise.resolve(null)));
        for (let i = 0; i < slice.length; i++) {
            outputs.push({ idx: outputs.length, text: slice[i].text, lang: slice[i].lang, bert: bertResults[i], lstm: lstmResults[i] });
        }
        renderRows(outputs);
        setProgress(Math.round((end / prepared.length) * 100));
        setStatus(`Running inference... ${end}/${prepared.length}`);
        renderSummary(outputs);
    }

    renderRows(outputs);
    setStatus(`Done. Processed ${outputs.length} rows.`);
    setProgress(100);
    setTimeout(() => setProgress(0, false), 400);
    if (exportBtn) exportBtn.disabled = outputs.length === 0;
    window.__lastOutputs = outputs;
    currentOutputs = outputs;
    filteredOutputs = outputs;
    renderSummary(outputs);
    if (stopBtn) stopBtn.disabled = true;
}

async function loadSampleCSV() {
    try {
        const res = await fetch("sample.csv");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
        return parsed;
    } catch (err) {
        console.error("Failed to load sample.csv", err);
        alert("Could not load sample.csv. If you opened this file directly (file://), please run a local server and reload.");
        throw err;
    }
}

function wireUI() {
    const fileInput = document.getElementById("csvInput");
    const runBtn = document.getElementById("runBtn");
    const useBert = document.getElementById("useBert");
    const useLstm = document.getElementById("useLstm");
    const langMode = document.getElementById("langMode");
    const loadSample = document.getElementById("loadSample");
    const bertModel = document.getElementById("bertModel");
    const exportBtn = document.getElementById("exportBtn");
    const stopBtn = document.getElementById("stopBtn");
    const helpBtn = document.getElementById("helpBtn");
    const langFilter = document.getElementById("langFilter");
    const textFilter = document.getElementById("textFilter");
    const onlyDisagreements = document.getElementById("onlyDisagreements");
    const clearFilters = document.getElementById("clearFilters");

    let parsedData = null;

    fileInput.addEventListener("change", async (e) => {
        if (!fileInput.files || fileInput.files.length === 0) return;
        setStatus("Parsing CSV...");
        parsedData = await parseCSVFile(fileInput.files[0]);
        setStatus(`Loaded ${parsedData.data.length} rows.`);
    });

    loadSample.addEventListener("click", async () => {
        setStatus("Loading sample.csv...");
        parsedData = await loadSampleCSV();
        setStatus(`Loaded ${parsedData.data.length} rows from sample.csv.`);
    });

    // Restore persisted settings
    try {
        const savedModel = localStorage.getItem("bert_model_id");
        if (savedModel && bertModel) { bertModel.value = savedModel; bertModelId = savedModel; }
        const savedMax = localStorage.getItem("max_rows");
        if (savedMax) { const el = document.getElementById("maxRows"); if (el) el.value = savedMax; }
        const savedUseBert = localStorage.getItem("use_bert");
        if (savedUseBert !== null && useBert) useBert.checked = savedUseBert === "1";
        const savedUseLstm = localStorage.getItem("use_lstm");
        if (savedUseLstm !== null) useLstm.checked = savedUseLstm === "1";
        const savedLangMode = localStorage.getItem("lang_mode");
        if (savedLangMode && langMode) langMode.value = savedLangMode;
    } catch (e) {}

    runBtn.addEventListener("click", async () => {
        if (!parsedData || !parsedData.data || parsedData.data.length === 0) {
            alert("Please upload a CSV or load the sample first.");
            return;
        }
        // Reset BERT pipeline when model changes
        if (bertModel && bertModel.value !== bertModelId) {
            bertModelId = bertModel.value;
            bertClassifier = null;
        }
        // Persist settings
        try {
            localStorage.setItem("bert_model_id", bertModelId);
            const el = document.getElementById("maxRows");
            if (el) localStorage.setItem("max_rows", String(el.value || ""));
            localStorage.setItem("use_bert", useBert ? (useBert.checked ? "1" : "0") : "0");
            localStorage.setItem("use_lstm", useLstm.checked ? "1" : "0");
            localStorage.setItem("lang_mode", langMode.value);
        } catch (e) {}
        const opts = {
            useBert: !!(useBert && useBert.checked),
            useLstm: useLstm.checked,
            langMode: langMode.value
        };
        await handleAnalyze(parsedData.data, opts);
        populateLangFilter();
        applyFilters();
    });

    if (stopBtn) {
        stopBtn.addEventListener("click", () => {
            stopRequested = true;
            stopBtn.disabled = true;
            setStatus("Stopping...");
        });
    }

    // BERT UI removed; keep settings compatible if elements are present
    if (bertModel) {
        bertModel.addEventListener("change", () => {
            bertModelId = bertModel.value;
            bertClassifier = null;
        });
    }

    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
            const base = filteredOutputs && filteredOutputs.length ? filteredOutputs : (window.__lastOutputs || []);
            const rows = base.map(r => ({
                index: r.idx + 1,
                text: r.text,
                language: r.lang || "",
                bert_label: r.bert ? r.bert.label : "",
                bert_confidence: r.bert ? r.bert.score.toFixed(4) : "",
                lstm_label: r.lstm ? r.lstm.label : "",
                lstm_confidence: r.lstm ? r.lstm.score.toFixed(4) : ""
            }));
            if (rows.length === 0) return;
            const csv = Papa.unparse(rows);
            const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "sentiment_results.csv";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    function populateLangFilter() {
        if (!langFilter) return;
        const set = new Set();
        for (const r of currentOutputs) if (r.lang) set.add(r.lang);
        const cur = langFilter.value;
        langFilter.innerHTML = '<option value="">All</option>' + Array.from(set).sort().map(l => `<option value="${l}">${l}</option>`).join("");
        if ([...set, ""].includes(cur)) langFilter.value = cur;
    }

    function applyFilters() {
        let rows = currentOutputs.slice();
        const lf = langFilter ? langFilter.value : "";
        const q = (textFilter ? textFilter.value : "").trim().toLowerCase();
        const disag = false; // disagreements toggle removed
        if (lf) rows = rows.filter(r => r.lang === lf);
        if (q) rows = rows.filter(r => r.text.toLowerCase().includes(q));
        if (disag) rows = rows.filter(r => r.bert && r.lstm && r.bert.label !== r.lstm.label);
        filteredOutputs = rows;
        renderRows(rows);
        renderSummary(rows);
        setStatus(`Filtered ${rows.length} of ${currentOutputs.length}`);
        const exportBtn = document.getElementById("exportBtn");
        if (exportBtn) exportBtn.disabled = rows.length === 0;
    }

    if (langFilter) langFilter.addEventListener("change", applyFilters);
    if (textFilter) textFilter.addEventListener("input", applyFilters);
    // disagreements toggle removed
    if (clearFilters) clearFilters.addEventListener("click", () => {
        if (langFilter) langFilter.value = "";
        if (textFilter) textFilter.value = "";
        if (onlyDisagreements) onlyDisagreements.checked = false;
        applyFilters();
    });

    if (helpBtn) helpBtn.addEventListener("click", () => {
        alert("Steps:\n1) Upload CSV or load sample\n2) Choose models/options\n3) Analyze\n\nCSV expects a column named 'text' (fallbacks: Tweet, tweet, Text). All processing is local in your browser.");
    });
}

document.addEventListener("DOMContentLoaded", () => {
    wireUI();
    if (location.protocol === "file:") {
        setStatus("Warning: Running from file://. Use a local server for sample/model loading.");
    }
});


