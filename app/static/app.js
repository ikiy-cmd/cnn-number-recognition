/* global fetch, Image, URL, requestAnimationFrame, setTimeout, clearTimeout */

const ACCENT = "#c9f65a";
const INK = "#171a11";
const PAD_SIZE = 840;
const SRC_SIZE = 28;
const NS = "http://www.w3.org/2000/svg";
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  metrics: null,
  health: null,
  samples: [],
  depth: 10,
  desiredDepth: 10,
  formInitialized: false,
  animating: false,
  animTarget: 10,
  forwardToken: 0,
  forwardTimer: null,
  last: null,
  activations: null,
  currentPixels: null,
  sampleMode: false,
  samplePixels: null,
  drawing: false,
  pointerId: null,
  lastPoint: null,
  brush: 21,
  undo: [],
  predictToken: 0,
  predicting: false,
  recommended: 10,
  training: null,
  trainStatus: null,
  trainingActive: false,
  pollTimer: null,
  firstTrainingPoll: true,
  booted: false,
};

const pad = $("#pad");
const padCtx = pad.getContext("2d", { willReadFrequently: true });
const preview = $("#preview");
const previewCtx = preview.getContext("2d");
const tooltip = $("#chart-tooltip");

const workCanvas = document.createElement("canvas");
workCanvas.width = SRC_SIZE;
workCanvas.height = SRC_SIZE;
const workCtx = workCanvas.getContext("2d", { willReadFrequently: true });

const shiftCanvas = document.createElement("canvas");
shiftCanvas.width = SRC_SIZE;
shiftCanvas.height = SRC_SIZE;
const shiftCtx = shiftCanvas.getContext("2d", { willReadFrequently: true });

/* ------------------------------ small utils ------------------------------ */

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const pad2 = (value) => String(value).padStart(2, "0");
const pct = (value, digits = 2) => (value == null ? "--" : (value * 100).toFixed(digits));
const isFiniteNumber = (value) => typeof value === "number" && Number.isFinite(value);

function formatParams(value) {
  if (!isFiniteNumber(value)) return "--";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1000) return `${Math.round(value / 1000)}K`;
  return String(value);
}

function formatClock(seconds) {
  if (!isFiniteNumber(seconds) || seconds < 0) return "--:--";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${pad2(minutes)}:${pad2(secs)}`;
}

function formatDuration(seconds) {
  if (!isFiniteNumber(seconds)) return "--";
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  if (minutes >= 60) {
    return `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分`;
  }
  return minutes > 0 ? `${minutes} 分 ${secs} 秒` : `${secs} 秒`;
}

function formatStamp(iso) {
  if (!iso) return "--";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "--";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function svgEl(tag, attrs = {}, text = null) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  });
  if (text !== null) node.textContent = text;
  return node;
}

let toastTimer = null;
function toast(message, isError = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("is-error", Boolean(isError));
  node.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("is-visible"), 3600);
}

/* ------------------------------ canvas setup ------------------------------ */

function setBrush(value) {
  state.brush = Number(value);
  $("#brush-value").textContent = String(state.brush);
  $("#brush-range").value = String(state.brush);
}

function padPoint(event) {
  const rect = pad.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * PAD_SIZE,
    y: ((event.clientY - rect.top) / rect.height) * PAD_SIZE,
  };
}

function lineWidth() {
  return (state.brush / 280) * PAD_SIZE;
}

function pushUndo() {
  try {
    state.undo.push(padCtx.getImageData(0, 0, PAD_SIZE, PAD_SIZE));
    if (state.undo.length > 6) state.undo.shift();
  } catch (error) {
    state.undo = [];
  }
  updateUndoButton();
}

function updateUndoButton() {
  $("#undo-btn").disabled = state.undo.length === 0;
}

function showHint(show) {
  $("#canvas-hint").classList.toggle("is-hidden", !show);
}

function beginStroke(event) {
  if (event.button !== undefined && event.button !== 0 && event.pointerType === "mouse") return;
  if (state.drawing) return;
  event.preventDefault();
  pad.setPointerCapture(event.pointerId);
  state.pointerId = event.pointerId;
  state.drawing = true;

  pushUndo();
  if (state.sampleMode) {
    padCtx.clearRect(0, 0, PAD_SIZE, PAD_SIZE);
    state.sampleMode = false;
    state.samplePixels = null;
  }
  const point = padPoint(event);
  state.lastPoint = point;
  padCtx.strokeStyle = INK;
  padCtx.fillStyle = INK;
  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  padCtx.beginPath();
  padCtx.arc(point.x, point.y, lineWidth() / 2, 0, Math.PI * 2);
  padCtx.fill();
  showHint(false);
}

function moveStroke(event) {
  if (!state.drawing || event.pointerId !== state.pointerId) return;
  event.preventDefault();
  const point = padPoint(event);
  const last = state.lastPoint || point;
  padCtx.strokeStyle = INK;
  padCtx.lineWidth = lineWidth();
  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  padCtx.beginPath();
  padCtx.moveTo(last.x, last.y);
  padCtx.lineTo(point.x, point.y);
  padCtx.stroke();
  state.lastPoint = point;
}

function endStroke(event) {
  if (!state.drawing || (event.pointerId !== undefined && event.pointerId !== state.pointerId)) {
    return;
  }
  state.drawing = false;
  state.pointerId = null;
  state.lastPoint = null;
  if (pad.hasPointerCapture?.(event.pointerId)) {
    try {
      pad.releasePointerCapture(event.pointerId);
    } catch (error) {
      /* ignore */
    }
  }
  setTimeout(runCanvasPrediction, 50);
}

function clearPad({ keepUndo = false } = {}) {
  if (!keepUndo) pushUndo();
  padCtx.clearRect(0, 0, PAD_SIZE, PAD_SIZE);
  state.sampleMode = false;
  state.samplePixels = null;
  showHint(true);
  resetPrediction();
  drawPreviewEmpty();
}

function undoPad() {
  const snapshot = state.undo.pop();
  if (snapshot) {
    padCtx.putImageData(snapshot, 0, 0);
    state.sampleMode = false;
    state.samplePixels = null;
    showHint(false);
    updateUndoButton();
    setTimeout(runCanvasPrediction, 30);
  } else {
    padCtx.clearRect(0, 0, PAD_SIZE, PAD_SIZE);
    showHint(true);
    resetPrediction();
    drawPreviewEmpty();
  }
  updateUndoButton();
}

/* -------------------------- pixel normalisation -------------------------- */

function extractPixels(ctx, width, height, threshold = 10) {
  const data = ctx.getImageData(0, 0, width, height).data;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      if (data[(row + x) * 4 + 3] > threshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) return null;

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  if (boxW * boxH < 12) return null;

  const scale = 20 / Math.max(boxW, boxH);
  const drawW = Math.max(1, Math.round(boxW * scale));
  const drawH = Math.max(1, Math.round(boxH * scale));

  workCtx.clearRect(0, 0, SRC_SIZE, SRC_SIZE);
  workCtx.imageSmoothingEnabled = true;
  workCtx.imageSmoothingQuality = "high";
  workCtx.drawImage(
    ctx.canvas,
    minX,
    minY,
    boxW,
    boxH,
    (SRC_SIZE - drawW) / 2,
    (SRC_SIZE - drawH) / 2,
    drawW,
    drawH
  );

  const small = workCtx.getImageData(0, 0, SRC_SIZE, SRC_SIZE).data;
  let mass = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = 0; y < SRC_SIZE; y += 1) {
    for (let x = 0; x < SRC_SIZE; x += 1) {
      const alpha = small[(y * SRC_SIZE + x) * 4 + 3];
      if (alpha > 0) {
        mass += alpha;
        sumX += x * alpha;
        sumY += y * alpha;
      }
    }
  }
  if (mass < 300) return null;

  const dx = clamp(13.5 - sumX / mass, -3.5, 3.5);
  const dy = clamp(13.5 - sumY / mass, -3.5, 3.5);

  shiftCtx.clearRect(0, 0, SRC_SIZE, SRC_SIZE);
  shiftCtx.save();
  shiftCtx.translate(dx, dy);
  shiftCtx.drawImage(workCanvas, 0, 0);
  shiftCtx.restore();

  const finalData = shiftCtx.getImageData(0, 0, SRC_SIZE, SRC_SIZE).data;
  const pixels = new Float32Array(SRC_SIZE * SRC_SIZE);
  let coverage = 0;
  for (let index = 0; index < pixels.length; index += 1) {
    const value = finalData[index * 4 + 3] / 255;
    pixels[index] = value;
    if (value > 0.2) coverage += 1;
  }
  if (coverage < 8) return null;
  return { pixels, coverage };
}

function paintInputCanvas(canvas, pixels) {
  const ctx = canvas.getContext("2d");
  const size = canvas.width;
  const scale = size / SRC_SIZE;
  const image = ctx.createImageData(size, size);
  const data = image.data;
  for (let y = 0; y < size; y += 1) {
    const sy = Math.min(SRC_SIZE - 1, Math.floor(y / scale));
    for (let x = 0; x < size; x += 1) {
      const sx = Math.min(SRC_SIZE - 1, Math.floor(x / scale));
      const value = Math.round((pixels[sy * SRC_SIZE + sx] || 0) * 255);
      const offset = (y * size + x) * 4;
      data[offset] = value;
      data[offset + 1] = value;
      data[offset + 2] = value;
      data[offset + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}

function paintEmptyCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#050705";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawPreviewFromPixels(pixels) {
  paintInputCanvas(preview, pixels);
}

function drawPreviewEmpty() {
  paintEmptyCanvas(preview);
  const featureInput = $("#feature-input");
  if (featureInput) paintEmptyCanvas(featureInput);
}

function drawPixelsToPad(pixels) {
  const temp = document.createElement("canvas");
  temp.width = SRC_SIZE;
  temp.height = SRC_SIZE;
  const tempCtx = temp.getContext("2d");
  const image = tempCtx.createImageData(SRC_SIZE, SRC_SIZE);
  for (let index = 0; index < pixels.length; index += 1) {
    const offset = index * 4;
    image.data[offset] = 23;
    image.data[offset + 1] = 26;
    image.data[offset + 2] = 17;
    image.data[offset + 3] = Math.round(clamp(pixels[index], 0, 1) * 255);
  }
  tempCtx.putImageData(image, 0, 0);

  padCtx.clearRect(0, 0, PAD_SIZE, PAD_SIZE);
  padCtx.imageSmoothingEnabled = true;
  padCtx.imageSmoothingQuality = "high";
  padCtx.drawImage(temp, 0, 0, SRC_SIZE, SRC_SIZE, 0, 0, PAD_SIZE, PAD_SIZE);
}

/* ------------------------------ inference ------------------------------ */

function setResultState(label, mode = "idle") {
  const node = $("#result-state");
  node.textContent = label;
  node.classList.toggle("is-live", mode === "live");
  node.classList.toggle("is-busy", mode === "busy");
}

function resetPrediction() {
  cancelForward({ restoreDepth: true });
  state.last = null;
  state.activations = null;
  state.currentPixels = null;
  state.predicting = false;
  const resultReplay = $("#replay-result");
  if (resultReplay) resultReplay.disabled = true;
  state.predictToken += 1;
  $("#big-digit").textContent = "?";
  $("#big-digit").classList.add("is-empty");
  $("#confidence").textContent = "--";
  $("#confidence-note").textContent = "等待书写";
  setResultState("待输入");
  renderProbChart(null);
  renderMeta(null);
  renderVision();
}

async function sendPrediction(pixels, token = null) {
  const requestToken = token ?? ++state.predictToken;
  if (!token) state.predictToken = requestToken;
  state.predicting = true;
  setResultState("识别中", "busy");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels: Array.from(pixels) }),
    });
    if (!response.ok) {
      let detail = "识别失败";
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (error) {
        /* ignore */
      }
      throw new Error(detail);
    }
    const data = await response.json();
    if (requestToken !== state.predictToken) return;
    state.last = data;
    state.activations = data.activations || null;
    state.currentPixels = pixels;
    state.predicting = false;
    playForward(state.desiredDepth);
  } catch (error) {
    if (requestToken !== state.predictToken) return;
    state.predicting = false;
    state.last = null;
    state.activations = null;
    state.currentPixels = null;
    $("#confidence-note").textContent = "无法识别";
    setResultState("错误");
    renderVision();
    toast(error.message || "识别失败", true);
  }
}

function runCanvasPrediction() {
  if (state.sampleMode) return;
  const extracted = extractPixels(padCtx, PAD_SIZE, PAD_SIZE);
  if (!extracted) {
    resetPrediction();
    drawPreviewEmpty();
    return;
  }
  drawPreviewFromPixels(extracted.pixels);
  sendPrediction(extracted.pixels);
}

function loadSample(sample) {
  const pixels = new Float32Array(sample.pixels.length);
  for (let index = 0; index < sample.pixels.length; index += 1) {
    pixels[index] = sample.pixels[index] / 255;
  }
  pushUndo();
  drawPixelsToPad(pixels);
  drawPreviewFromPixels(pixels);
  state.sampleMode = true;
  state.samplePixels = pixels;
  state.predictToken += 1;
  showHint(false);
  sendPrediction(pixels, state.predictToken);
}

/* ------------------------------ result render ------------------------------ */

function currentDepthMetric() {
  if (!state.metrics?.per_depth) return null;
  return state.metrics.per_depth[state.depth - 1] || null;
}

function renderResult() {
  const metric = currentDepthMetric();
  const data = state.last?.depths?.[state.depth - 1] || null;
  const digitNode = $("#big-digit");
  $("#prob-depth").textContent = `L${pad2(state.depth)}`;
  $("#probe-sub").textContent = metric
    ? `PROBE ${state.depth} · ${metric.channels} CH · ${metric.feature_size} × ${metric.feature_size}`
    : `PROBE ${state.depth}`;

  $("#digit-kicker").textContent = `CONV ${pad2(state.depth)} OUTPUT`;
  if (!data) {
    digitNode.textContent = "?";
    digitNode.classList.add("is-empty");
    $("#confidence").textContent = "--";
    $("#confidence-note").textContent = "等待书写";
    if (metric) setResultState("待输入");
    renderProbChart(null);
    renderMeta(null);
    return;
  }

  digitNode.textContent = String(data.pred);
  digitNode.classList.remove("is-empty");
  if (!state.animating) {
    digitNode.classList.remove("pop");
    void digitNode.offsetWidth;
    digitNode.classList.add("pop");
  }
  $("#confidence").textContent = (data.confidence * 100).toFixed(1);
  $("#confidence-note").textContent = `第 ${state.depth} 层卷积后判读`;
  if (state.animating) {
    setResultState(`前向 ${state.depth}/${state.animTarget}`, "live");
  } else {
    setResultState("已判读", "live");
  }
  renderProbChart(data);
  renderMeta(data);
}

function renderProbChart(data) {
  const host = $("#prob-chart");
  if (!host.children.length) {
    for (let digit = 0; digit < 10; digit += 1) {
      const col = document.createElement("div");
      col.className = "prob-col";
      col.dataset.digit = String(digit);
      const track = document.createElement("div");
      track.className = "prob-track";
      const bar = document.createElement("div");
      bar.className = "prob-bar";
      bar.style.height = "2px";
      track.appendChild(bar);
      const label = document.createElement("div");
      label.className = "prob-digit";
      label.textContent = String(digit);
      col.append(track, label);
      host.appendChild(col);
    }
  }

  const probs = data?.probs || null;
  const top = data ? data.pred : -1;
  $$(".prob-col", host).forEach((col) => {
    const digit = Number(col.dataset.digit);
    const value = probs ? probs[digit] : 0;
    const bar = $(".prob-bar", col);
    bar.style.height = `${Math.max(2, value * 100)}%`;
    bar.style.opacity = value > 0.001 ? "1" : "0.35";
    col.classList.toggle("is-top", digit === top);
  });
}

function renderMeta(data) {
  const metric = currentDepthMetric();
  $("#meta-acc").textContent = metric ? `${pct(metric.test_acc)}%` : "--";
  if (metric) {
    if (metric.depth === 1) {
      $("#meta-delta").textContent = "基线";
    } else {
      const delta = metric.delta * 100;
      $("#meta-delta").textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} pt`;
    }
    $("#meta-feature").textContent = `${metric.channels} ch · ${metric.feature_size}×${metric.feature_size}`;
  } else {
    $("#meta-delta").textContent = "--";
    $("#meta-feature").textContent = "--";
  }
  $("#meta-latency").textContent = state.last ? `${state.last.latency_ms.toFixed(1)} ms` : "--";
}

/* ------------------------------ depth control ------------------------------ */

function buildDepthTicks() {
  const host = $("#depth-ticks");
  host.innerHTML = "";
  for (let depth = 1; depth <= 10; depth += 1) {
    const tick = document.createElement("span");
    tick.textContent = String(depth);
    host.appendChild(tick);
  }
  updateDepthTicks();
}

function updateDepthTicks() {
  $$("#depth-ticks span").forEach((node, index) => {
    const depth = index + 1;
    node.classList.toggle("is-active", depth === state.depth);
    node.classList.toggle("is-future", state.animating && depth > state.depth);
  });
}

function updateLayerActive() {
  $$("#layer-strip .layer-card").forEach((card) => {
    const depth = Number(card.dataset.depth);
    const active = depth === state.depth;
    card.classList.toggle("active", active);
    card.classList.toggle("is-future", state.animating && depth > state.depth);
    card.setAttribute("aria-pressed", String(active));
  });
}

function syncDepthChrome() {
  const slider = $("#depth-range");
  if (slider) slider.value = String(state.depth);
  const output = $("#depth-output");
  if (output) output.textContent = `L${pad2(state.depth)}`;
  updateDepthTicks();
  updateLayerActive();
  updateArchActive();
}

function cancelForward({ restoreDepth = false } = {}) {
  const wasAnimating = state.animating;
  state.forwardToken += 1;
  if (state.forwardTimer) {
    clearTimeout(state.forwardTimer);
    state.forwardTimer = null;
  }
  state.animating = false;
  document.body.classList.remove("is-forwarding");
  const grid = $("#feature-grid");
  if (grid) grid.classList.remove("is-scanning");
  if (restoreDepth && wasAnimating) {
    state.depth = state.desiredDepth;
    syncDepthChrome();
  }
}

function finishForward(token) {
  if (token !== state.forwardToken) return;
  state.forwardTimer = null;
  state.animating = false;
  document.body.classList.remove("is-forwarding");
  const grid = $("#feature-grid");
  if (grid) grid.classList.remove("is-scanning");
  state.depth = state.animTarget;
  syncDepthChrome();
  renderResult();
  renderVision();
  renderChart(false);
  renderMatrix();
}

function playForward(targetDepth = state.desiredDepth) {
  if (!state.last?.depths?.length) return;
  const target = clamp(Number(targetDepth) || 1, 1, state.last.depths.length);
  cancelForward();
  state.animTarget = target;
  state.desiredDepth = target;

  if (REDUCED_MOTION || document.hidden || target <= 1) {
    state.depth = target;
    syncDepthChrome();
    renderResult();
    renderVision();
    renderChart(false);
    renderMatrix();
    return;
  }

  const token = ++state.forwardToken;
  state.animating = true;
  state.depth = 1;
  document.body.classList.add("is-forwarding");
  const stepMs = window.innerWidth < 760 ? 120 : 92;

  const tick = () => {
    if (token !== state.forwardToken) return;
    syncDepthChrome();
    renderResult();
    renderVision();
    if (state.depth >= state.animTarget) {
      state.forwardTimer = setTimeout(() => finishForward(token), stepMs);
      return;
    }
    state.forwardTimer = setTimeout(() => {
      if (token !== state.forwardToken) return;
      state.depth += 1;
      tick();
    }, stepMs);
  };

  tick();
}

function setDepth(depth) {
  const wasAnimating = state.animating;
  cancelForward();
  const max = state.metrics?.per_depth?.length || 10;
  const next = clamp(Number(depth) || 1, 1, max);
  state.desiredDepth = next;
  if (state.depth === next && !wasAnimating) return;
  state.depth = next;
  syncDepthChrome();
  renderResult();
  renderMatrix();
  renderChart(false);
  renderVision();
}

/* ------------------------------ sample strip ------------------------------ */

function renderThumb(canvas, pixels) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#050705";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const temp = document.createElement("canvas");
  temp.width = SRC_SIZE;
  temp.height = SRC_SIZE;
  const tempCtx = temp.getContext("2d");
  const image = tempCtx.createImageData(SRC_SIZE, SRC_SIZE);
  for (let index = 0; index < pixels.length; index += 1) {
    const offset = index * 4;
    image.data[offset] = 233;
    image.data[offset + 1] = 237;
    image.data[offset + 2] = 223;
    image.data[offset + 3] = pixels[index];
  }
  tempCtx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(temp, 0, 0, SRC_SIZE, SRC_SIZE, 4, 4, canvas.width - 8, canvas.height - 8);
}

function renderSamples() {
  const host = $("#sample-strip");
  host.innerHTML = "";
  if (!state.samples.length) {
    const empty = document.createElement("span");
    empty.className = "samples-empty";
    empty.textContent = "样本加载失败";
    host.appendChild(empty);
    return;
  }
  state.samples.forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-btn";
    button.title = `真实测试样本 · 标签 ${sample.label}`;
    button.setAttribute("aria-label", `载入标签为 ${sample.label} 的真实测试样本`);
    const canvas = document.createElement("canvas");
    canvas.width = 68;
    canvas.height = 68;
    renderThumb(canvas, sample.pixels);
    button.appendChild(canvas);
    button.addEventListener("click", () => loadSample(sample));
    host.appendChild(button);
  });
}

/* ------------------------------ layer strip ------------------------------ */

function computeRecommended() {
  const per = state.metrics?.per_depth;
  if (!per?.length) return 10;
  const best = Math.max(...per.map((item) => item.test_acc));
  const found = per.find((item) => item.test_acc >= best - 0.0015);
  return found ? found.depth : per.length;
}

function renderLayers() {
  const host = $("#layer-strip");
  host.innerHTML = "";
  const per = state.metrics?.per_depth;
  if (!per?.length) {
    for (let depth = 1; depth <= 10; depth += 1) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "layer-card";
      card.disabled = true;
      card.innerHTML = `
        <span class="layer-top"><span class="layer-idx">L${pad2(depth)}</span></span>
        <span class="layer-acc">--<em>%</em></span>
        <span class="layer-meta">等待训练</span>
        <span class="layer-bar"><i style="width:0%"></i></span>`;
      host.appendChild(card);
    }
    return;
  }

  state.recommended = computeRecommended();
  const min = Math.min(...per.map((item) => item.test_acc));
  const domainMin = Math.max(0.5, min - 0.02);

  per.forEach((item) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "layer-card";
    card.dataset.depth = String(item.depth);
    const flags = [];
    if (item.depth === state.recommended) flags.push('<span class="flag reco">推荐</span>');
    if (item.pool) flags.push('<span class="flag">POOL</span>');
    const width = clamp(((item.test_acc - domainMin) / (1 - domainMin)) * 100, 3, 100);
    card.innerHTML = `
      <span class="layer-top">
        <span class="layer-idx">L${pad2(item.depth)}</span>
        <span class="layer-flags">${flags.join("")}</span>
      </span>
      <span class="layer-acc">${(item.test_acc * 100).toFixed(2)}<em>%</em></span>
      <span class="layer-meta">${item.channels} ch · ${item.feature_size}²</span>
      <span class="layer-bar"><i style="width:${width.toFixed(1)}%"></i></span>`;
    card.addEventListener("click", () => setDepth(item.depth));
    host.appendChild(card);
  });
  updateLayerActive();
}

/* ------------------------------ chart ------------------------------ */

function renderChart(animate = true) {
  const host = $("#chart");
  host.innerHTML = "";
  tooltip.hidden = true;
  const per = state.metrics?.per_depth;
  if (!per?.length) {
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = "尚无训练指标";
    host.appendChild(empty);
    return;
  }

  const width = Math.max(300, Math.round(host.clientWidth || 900));
  const compact = width < 560;
  const height = compact ? 250 : 320;
  const padL = compact ? 34 : 48;
  const padR = compact ? 12 : 22;
  const padT = compact ? 22 : 28;
  const padB = compact ? 30 : 42;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const accs = per.map((item) => item.test_acc * 100);
  const maxAcc = Math.max(...accs);
  const minAcc = Math.min(...accs);
  let yMin = Math.floor(minAcc - 1);
  if (maxAcc - yMin < 2.5) yMin = Math.floor(minAcc - 1.5);
  yMin = Math.max(0, yMin);
  const yMax = 100;

  const xAt = (index) => padL + (index / (per.length - 1)) * plotW;
  const yAt = (value) => padT + ((yMax - value) / (yMax - yMin)) * plotH;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "每一层卷积后的测试集识别率曲线",
  });
  svg.classList.add("chart-svg");

  const defs = svgEl("defs");
  const gradient = svgEl("linearGradient", {
    id: "chart-area",
    x1: "0",
    y1: "0",
    x2: "0",
    y2: "1",
  });
  gradient.append(
    svgEl("stop", { offset: "0%", "stop-color": ACCENT, "stop-opacity": "0.22" }),
    svgEl("stop", { offset: "100%", "stop-color": ACCENT, "stop-opacity": "0" })
  );
  defs.appendChild(gradient);
  svg.appendChild(defs);

  for (let tick = 0; tick < 5; tick += 1) {
    const value = yMin + ((yMax - yMin) / 4) * tick;
    const y = yAt(value);
    const labelText = (yMax - yMin) / 4 >= 1 ? value.toFixed(0) : value.toFixed(1);
    svg.appendChild(
      svgEl("line", {
        x1: padL,
        y1: y,
        x2: width - padR,
        y2: y,
        stroke: "rgba(231,238,214,0.08)",
        "stroke-width": "1",
      })
    );
    svg.appendChild(
      svgEl(
        "text",
        {
          x: padL - 10,
          y: y + 3,
          "text-anchor": "end",
          fill: "#6b7362",
          "font-size": compact ? "7.5" : "9",
          "font-family": "var(--mono)",
        },
        labelText
      )
    );
  }

  svg.appendChild(
    svgEl("line", {
      x1: padL,
      y1: yAt(yMin),
      x2: width - padR,
      y2: yAt(yMin),
      stroke: "rgba(231,238,214,0.16)",
      "stroke-width": "1",
    })
  );
  if (!compact) {
    svg.appendChild(
      svgEl(
        "text",
        {
          x: padL - 10,
          y: padT - 10,
          "text-anchor": "end",
          fill: "#6b7362",
          "font-size": "8",
          "font-family": "var(--mono)",
        },
        "ACC %"
      )
    );
  }

  const points = per.map((item, index) => ({
    x: xAt(index),
    y: yAt(item.test_acc * 100),
    value: item.test_acc * 100,
    item,
  }));

  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
    .join(" ");
  const areaPath = `${linePath} L${points[points.length - 1].x.toFixed(2)},${(padT + plotH).toFixed(
    2
  )} L${points[0].x.toFixed(2)},${(padT + plotH).toFixed(2)} Z`;

  svg.appendChild(svgEl("path", { d: areaPath, fill: "url(#chart-area)", stroke: "none" }));
  const path = svgEl("path", {
    d: linePath,
    fill: "none",
    stroke: ACCENT,
    "stroke-width": "1.8",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
  });
  svg.appendChild(path);

  const rec = state.recommended;
  const recIndex = clamp(rec - 1, 0, per.length - 1);
  const recX = xAt(recIndex);
  const recoLabel = $("#chart-reco");
  if (recoLabel) recoLabel.textContent = `推荐 L${rec}`;
  const chartSub = $("#chart-sub");
  if (chartSub && state.metrics) {
    const base = `${per.length} probes · ${state.metrics.dataset.test.toLocaleString("en-US")} test images`;
    chartSub.textContent = compact ? `${base} · 推荐 L${rec}` : base;
  }
  svg.appendChild(
    svgEl("line", {
      x1: recX,
      y1: padT,
      x2: recX,
      y2: padT + plotH,
      stroke: "rgba(233,237,223,0.32)",
      "stroke-width": "1",
      "stroke-dasharray": "3 5",
    })
  );

  const activeX = xAt(state.depth - 1);
  svg.appendChild(
    svgEl("line", {
      x1: activeX,
      y1: padT,
      x2: activeX,
      y2: padT + plotH,
      stroke: "rgba(201,246,90,0.34)",
      "stroke-width": "1",
      "stroke-dasharray": "2 4",
    })
  );

  points.forEach((point, index) => {
    const active = index === state.depth - 1;
    svg.appendChild(
      svgEl("circle", {
        cx: point.x,
        cy: point.y,
        r: active ? 5 : 3.2,
        fill: active ? ACCENT : "#0e110d",
        stroke: ACCENT,
        "stroke-width": active ? "2" : "1.5",
      })
    );
    if (!compact || active) {
      svg.appendChild(
        svgEl(
          "text",
          {
            x: point.x,
            y: point.y - (active ? 12 : 10),
            "text-anchor": "middle",
            fill: active ? ACCENT : "#9aa491",
            "font-size": active ? "9.5" : (compact ? "8" : "9"),
            "font-family": "var(--mono)",
            "font-weight": active ? "600" : "400",
          },
          point.value.toFixed(2)
        )
      );
    }
    svg.appendChild(
      svgEl(
        "text",
        {
          x: point.x,
          y: height - 11,
          "text-anchor": "middle",
          fill: active ? ACCENT : "#6b7362",
          "font-size": compact ? "8" : "9",
          "font-family": "var(--mono)",
        },
        String(index + 1)
      )
    );
  });

  const hitLayer = svgEl("g", {});
  points.forEach((point, index) => {
    const hitWidth = compact ? 30 : 64;
    const rect = svgEl("rect", {
      x: point.x - hitWidth / 2,
      y: padT - 10,
      width: hitWidth,
      height: plotH + 20,
      fill: "transparent",
      style: "cursor:pointer",
      "data-index": String(index),
    });
    const showTip = () => {
      const circleRect = svg.getBoundingClientRect();
      const shellRect = $(".chart-shell").getBoundingClientRect();
      const scale = circleRect.width / width;
      const left = circleRect.left - shellRect.left + point.x * scale;
      const top = circleRect.top - shellRect.top + point.y * scale;
      tooltip.hidden = false;
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
      const delta = point.item.delta * 100;
      tooltip.innerHTML = `<span class="tt-title">CONV ${pad2(point.item.depth)}</span>
        测试准确率 <b>${point.value.toFixed(2)}%</b><br>
        较前层 ${delta >= 0 ? "+" : ""}${delta.toFixed(2)} pt`;
    };
    rect.addEventListener("mouseenter", showTip);
    rect.addEventListener("mousemove", showTip);
    rect.addEventListener("mouseleave", () => {
      tooltip.hidden = true;
    });
    rect.addEventListener("click", () => setDepth(index + 1));
    hitLayer.appendChild(rect);
  });
  svg.appendChild(hitLayer);

  host.appendChild(svg);

  if (animate && !REDUCED_MOTION) {
    const length = path.getTotalLength();
    path.style.strokeDasharray = `${length}`;
    path.style.strokeDashoffset = `${length}`;
    path.getBoundingClientRect();
    requestAnimationFrame(() => {
      path.style.transition = "stroke-dashoffset 1.15s cubic-bezier(0.22,1,0.36,1)";
      path.style.strokeDashoffset = "0";
    });
  }
}

/* ------------------------------ matrix ------------------------------ */

function renderMatrix() {
  const host = $("#matrix");
  host.innerHTML = "";
  const metric = currentDepthMetric();
  const confusion = state.metrics?.confusion?.[state.depth - 1];
  $("#matrix-sub").textContent = `第 ${state.depth} 层 · 测试集`;
  if (!metric || !confusion) {
    const empty = document.createElement("div");
    empty.className = "matrix-empty";
    empty.textContent = "尚无混淆矩阵";
    host.appendChild(empty);
    $("#matrix-note").textContent = "对角线越亮，判读越准";
    return;
  }
  $("#matrix-note").textContent = `L${pad2(state.depth)} 正确率 ${pct(metric.test_acc)}%`;

  const grid = document.createElement("div");
  grid.className = "matrix";

  const corner = document.createElement("div");
  corner.className = "mx-axis";
  grid.appendChild(corner);
  for (let col = 0; col < 10; col += 1) {
    const label = document.createElement("div");
    label.className = "mx-axis";
    label.textContent = String(col);
    grid.appendChild(label);
  }

  for (let row = 0; row < 10; row += 1) {
    const rowLabel = document.createElement("div");
    rowLabel.className = "mx-axis";
    rowLabel.textContent = String(row);
    grid.appendChild(rowLabel);
    const rowData = confusion[row];
    const rowMax = Math.max(...rowData, 1);
    for (let col = 0; col < 10; col += 1) {
      const count = rowData[col] || 0;
      const alpha = count === 0 ? 0 : 0.06 + 0.9 * (count / rowMax);
      const cell = document.createElement("div");
      cell.className = "mx-cell";
      cell.textContent = count ? String(count) : "";
      cell.style.background = alpha ? `rgba(201, 246, 90, ${alpha.toFixed(3)})` : "rgba(233,237,223,0.03)";
      cell.style.color = alpha > 0.45 ? "#10130b" : "rgba(233,237,223,0.66)";
      if (row === col) cell.classList.add("diag");
      cell.title = `真实 ${row} → 判读 ${col}：${count} 张`;
      grid.appendChild(cell);
    }
  }
  host.appendChild(grid);
}

/* ------------------------------ cnn vision ------------------------------ */

function heatColor(value) {
  const t = Math.pow(clamp(value, 0, 1), 0.85);
  return [
    Math.round(8 + 193 * t),
    Math.round(12 + 234 * t),
    Math.round(7 + 83 * t),
  ];
}

function updateArchActive() {
  $$("#arch-flow button.arch-node").forEach((node) => {
    const depth = Number(node.dataset.depth);
    const active = depth === state.depth;
    const future = state.animating && depth > state.depth;
    node.classList.toggle("active", active);
    node.classList.toggle("is-future", future);
    node.setAttribute("aria-pressed", String(active));
  });
}

function renderArchitecture() {
  const host = $("#arch-flow");
  if (!host) return;
  const per = state.metrics?.per_depth;
  host.innerHTML = "";
  if (!per?.length) {
    const empty = document.createElement("span");
    empty.className = "arch-empty";
    empty.textContent = "等待训练指标";
    host.appendChild(empty);
    return;
  }

  const input = document.createElement("div");
  input.className = "arch-node input";
  input.innerHTML = '<span class="k">INPUT</span><span class="n">28²</span><span class="s">1 ch</span>';
  host.appendChild(input);

  per.forEach((item, index) => {
    if (index > 0) {
      const link = document.createElement("div");
      link.className = "arch-link";
      const prev = per[index - 1];
      link.innerHTML = `<span>→</span>${prev.pool ? '<span class="pool">POOL</span>' : ""}`;
      host.appendChild(link);
    }
    const node = document.createElement("button");
    node.type = "button";
    node.className = "arch-node";
    node.dataset.depth = String(item.depth);
    node.setAttribute("aria-pressed", "false");
    node.innerHTML = `<span class="k">CONV ${pad2(item.depth)}</span>
      <span class="n">${item.feature_size}²</span>
      <span class="s">${item.channels} ch</span>`;
    node.addEventListener("click", () => setDepth(item.depth));
    host.appendChild(node);
  });

  const link = document.createElement("div");
  link.className = "arch-link";
  link.innerHTML = "<span>→</span>";
  host.appendChild(link);

  const output = document.createElement("div");
  output.className = "arch-node output";
  output.innerHTML = '<span class="k">OUTPUT</span><span class="n">10</span><span class="s">softmax</span>';
  host.appendChild(output);

  updateArchActive();
}

function renderFeatureMaps() {
  const host = $("#feature-grid");
  if (!host) return;
  host.innerHTML = "";
  host.classList.toggle("is-scanning", state.animating);
  const replay = $("#feature-replay");
  if (replay) replay.disabled = !state.last;
  const resultReplay = $("#replay-result");
  if (resultReplay) resultReplay.disabled = !state.last;
  const figure = $("#vision-figure");
  figure.textContent = `L${pad2(state.depth)}`;

  const inputCanvas = $("#feature-input");
  if (state.currentPixels) {
    paintInputCanvas(inputCanvas, state.currentPixels);
  } else {
    paintEmptyCanvas(inputCanvas);
  }

  const activation = state.activations?.[state.depth - 1];
  if (!activation) {
    $("#feature-sub").textContent = "等待一次识别";
    $("#feature-meta").textContent = "8 CH";
    $("#feature-density").textContent = "激活密度 --";
    for (let index = 0; index < 8; index += 1) {
      const tile = document.createElement("div");
      tile.className = "feature-tile is-empty";
      tile.textContent = "CH --";
      host.appendChild(tile);
    }
    return;
  }

  $("#feature-sub").textContent = `第 ${activation.depth} 层 · ${activation.channels} 通道 · ${activation.size} × ${activation.size}`;
  $("#feature-meta").textContent = `${activation.maps.length} CH · ${activation.map_size}²`;
  $("#feature-density").textContent = `激活密度 ${(activation.density * 100).toFixed(1)}%`;

  activation.maps.forEach((map) => {
    const tile = document.createElement("div");
    tile.className = "feature-tile";
    tile.title = `通道 ${map.channel} · 最大激活 ${map.max}`;

    const canvas = document.createElement("canvas");
    canvas.width = activation.map_size;
    canvas.height = activation.map_size;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(activation.map_size, activation.map_size);
    for (let index = 0; index < map.values.length; index += 1) {
      const [r, g, b] = heatColor(map.values[index] / 100);
      const offset = index * 4;
      image.data[offset] = r;
      image.data[offset + 1] = g;
      image.data[offset + 2] = b;
      image.data[offset + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);

    const channelLabel = document.createElement("span");
    channelLabel.className = "ch";
    channelLabel.textContent = `CH ${pad2(map.channel)}`;
    const maxLabel = document.createElement("span");
    maxLabel.className = "mx";
    maxLabel.textContent = map.max.toFixed(1);

    tile.append(canvas, channelLabel, maxLabel);
    host.appendChild(tile);
  });
}

function renderEvolution() {
  const host = $("#evolution");
  if (!host) return;
  host.innerHTML = "";

  if (!state.last?.depths?.length) {
    const empty = document.createElement("div");
    empty.className = "evolution-empty";
    empty.textContent = "完成一次识别后显示逐层概率";
    host.appendChild(empty);
    $("#evolution-note").textContent = "完成一次识别后显示每个深度的概率分布。";
    return;
  }

  const depths = state.last.depths;
  const finalPred = depths[depths.length - 1].pred;
  const finalConfidence = depths[depths.length - 1].confidence;

  const corner = document.createElement("div");
  corner.className = "ev-label ev-head ev-corner";
  host.appendChild(corner);
  for (let digit = 0; digit < 10; digit += 1) {
    const label = document.createElement("div");
    label.className = "ev-label ev-head";
    if (digit === finalPred) label.classList.add("is-winner");
    label.textContent = String(digit);
    host.appendChild(label);
  }

  depths.forEach((item) => {
    const future = state.animating && item.depth > state.depth;
    const rowLabel = document.createElement("div");
    rowLabel.className = "ev-label";
    if (item.depth === state.depth) rowLabel.classList.add("is-active");
    if (future) rowLabel.classList.add("is-future");
    rowLabel.textContent = `L${pad2(item.depth)}`;
    host.appendChild(rowLabel);

    for (let digit = 0; digit < 10; digit += 1) {
      const probability = item.probs[digit] || 0;
      const cell = document.createElement("div");
      cell.className = "ev-cell";
      if (item.depth === state.depth) cell.classList.add("is-active-row");
      if (future) cell.classList.add("is-future");
      const alpha = 0.015 + 0.86 * Math.pow(probability, 0.72);
      cell.style.background = `rgba(201, 246, 90, ${alpha.toFixed(3)})`;
      cell.title = `第 ${item.depth} 层 · 数字 ${digit} · ${(probability * 100).toFixed(1)}%`;
      host.appendChild(cell);
    }
  });

  if (state.animating) {
    $("#evolution-note").innerHTML = `正在前向传播 · 第 <b>${state.depth}</b> / ${state.animTarget} 层`;
    return;
  }

  const firstMatch = depths.find((item) => item.pred === finalPred)?.depth || finalPred;
  let flips = 0;
  for (let index = 1; index < depths.length; index += 1) {
    if (depths[index].pred !== depths[index - 1].pred) flips += 1;
  }
  $("#evolution-note").innerHTML = `第 ${firstMatch} 层首次指向 <b>${finalPred}</b>，十层之间改判 ${flips} 次。最终置信度 ${(finalConfidence * 100).toFixed(1)}%。`;
}

function renderVision() {
  renderFeatureMaps();
  renderEvolution();
  updateArchActive();
}

/* ------------------------------ summary / facts ------------------------------ */

function renderSummary() {
  const metrics = state.metrics;
  if (!metrics) {
    $("#stat-acc").textContent = "--";
    $("#stat-params").textContent = "--";
    $("#best-acc").textContent = "--";
    $("#chart-sub").textContent = "等待训练指标";
    $("#data-total").textContent = "--";
    $("#data-split").textContent = "--";
    $("#model-params").textContent = "--";
    $("#model-runtime").textContent = "--";
    $("#last-run").textContent = "--";
    return;
  }

  const best = Math.max(...metrics.per_depth.map((item) => item.test_acc));
  $("#stat-acc").textContent = pct(metrics.test_acc);
  $("#stat-params").textContent = formatParams(metrics.total_params);
  $("#best-acc").textContent = `${pct(best)}%`;
  $("#chart-sub").textContent = `${metrics.per_depth.length} probes · ${metrics.dataset.test.toLocaleString(
    "en-US"
  )} test images`;
  $("#data-total").textContent = "70,000";
  $("#data-split").textContent = `${metrics.dataset.train.toLocaleString("en-US")} / ${metrics.dataset.val.toLocaleString(
    "en-US"
  )} / ${metrics.dataset.test.toLocaleString("en-US")}`;
  $("#model-params").textContent = `${formatParams(metrics.total_params)} PARAMS`;
  $("#model-runtime").textContent = `PyTorch ${metrics.torch_version} · ${metrics.threads}T`;
  $("#last-run").textContent = formatStamp(metrics.created_at);
}

function renderTrainSummary() {
  const host = $("#train-summary");
  if (!host) return;
  const values = host.querySelectorAll("div b");
  if (!values.length) return;
  const metrics = state.metrics;
  if (!metrics) {
    values.forEach((node) => {
      node.textContent = "--";
    });
    return;
  }
  values[0].textContent = `${metrics.config.epochs} 轮`;
  values[1].textContent = metrics.dataset.train.toLocaleString("en-US");
  values[2].textContent = String(metrics.config.batch_size);
  values[3].textContent = formatDuration(metrics.elapsed_s);
}

function renderHistory() {
  const host = $("#history-list");
  host.innerHTML = "";
  const history = state.metrics?.history || [];
  if (!history.length) {
    const empty = document.createElement("span");
    empty.className = "history-empty";
    empty.textContent = "尚无训练记录";
    host.appendChild(empty);
    return;
  }
  const rows = history.slice(-6);
  rows.forEach((record) => {
    const row = document.createElement("div");
    row.className = "history-row";
    const label = document.createElement("span");
    label.textContent = `EP ${pad2(record.epoch)}`;
    const bar = document.createElement("div");
    bar.className = "history-bar";
    const fill = document.createElement("i");
    const width = clamp((record.val_acc * 100 - 90) * 10, 3, 100);
    fill.style.width = `${width.toFixed(1)}%`;
    bar.appendChild(fill);
    const value = document.createElement("span");
    value.textContent = `${(record.val_acc * 100).toFixed(2)}%`;
    row.append(label, bar, value);
    host.appendChild(row);
  });
}

function renderTrainingFacts() {
  renderSummary();
  renderHistory();
  updateEstimate();
}

function updateEstimate() {
  const epochs = Number($("#epochs").value) || 6;
  const size = Number($("#train-size").value) || 55000;
  let base = 42;
  if (state.metrics?.history?.length) {
    const durations = state.metrics.history.map((item) => item.seconds).filter(isFiniteNumber);
    if (durations.length) {
      base = durations.reduce((sum, value) => sum + value, 0) / durations.length;
    }
  }
  const threads = state.health?.cpu_count || 16;
  const seconds = (base * epochs * size) / 55000 + 12;
  $("#train-estimate").textContent = `预计用时 ${formatDuration(seconds)} · 本机 ${threads} 线程 CPU`;
}

/* ------------------------------ training ------------------------------ */

const TRAIN_LABELS = {
  idle: "空闲",
  starting: "启动中",
  running: "训练中",
  evaluating: "评估中",
  done: "已完成",
  stopped: "已停止",
  interrupted: "已中断",
  error: "出错",
  unavailable: "不可用",
};

function trainingIsActive(training) {
  if (!training) return false;
  return Boolean(
    training.active && ["starting", "running", "evaluating"].includes(training.status)
  );
}

function applyTraining(training, { initial = false } = {}) {
  const previous = state.trainStatus;
  state.training = training;
  state.trainStatus = training?.status || "idle";
  state.trainingActive = trainingIsActive(training);
  renderTraining(training);
  updateHeaderStatus();

  if (initial) return;

  if (previous !== state.trainStatus) {
    if (training.status === "done") {
      toast("训练完成，模型已热加载");
      reloadAfterTraining();
    } else if (training.status === "error") {
      toast(training.message || "训练失败", true);
    } else if (training.status === "stopped") {
      toast("训练已停止，保留当前模型");
    } else if (training.status === "interrupted") {
      toast("训练进程已中断，保留当前模型", true);
    }
  }
}

async function reloadAfterTraining() {
  await loadHealth();
  await loadMetrics();
  if (state.sampleMode && state.samplePixels) {
    sendPrediction(state.samplePixels);
    return;
  }
  if (state.last) {
    const extracted = extractPixels(padCtx, PAD_SIZE, PAD_SIZE);
    if (extracted) {
      drawPreviewFromPixels(extracted.pixels);
      sendPrediction(extracted.pixels);
    }
  }
}

function renderTraining(training) {
  if (!training) return;
  const rawStatus = training.status || "idle";
  const active = trainingIsActive(training);
  const status =
    !active && ["starting", "running", "evaluating"].includes(rawStatus)
      ? "interrupted"
      : rawStatus;
  const statusNode = $("#train-status");
  statusNode.classList.toggle("is-running", active);
  statusNode.classList.toggle("is-error", status === "error");
  statusNode.classList.toggle("is-done", status === "done");
  $("span", statusNode).textContent = TRAIN_LABELS[status] || status;

  $("#train-start").disabled = active;
  $("#epochs").disabled = active;
  $("#train-size").disabled = active;
  if (active) {
    const epochSelect = $("#epochs");
    const sizeSelect = $("#train-size");
    if (training.epochs && Array.from(epochSelect.options).some((option) => option.value === String(training.epochs))) {
      epochSelect.value = String(training.epochs);
    }
    if (
      training.train_size &&
      Array.from(sizeSelect.options).some((option) => option.value === String(training.train_size))
    ) {
      sizeSelect.value = String(training.train_size);
    }
    updateEstimate();
  }
  $("#stop-field").classList.toggle("is-hidden", !active);
  $("#train-form").querySelector(".field-action:not(#stop-field)").classList.toggle("is-hidden", active);
  $("#train-sub").textContent = active
    ? "训练正在进行，页面可以保持打开"
    : status === "done"
      ? "训练完成，模型已热加载"
      : "选择轮次与样本量后开始";

  const totalSteps = Number(training.total_steps) || 0;
  const step = Number(training.global_step) || 0;
  let progress = totalSteps ? (step / totalSteps) * 100 : 0;
  if (status === "done") progress = 100;
  const fill = $("#progress-fill");
  fill.style.width = `${clamp(progress, 0, 100)}%`;
  fill.classList.toggle("is-indeterminate", status === "evaluating");
  $("#progress-percent").textContent = `${Math.round(clamp(progress, 0, 100))}%`;

  $("#progress-epoch").textContent = `EPOCH ${training.epoch || 0} / ${training.epochs || "--"}`;
  $("#progress-batch").textContent = `BATCH ${training.step || 0} / ${training.steps_per_epoch || "--"}`;
  $("#progress-loss").textContent = isFiniteNumber(training.loss)
    ? `LOSS ${Number(training.loss).toFixed(4)}`
    : "LOSS --";
  $("#progress-eta").textContent = `ETA ${formatClock(training.eta_s)}`;
  const messageMap = {
    idle: "尚未开始训练",
    evaluating: "正在 10,000 张测试集上评估 10 个探针",
    done: "训练完成，模型已保存并热加载",
    stopped: "训练已停止，旧模型继续服务",
    interrupted: "训练被中断",
  };
  $("#progress-message").textContent =
    messageMap[status] || training.message || "尚未开始训练";

  const log = training.log?.length ? training.log.join("\n") : "等待训练任务。";
  const logNode = $("#log");
  if (logNode.textContent !== log) {
    logNode.textContent = log;
    logNode.scrollTop = logNode.scrollHeight;
  }
  $("#log-state").textContent = String(status).toUpperCase();

  updateHeaderStatus();
}

function updateHeaderStatus() {
  const node = $("#model-status");
  const text = $("#model-status-text");
  const training = state.training;
  const active = state.trainingActive;
  node.classList.remove("is-training", "is-idle", "is-error");
  if (active && training) {
    const total = Number(training.total_steps) || 0;
    const step = Number(training.global_step) || 0;
    const value = total ? Math.round((step / total) * 100) : 0;
    text.textContent = `训练中 ${value}%`;
    node.classList.add("is-training");
  } else if (state.health?.model_loaded) {
    text.textContent = "模型就绪";
  } else if (state.trainStatus === "error") {
    text.textContent = "训练出错";
    node.classList.add("is-error");
  } else {
    text.textContent = "模型未就绪";
    node.classList.add("is-idle");
  }
}

async function pollTraining() {
  clearTimeout(state.pollTimer);
  try {
    const response = await fetch("/api/train/status", { cache: "no-store" });
    if (!response.ok) throw new Error("status unavailable");
    const training = await response.json();
    applyTraining(training, { initial: state.firstTrainingPoll });
    state.firstTrainingPoll = false;
  } catch (error) {
    state.firstTrainingPoll = false;
  }
  const delay = state.trainingActive ? 1200 : 5000;
  state.pollTimer = setTimeout(pollTraining, delay);
}

async function startTraining(event) {
  event.preventDefault();
  const epochs = Number($("#epochs").value);
  const trainSize = Number($("#train-size").value);
  const button = $("#train-start");
  button.disabled = true;
  try {
    const response = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs, train_size: trainSize, augment: true }),
    });
    if (!response.ok) {
      let detail = "无法启动训练";
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (error) {
        /* ignore */
      }
      throw new Error(detail);
    }
    toast("训练已启动，模型会在完成后自动加载");
    setTimeout(pollTraining, 300);
  } catch (error) {
    button.disabled = false;
    toast(error.message || "无法启动训练", true);
  }
}

async function stopTraining() {
  const button = $("#train-stop");
  button.disabled = true;
  try {
    const response = await fetch("/api/train/stop", { method: "POST" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || "停止失败");
    toast(body.message || "训练已停止");
    setTimeout(pollTraining, 300);
  } catch (error) {
    toast(error.message || "停止失败", true);
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------ data loading ------------------------------ */

async function loadHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("health failed");
    state.health = await response.json();
  } catch (error) {
    state.health = null;
  }
  updateHeaderStatus();
  updateEstimate();
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics", { cache: "no-store" });
    if (!response.ok) {
      state.metrics = null;
      renderAll();
      return;
    }
    state.metrics = await response.json();
    const maxDepth = state.metrics.per_depth.length;
    if (state.depth > maxDepth) state.depth = maxDepth;
    if (state.desiredDepth > maxDepth) state.desiredDepth = maxDepth;
    $("#depth-range").max = String(maxDepth);
    renderAll();
  } catch (error) {
    state.metrics = null;
    renderAll();
    toast("无法读取模型指标", true);
  }
}

async function loadSamples() {
  try {
    const response = await fetch("/api/samples", { cache: "no-store" });
    if (!response.ok) throw new Error("samples failed");
    const body = await response.json();
    state.samples = body.samples || [];
  } catch (error) {
    state.samples = [];
  }
  renderSamples();
}

async function loadImageFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    toast("请选择图片文件", true);
    return;
  }
  const url = URL.createObjectURL(file);
  try {
    const image = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("图片读取失败"));
      img.src = url;
    });

    const size = 560;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, size, size);
    const scale = Math.min((size * 0.86) / image.width, (size * 0.86) / image.height);
    const drawW = Math.max(1, image.width * scale);
    const drawH = Math.max(1, image.height * scale);
    ctx.drawImage(image, (size - drawW) / 2, (size - drawH) / 2, drawW, drawH);

    const data = ctx.getImageData(0, 0, size, size).data;
    const originX = (size - drawW) / 2;
    const originY = (size - drawH) / 2;
    let borderSum = 0;
    let borderCount = 0;
    const sampleBorder = (x, y) => {
      const px = clamp(Math.round(x), 0, size - 1);
      const py = clamp(Math.round(y), 0, size - 1);
      const offset = (py * size + px) * 4;
      borderSum += (data[offset] + data[offset + 1] + data[offset + 2]) / 3;
      borderCount += 1;
    };
    const inset = Math.min(4, drawW / 8, drawH / 8);
    for (let step = 0; step <= 24; step += 1) {
      const x = originX + inset + ((drawW - inset * 2) * step) / 24;
      const y = originY + inset + ((drawH - inset * 2) * step) / 24;
      sampleBorder(x, originY + inset);
      sampleBorder(x, originY + drawH - inset);
      sampleBorder(originX + inset, y);
      sampleBorder(originX + drawW - inset, y);
    }
    const lightBackground = borderSum / Math.max(1, borderCount) > 127;

    const alphaCanvas = document.createElement("canvas");
    alphaCanvas.width = size;
    alphaCanvas.height = size;
    const alphaCtx = alphaCanvas.getContext("2d", { willReadFrequently: true });
    const alphaImage = alphaCtx.createImageData(size, size);
    for (let pixel = 0; pixel < size * size; pixel += 1) {
      const offset = pixel * 4;
      const lum = (data[offset] * 0.299 + data[offset + 1] * 0.587 + data[offset + 2] * 0.114) / 255;
      let ink = lightBackground ? 1 - lum : lum;
      ink = clamp((ink - 0.08) / 0.82, 0, 1);
      alphaImage.data[offset] = 23;
      alphaImage.data[offset + 1] = 26;
      alphaImage.data[offset + 2] = 17;
      alphaImage.data[offset + 3] = Math.round(ink * 255);
    }
    alphaCtx.putImageData(alphaImage, 0, 0);

    const extracted = extractPixels(alphaCtx, size, size, 24);
    if (!extracted) {
      toast("没有在图片里找到清晰的数字", true);
      return;
    }
    pushUndo();
    drawPixelsToPad(extracted.pixels);
    drawPreviewFromPixels(extracted.pixels);
    state.sampleMode = true;
    state.samplePixels = extracted.pixels;
    state.predictToken += 1;
    showHint(false);
    sendPrediction(extracted.pixels, state.predictToken);
    toast("图片已载入，并自动居中归一化");
  } catch (error) {
    toast(error.message || "图片读取失败", true);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/* ------------------------------ nav highlight ------------------------------ */

function setupNavHighlight() {
  const links = $$(".nav-link");
  const sections = ["lab", "layers", "vision", "train"]
    .map((id) => document.getElementById(id))
    .filter(Boolean);
  if (!sections.length) return;

  let ticking = false;
  const update = () => {
    ticking = false;
    const atBottom =
      window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
    const probe = window.scrollY + Math.min(320, window.innerHeight * 0.38);
    let current = sections[0].id;
    sections.forEach((section) => {
      const top = section.getBoundingClientRect().top + window.scrollY;
      if (top <= probe) current = section.id;
    });
    if (atBottom) current = sections[sections.length - 1].id;
    links.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${current}`);
    });
  };
  const request = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(update);
  };
  window.addEventListener("scroll", request, { passive: true });
  window.addEventListener("resize", request);
  update();
}

/* ------------------------------ boot ------------------------------ */

function renderAll() {
  const maxDepth = state.metrics?.per_depth?.length || 10;
  state.depth = clamp(state.depth, 1, maxDepth);
  state.desiredDepth = clamp(state.desiredDepth, 1, maxDepth);
  $("#depth-range").max = String(maxDepth);
  $("#depth-range").value = String(state.depth);
  $("#depth-output").textContent = `L${pad2(state.depth)}`;
  updateDepthTicks();

  if (state.metrics && !state.formInitialized) {
    const epochsSelect = $("#epochs");
    const sizeSelect = $("#train-size");
    const epochs = String(state.metrics.config.epochs);
    const size = String(state.metrics.dataset.train);
    if (Array.from(epochsSelect.options).some((option) => option.value === epochs)) {
      epochsSelect.value = epochs;
    }
    if (Array.from(sizeSelect.options).some((option) => option.value === size)) {
      sizeSelect.value = size;
    }
    state.formInitialized = true;
  }

  renderSummary();
  renderLayers();
  renderChart(false);
  renderMatrix();
  renderHistory();
  renderTrainSummary();
  renderResult();
  renderArchitecture();
  renderVision();
  updateEstimate();
  if (!state.metrics) {
    $("#probe-sub").textContent = "PROBE --";
  }
}

async function boot() {
  drawPreviewEmpty();
  buildDepthTicks();
  setBrush(21);
  updateUndoButton();

  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  pad.addEventListener("pointerdown", beginStroke);
  pad.addEventListener("pointermove", moveStroke);
  pad.addEventListener("pointerup", endStroke);
  pad.addEventListener("pointercancel", endStroke);
  pad.addEventListener("pointerleave", (event) => {
    if (state.drawing && event.buttons === 0) endStroke(event);
  });

  $("#clear-btn").addEventListener("click", () => clearPad());
  $("#undo-btn").addEventListener("click", undoPad);
  $("#feature-replay").addEventListener("click", () => {
    if (state.last) playForward(state.desiredDepth);
  });
  $("#replay-result").addEventListener("click", () => {
    if (state.last) playForward(state.desiredDepth);
  });
  $("#brush-range").addEventListener("input", (event) => setBrush(event.target.value));
  $("#depth-range").addEventListener("input", (event) => setDepth(event.target.value));
  $("#train-form").addEventListener("submit", startTraining);
  $("#train-stop").addEventListener("click", stopTraining);
  $("#epochs").addEventListener("change", updateEstimate);
  $("#train-size").addEventListener("change", updateEstimate);
  $("#file-input").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) loadImageFile(file);
  });
  $(".file-btn").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("#file-input").click();
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) pollTraining();
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => renderChart(false), 180);
  });

  setupNavHighlight();

  await Promise.all([loadHealth(), loadMetrics()]);
  await loadSamples();
  renderAll();
  state.booted = true;
  pollTraining();
}

boot();
