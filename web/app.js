import init, { TimeStretchNode, detect_bpm, list_presets } from './pkg/timestretch_web.js';

// ── State ──────────────────────────────────────────────
let audioCtx = null;
let workletNode = null;
let processor = null;      // TimeStretchNode (WASM)
let sourceAudio = null;     // Float32Array, stereo interleaved
let sampleRate = 44100;
let readPos = 0;
let loopTimer = null;
let playing = false;
let paused = false;

let userStretchRatio = 1.0;
let pitchSemitones = 0;
let pitchFactor = 1.0;
let currentPreset = '';

// ── DOM refs ───────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseLink = document.getElementById('browse-link');
const infoPanel = document.getElementById('info-panel');
const controlsPanel = document.getElementById('controls-panel');
const transportPanel = document.getElementById('transport-panel');
const waveformPanel = document.getElementById('waveform-panel');
const statusEl = document.getElementById('status');

const infoName = document.getElementById('info-name');
const infoDuration = document.getElementById('info-duration');
const infoSampleRate = document.getElementById('info-samplerate');
const infoChannels = document.getElementById('info-channels');
const infoBpm = document.getElementById('info-bpm');

const stretchSlider = document.getElementById('stretch-slider');
const stretchVal = document.getElementById('stretch-val');
const pitchSlider = document.getElementById('pitch-slider');
const pitchVal = document.getElementById('pitch-val');
const presetSelect = document.getElementById('preset-select');

const btnPlay = document.getElementById('btn-play');
const btnPause = document.getElementById('btn-pause');
const btnStop = document.getElementById('btn-stop');
const positionEl = document.getElementById('position');

const waveformCanvas = document.getElementById('waveform');
const waveformCtx = waveformCanvas.getContext('2d');

// ── Helpers ────────────────────────────────────────────
function formatTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

// ── WASM init ──────────────────────────────────────────
async function initWasm() {
  await init();
  setStatus('Ready — load an audio file to begin');
}

// ── Audio context + worklet setup ──────────────────────
async function ensureAudioContext() {
  if (audioCtx) return;
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule('processor.js');
}

function createWorkletNode() {
  if (workletNode) {
    workletNode.disconnect();
    workletNode.port.postMessage({ type: 'stop' });
  }
  workletNode = new AudioWorkletNode(audioCtx, 'timestretch-processor', {
    outputChannelCount: [2],
    numberOfOutputs: 1,
  });
  workletNode.connect(audioCtx.destination);

  workletNode.port.onmessage = (e) => {
    if (e.data.type === 'underrun' && playing && !paused) {
      // Underrun is expected during startup latency; ignore silently
    }
  };
}

// ── File loading ───────────────────────────────────────
async function loadFile(file) {
  setStatus('Decoding audio...');
  stop();

  await ensureAudioContext();

  const arrayBuffer = await file.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  sampleRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;
  const numFrames = audioBuffer.length;

  // Convert to stereo interleaved
  const left = audioBuffer.getChannelData(0);
  const right = numChannels >= 2 ? audioBuffer.getChannelData(1) : left;
  sourceAudio = new Float32Array(numFrames * 2);
  for (let i = 0; i < numFrames; i++) {
    sourceAudio[i * 2] = left[i];
    sourceAudio[i * 2 + 1] = right[i];
  }

  // Update info
  infoName.textContent = file.name;
  infoDuration.textContent = formatTime(audioBuffer.duration);
  infoSampleRate.textContent = `${sampleRate} Hz`;
  infoChannels.textContent = numChannels === 1 ? 'Mono → Stereo' : `${numChannels}ch`;

  // Detect BPM (on left channel)
  setStatus('Detecting BPM...');
  const bpm = detect_bpm(left, sampleRate);
  infoBpm.textContent = bpm > 0 ? `${bpm.toFixed(1)}` : 'N/A';

  // Show panels
  show(infoPanel);
  show(controlsPanel);
  show(transportPanel);
  show(waveformPanel);

  // Draw waveform
  drawWaveform(left);

  // Create processor
  createProcessor();

  btnPlay.disabled = false;
  btnPause.disabled = true;
  btnStop.disabled = true;

  setStatus('Ready to play');
  updatePosition();
}

// ── Processor management ───────────────────────────────
function createProcessor() {
  const preset = currentPreset || undefined;
  const effectiveRatio = userStretchRatio * pitchFactor;
  processor = new TimeStretchNode(sampleRate, effectiveRatio, preset || undefined);
}

function updateEffectiveRatio() {
  pitchFactor = Math.pow(2, pitchSemitones / 12);
  const effectiveRatio = userStretchRatio * pitchFactor;
  if (processor) {
    processor.set_stretch_ratio(effectiveRatio);
  }
}

// ── Playback ───────────────────────────────────────────
function play() {
  if (!sourceAudio || !processor) return;
  if (paused) {
    paused = false;
    playing = true;
    startLoop();
    updateTransportButtons();
    return;
  }

  // Start fresh
  readPos = 0;
  processor.reset();
  createWorkletNode();
  workletNode.port.postMessage({ type: 'reset' });

  playing = true;
  paused = false;
  startLoop();
  updateTransportButtons();
  setStatus('Playing');
}

function pause() {
  if (!playing) return;
  paused = true;
  stopLoop();
  updateTransportButtons();
  setStatus('Paused');
}

function stop() {
  playing = false;
  paused = false;
  stopLoop();
  readPos = 0;
  if (workletNode) {
    workletNode.port.postMessage({ type: 'stop' });
  }
  if (processor) {
    processor.reset();
  }
  updateTransportButtons();
  updatePosition();
  drawPlayhead(-1);
  if (sourceAudio) setStatus('Stopped');
}

function updateTransportButtons() {
  btnPlay.disabled = !sourceAudio || (playing && !paused);
  btnPause.disabled = !playing || paused;
  btnStop.disabled = !playing && !paused;
}

// ── Processing loop ────────────────────────────────────
const CHUNK_FRAMES = 2048;
const LOOP_INTERVAL_MS = 20;

function startLoop() {
  if (loopTimer) return;
  loopTimer = setInterval(processLoop, LOOP_INTERVAL_MS);
}

function stopLoop() {
  if (loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
  }
}

function processLoop() {
  if (!playing || paused || !sourceAudio || !processor) return;

  // How many source frames to read (adjusted for pitch)
  const rawFrames = Math.round(CHUNK_FRAMES * pitchFactor);
  const chunkSamples = rawFrames * 2; // stereo interleaved

  if (readPos >= sourceAudio.length) {
    // End of source — flush remaining
    const tail = processor.flush();
    if (tail.length > 0) {
      sendToWorklet(tail);
    }
    // Wait a moment for the worklet to drain, then stop
    setTimeout(() => stop(), 200);
    stopLoop();
    return;
  }

  // Get the next chunk with pitch-based resampling
  let chunk;
  if (Math.abs(pitchFactor - 1.0) < 0.001) {
    // No pitch shift — read directly
    const end = Math.min(readPos + chunkSamples, sourceAudio.length);
    chunk = sourceAudio.subarray(readPos, end);
    readPos = end;
  } else {
    // Resample source for pitch shift via linear interpolation
    chunk = resampleChunk(CHUNK_FRAMES);
  }

  // Process through WASM
  const output = processor.process(chunk);

  if (output.length > 0) {
    sendToWorklet(output);
  }

  updatePosition();
  drawPlayhead(readPos / sourceAudio.length);
}

function resampleChunk(outFrames) {
  const result = new Float32Array(outFrames * 2);
  const totalSourceFrames = sourceAudio.length / 2;

  for (let i = 0; i < outFrames; i++) {
    const srcFrame = (readPos / 2) + i * pitchFactor;
    if (srcFrame >= totalSourceFrames - 1) {
      // Past end — fill rest with zeros
      break;
    }
    const idx0 = Math.floor(srcFrame);
    const frac = srcFrame - idx0;
    const idx1 = Math.min(idx0 + 1, totalSourceFrames - 1);

    // Left channel
    result[i * 2] = sourceAudio[idx0 * 2] * (1 - frac) + sourceAudio[idx1 * 2] * frac;
    // Right channel
    result[i * 2 + 1] = sourceAudio[idx0 * 2 + 1] * (1 - frac) + sourceAudio[idx1 * 2 + 1] * frac;
  }

  // Advance read position by the number of source frames consumed
  const framesConsumed = outFrames * pitchFactor;
  readPos += Math.round(framesConsumed) * 2;
  readPos = Math.min(readPos, sourceAudio.length);

  return result;
}

function sendToWorklet(samples) {
  if (!workletNode) return;
  // Copy so we can transfer
  const copy = new Float32Array(samples);
  workletNode.port.postMessage({ type: 'audio', samples: copy }, [copy.buffer]);
}

// ── Position display ───────────────────────────────────
function updatePosition() {
  if (!sourceAudio) return;
  const totalFrames = sourceAudio.length / 2;
  const currentFrame = readPos / 2;
  const currentSec = currentFrame / sampleRate;
  const totalSec = totalFrames / sampleRate;
  positionEl.textContent = `${formatTime(currentSec)} / ${formatTime(totalSec)}`;
}

// ── Waveform drawing ───────────────────────────────────
function drawWaveform(channelData) {
  const canvas = waveformCanvas;
  const ctx = waveformCtx;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const mid = h / 2;

  ctx.fillStyle = '#12121f';
  ctx.fillRect(0, 0, w, h);

  ctx.fillStyle = '#00d4ff33';
  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth = 0.5;

  const step = Math.ceil(channelData.length / w);
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const start = x * step;
    let min = 0, max = 0;
    for (let j = 0; j < step && start + j < channelData.length; j++) {
      const val = channelData[start + j];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    const yTop = mid + min * mid;
    const yBot = mid + max * mid;
    ctx.fillRect(x, yTop, 1, yBot - yTop);
  }
  ctx.stroke();

  // Store for playhead overlay
  waveformCanvas._waveformDrawn = true;
}

function drawPlayhead(progress) {
  if (!waveformCanvas._waveformDrawn) return;

  const canvas = waveformCanvas;
  const ctx = waveformCtx;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  // Redraw waveform (we stored channel data indirectly — just redraw the playhead on top)
  // For efficiency, we re-draw the full waveform only when needed
  // Instead, overlay a semi-transparent line
  if (progress < 0) return;

  // We need to redraw — get left channel from sourceAudio
  if (!sourceAudio) return;
  const frames = sourceAudio.length / 2;
  const left = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    left[i] = sourceAudio[i * 2];
  }
  drawWaveform(left);

  if (progress >= 0 && progress <= 1) {
    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.strokeStyle = '#ff4444';
    ctx.lineWidth = 2;
    const x = progress * w;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.restore();
  }
}

// ── Event handlers ─────────────────────────────────────

// File drop
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

browseLink.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) loadFile(file);
});

// Transport
btnPlay.addEventListener('click', play);
btnPause.addEventListener('click', pause);
btnStop.addEventListener('click', stop);

// Controls
stretchSlider.addEventListener('input', () => {
  userStretchRatio = parseFloat(stretchSlider.value);
  stretchVal.textContent = `${userStretchRatio.toFixed(2)}x`;
  updateEffectiveRatio();
});

pitchSlider.addEventListener('input', () => {
  pitchSemitones = parseInt(pitchSlider.value, 10);
  const sign = pitchSemitones > 0 ? '+' : '';
  pitchVal.textContent = `${sign}${pitchSemitones} st`;
  updateEffectiveRatio();
});

presetSelect.addEventListener('change', () => {
  currentPreset = presetSelect.value;
  if (processor) {
    // Recreate processor with new preset
    const wasPlaying = playing;
    stop();
    createProcessor();
    if (wasPlaying) {
      setStatus('Preset changed — click Play to restart');
    }
  }
});

// ── Init ───────────────────────────────────────────────
initWasm().catch((err) => {
  setStatus(`Failed to load WASM: ${err.message}`);
  console.error(err);
});
