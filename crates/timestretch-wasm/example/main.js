// Offline example: decode a file, stretch it in one call, play the result.
// Build the package first (from the repository root):
//   wasm-pack build crates/timestretch-wasm --target web
import init, { SessionOptions, render } from "../pkg/timestretch_wasm.js";

const $ = (id) => document.getElementById(id);
const status = (text) => ($("status").textContent = text);

let context;
let decoded;
let playing;

function interleave(buffer, channels) {
  const out = new Float32Array(buffer.length * channels);
  for (let ch = 0; ch < channels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) out[i * channels + ch] = data[i];
  }
  return out;
}

function deinterleave(samples, channels, sampleRate) {
  const frames = samples.length / channels;
  const buffer = context.createBuffer(channels, frames, sampleRate);
  for (let ch = 0; ch < channels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < frames; i++) data[i] = samples[i * channels + ch];
  }
  return buffer;
}

function stop() {
  playing?.stop();
  playing = undefined;
  $("stop").disabled = true;
}

$("tempo").addEventListener("input", (e) => {
  $("tempo-value").textContent = Number(e.target.value).toFixed(2);
});
$("pitch").addEventListener("input", (e) => {
  $("pitch-value").textContent = e.target.value;
});

$("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  stop();
  $("render").disabled = true;
  status("Decoding…");
  try {
    // The session accepts 32–192 kHz sources, so decode at the device rate.
    context ??= new AudioContext();
    decoded = await context.decodeAudioData(await file.arrayBuffer());
    status(`${file.name}: ${decoded.duration.toFixed(1)} s, ${decoded.sampleRate} Hz`);
    $("render").disabled = false;
  } catch (error) {
    status(`Could not decode: ${error.message}`);
  }
});

$("render").addEventListener("click", async () => {
  stop();
  await context.resume();
  // The library handles mono and stereo; extra channels are dropped here.
  const channels = Math.min(decoded.numberOfChannels, 2);
  const options = new SessionOptions();
  options.sourceSampleRate = decoded.sampleRate;
  options.outputSampleRate = decoded.sampleRate;
  options.channels = channels;
  options.tempo = Number($("tempo").value);
  options.pitch = 2 ** (Number($("pitch").value) / 12);

  status("Rendering…");
  // Let the status paint: render() blocks this thread until it is done.
  // A real application would run it in a Web Worker.
  await new Promise((resolve) => setTimeout(resolve, 20));
  try {
    const started = performance.now();
    const output = render(interleave(decoded, channels), options);
    const seconds = (performance.now() - started) / 1000;
    const buffer = deinterleave(output, channels, decoded.sampleRate);
    status(
      `Rendered ${buffer.duration.toFixed(1)} s in ${seconds.toFixed(2)} s ` +
        `(${(decoded.duration / seconds).toFixed(1)}× realtime)`,
    );
    playing = new AudioBufferSourceNode(context, { buffer });
    playing.connect(context.destination);
    playing.start();
    $("stop").disabled = false;
  } catch (error) {
    status(`Render failed: ${error.message}`);
  } finally {
    options.free();
  }
});

$("stop").addEventListener("click", stop);

await init();
status("Pick an audio file.");
