// Node smoke test for the bindings. Build first:
//   wasm-pack build crates/timestretch-wasm --target nodejs --out-dir pkg-node
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const ts = createRequire(import.meta.url)("../pkg-node/timestretch_wasm.js");
const { ProcessStatus, Session, SessionOptions, Transport, render } = ts;

const RATE = 44100;
const CHANNELS = 2;
const FRAMES = RATE * 2;
const source = new Float32Array(FRAMES * CHANNELS);
for (let i = 0; i < FRAMES; i++) {
  const t = i / RATE;
  const v = 0.3 * Math.sin(2 * Math.PI * 220 * t) + 0.2 * Math.sin(2 * Math.PI * 1760 * t);
  source[i * 2] = v;
  source[i * 2 + 1] = v * 0.8;
}

function options(tempo) {
  const o = new SessionOptions();
  o.sourceSampleRate = RATE;
  o.outputSampleRate = RATE;
  o.channels = CHANNELS;
  o.tempo = tempo;
  return o;
}

// One-shot render: expected length, finite, audible.
const TEMPO = 1.25;
const rendered = render(source, options(TEMPO));
const renderedFrames = rendered.length / CHANNELS;
// The achieved ratio is quantized, so the length is close, not exact.
assert.ok(
  Math.abs(renderedFrames / (FRAMES / TEMPO) - 1) < 0.005,
  `rendered ${renderedFrames} frames`,
);
assert.ok(rendered.every(Number.isFinite));
assert.ok(rendered.some((x) => Math.abs(x) > 0.05));

// Streaming, shaped like a worker feeding an AudioWorklet: collected blocks
// wait in a queue with their stamps and are acknowledged only on "delivery",
// in 128-frame quanta with cumulative receipts.
function stream(session, { onTick } = {}) {
  const block = new Float32Array(1024 * CHANNELS);
  const queue = [];
  const out = [];
  let queued = 0;
  for (let tick = 0; !session.finished; tick++) {
    assert.ok(tick < 1_000_000, "stream did not finish");
    onTick?.(tick);
    if (queued < 4096) {
      const status = session.advance(32);
      assert.notEqual(status, ProcessStatus.Fault, session.fault);
      if (status === ProcessStatus.NeedSource) {
        const request = session.sourceRequest();
        const start = request.startFrame * CHANNELS;
        const end = Math.min(start + request.frames * CHANNELS, source.length);
        if (start === end) session.finishInput();
        else session.supply(request, source.subarray(start, end));
      }
      while (session.nextApplied()) {}
      const stamp = session.readOutput(block);
      if (stamp.frames > 0) {
        queue.push({
          epoch: stamp.epoch,
          start: stamp.startFrame,
          samples: block.slice(0, stamp.frames * CHANNELS),
        });
        queued += stamp.frames;
      }
    }
    // Deliver one render quantum.
    let need = 128;
    while (need > 0 && queue.length > 0) {
      const head = queue[0];
      const have = head.samples.length / CHANNELS;
      const take = Math.min(need, have);
      out.push(head.samples.subarray(0, take * CHANNELS));
      head.start += take;
      head.samples = head.samples.subarray(take * CHANNELS);
      if (take === have) queue.shift();
      need -= take;
      queued -= take;
      session.acknowledge(head.epoch, head.start);
    }
  }
  const total = out.reduce((n, a) => n + a.length, 0);
  const joined = new Float32Array(total);
  let at = 0;
  for (const a of out) {
    joined.set(a, at);
    at += a.length;
  }
  return joined;
}

{
  const o = options(TEMPO);
  o.sourceFrames = FRAMES;
  const session = new Session(o);
  assert.equal(session.channels, CHANNELS);
  assert.equal(session.transport, Transport.Playing);
  const streamed = stream(session);
  // Same adapter, different host partitioning: identical samples.
  assert.deepEqual(streamed, rendered);
  assert.equal(session.position, FRAMES);
  assert.ok(session.renderedFinished);
}

// A scheduled tempo change is applied and reported with achieved ratios.
{
  const o = options(1.0);
  o.sourceFrames = FRAMES;
  const session = new Session(o);
  const applied = [];
  let scheduled = false;
  const block = new Float32Array(1024 * CHANNELS);
  while (!session.finished) {
    const status = session.advance(32);
    assert.notEqual(status, ProcessStatus.Fault, session.fault);
    if (!scheduled && session.commitHorizon > 0) {
      session.schedule(7, session.commitHorizon + RATE / 2, 0.8, 1.0);
      scheduled = true;
    }
    if (status === ProcessStatus.NeedSource) {
      const request = session.sourceRequest();
      const start = request.startFrame * CHANNELS;
      session.supply(request, source.subarray(start, start + request.frames * CHANNELS));
    }
    for (let a; (a = session.nextApplied()); ) applied.push(a);
    const stamp = session.readOutput(block);
    session.acknowledge(stamp.epoch, stamp.startFrame + stamp.frames);
  }
  const change = applied.find((a) => a.commandId === 7);
  assert.ok(change, "scheduled command was never applied");
  assert.ok(Math.abs(change.tempo - 0.8) < 0.01, `achieved tempo ${change.tempo}`);
  assert.ok(change.appliedAt >= change.requestedAt);
}

// Seek starts a new epoch; a request from the old epoch is rejected.
{
  const o = options(1.0);
  o.sourceFrames = FRAMES;
  o.seekable = true;
  const session = new Session(o);
  let stale;
  for (let i = 0; i < 1000 && !stale; i++) {
    if (session.advance(8) === ProcessStatus.NeedSource) stale = session.sourceRequest();
  }
  assert.ok(stale, "session never asked for source");
  const before = session.epoch;
  const epoch = session.seek(RATE);
  assert.ok(epoch > before);
  assert.equal(session.epoch, epoch);
  const chunk = source.subarray(0, stale.frames * CHANNELS);
  assert.throws(() => session.supply(stale, chunk), /StaleSource/);
  const streamed = stream(session);
  assert.ok(Math.abs(streamed.length / CHANNELS - RATE) <= 2, "seek renders the remainder");
  assert.equal(session.position, FRAMES);
}

// Sequential sessions cannot seek; invalid ratios are rejected up front.
assert.throws(() => new Session(options(1.0)).seek(0), /UnsupportedOperation/);
assert.throws(() => new Session(options(8.0)), /InvalidParameter/);
assert.throws(() => render(source, options(0)), /InvalidParameter/);

console.log(`ok: ${renderedFrames} frames rendered, streaming output identical`);
