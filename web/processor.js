/**
 * AudioWorkletProcessor that receives stereo interleaved audio chunks
 * from the main thread and plays them back through a ring buffer.
 */
class TimeStretchProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Ring buffer: ~1 second of stereo audio at 48kHz (covers 44.1k too)
    this.bufferSize = 48000 * 2; // stereo interleaved
    this.ringBuffer = new Float32Array(this.bufferSize);
    this.writePos = 0;
    this.readPos = 0;
    this.samplesAvailable = 0;
    this.running = true;

    this.port.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'audio') {
        this._enqueue(msg.samples);
      } else if (msg.type === 'stop') {
        this.running = false;
      } else if (msg.type === 'reset') {
        this.writePos = 0;
        this.readPos = 0;
        this.samplesAvailable = 0;
        this.running = true;
      }
    };
  }

  _enqueue(samples) {
    const len = samples.length;
    if (len === 0) return;

    // Write into ring buffer, wrapping around
    for (let i = 0; i < len; i++) {
      this.ringBuffer[this.writePos] = samples[i];
      this.writePos = (this.writePos + 1) % this.bufferSize;
    }
    this.samplesAvailable += len;

    // Clamp to buffer size (drop oldest if overflow)
    if (this.samplesAvailable > this.bufferSize) {
      const overflow = this.samplesAvailable - this.bufferSize;
      this.readPos = (this.readPos + overflow) % this.bufferSize;
      this.samplesAvailable = this.bufferSize;
    }
  }

  process(inputs, outputs) {
    if (!this.running) return true;

    const output = outputs[0];
    if (!output || output.length < 2) return true;

    const left = output[0];
    const right = output[1];
    const frames = left.length; // 128 frames per render quantum
    const samplesNeeded = frames * 2; // stereo interleaved

    if (this.samplesAvailable >= samplesNeeded) {
      // Deinterleave from ring buffer to output channels
      for (let i = 0; i < frames; i++) {
        left[i] = this.ringBuffer[this.readPos];
        this.readPos = (this.readPos + 1) % this.bufferSize;
        right[i] = this.ringBuffer[this.readPos];
        this.readPos = (this.readPos + 1) % this.bufferSize;
      }
      this.samplesAvailable -= samplesNeeded;
    } else {
      // Underrun: output silence
      left.fill(0);
      right.fill(0);
      this.port.postMessage({ type: 'underrun' });
    }

    return true;
  }
}

registerProcessor('timestretch-processor', TimeStretchProcessor);
