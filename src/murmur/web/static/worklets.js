// Capture: the AudioContext runs at 16 kHz, so the browser resamples the mic for us.
// Packs 128-sample render quanta into the 512-sample int16 frames the server's VAD expects.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.frameSamples = options.processorOptions.frameSamples;
    this.buffer = new Int16Array(this.frameSamples);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      this.buffer[this.filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.filled === this.frameSamples) {
        this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
        this.buffer = new Int16Array(this.frameSamples);
        this.filled = 0;
      }
    }
    return true;
  }
}

// Playback: a queue of PCM chunks. "clear" drops everything instantly — that is barge-in.
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.playing = false;
    this.port.onmessage = (e) => {
      if (e.data === "clear") {
        this.queue = [];
        this.offset = 0;
      } else {
        this.queue.push(e.data);
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    let written = 0;
    while (written < out.length && this.queue.length > 0) {
      const chunk = this.queue[0];
      const n = Math.min(out.length - written, chunk.length - this.offset);
      out.set(chunk.subarray(this.offset, this.offset + n), written);
      written += n;
      this.offset += n;
      if (this.offset === chunk.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    out.fill(0, written);
    const playing = written > 0;
    if (playing !== this.playing) {
      this.playing = playing;
      this.port.postMessage({ playing });
    }
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
registerProcessor("playback", PlaybackProcessor);
