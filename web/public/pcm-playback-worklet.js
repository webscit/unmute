class PcmPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(0);
    this.port.onmessage = (event) => {
      if (event.data.type === "clear") {
        this._buffer = new Float32Array(0);
      } else if (event.data.samples) {
        const incoming = event.data.samples;
        const merged = new Float32Array(this._buffer.length + incoming.length);
        merged.set(this._buffer);
        merged.set(incoming, this._buffer.length);
        this._buffer = merged;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || !output[0]) return true;

    const channel = output[0];
    const needed = channel.length;

    if (this._buffer.length >= needed) {
      channel.set(this._buffer.subarray(0, needed));
      this._buffer = this._buffer.subarray(needed);
    } else {
      // Fill what we have, rest stays silent (zeroed by default)
      channel.set(this._buffer);
      this._buffer = new Float32Array(0);
    }

    return true;
  }
}

registerProcessor("pcm-playback-processor", PcmPlaybackProcessor);
