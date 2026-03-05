class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      // Copy the Float32 samples so the buffer stays valid after transfer
      const samples = input[0].slice();
      this.port.postMessage({ samples });
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
