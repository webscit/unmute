import { describe, it, expect } from "vitest";
import {
  float32ToInt16,
  int16ToFloat32,
  pcm16ToBase64,
  base64ToPcm16,
  resample,
} from "@/lib/audio/audio-utils";

// ---------------------------------------------------------------------------
// float32 ↔ int16 round-trips
// ---------------------------------------------------------------------------

describe("float32ToInt16 / int16ToFloat32", () => {
  it("round-trips a simple signal", () => {
    const original = new Float32Array([0, 0.5, -0.5, 1, -1]);
    const int16 = float32ToInt16(original);
    const recovered = int16ToFloat32(int16);

    for (let i = 0; i < original.length; i++) {
      expect(recovered[i]).toBeCloseTo(original[i], 3);
    }
  });

  it("clamps values outside [-1, 1]", () => {
    const input = new Float32Array([2, -2, 1.5, -1.5]);
    const int16 = float32ToInt16(input);
    const recovered = int16ToFloat32(int16);

    // All should be clamped to [-1, 1]
    for (let i = 0; i < recovered.length; i++) {
      expect(recovered[i]).toBeGreaterThanOrEqual(-1);
      expect(recovered[i]).toBeLessThanOrEqual(1);
    }
    // +2 clamps to +1, -2 clamps to -1
    expect(recovered[0]).toBeCloseTo(1, 3);
    expect(recovered[1]).toBeCloseTo(-1, 3);
  });

  it("preserves max positive amplitude (1.0)", () => {
    const input = new Float32Array([1.0]);
    const int16 = float32ToInt16(input);
    expect(int16[0]).toBe(0x7fff);
    const back = int16ToFloat32(int16);
    expect(back[0]).toBeCloseTo(1.0, 3);
  });

  it("preserves max negative amplitude (-1.0)", () => {
    const input = new Float32Array([-1.0]);
    const int16 = float32ToInt16(input);
    expect(int16[0]).toBe(-0x8000);
    const back = int16ToFloat32(int16);
    expect(back[0]).toBeCloseTo(-1.0, 3);
  });
});

// ---------------------------------------------------------------------------
// base64 round-trips
// ---------------------------------------------------------------------------

describe("pcm16ToBase64 / base64ToPcm16", () => {
  it("round-trips an int16 array", () => {
    const original = new Int16Array([0, 100, -100, 32767, -32768]);
    const b64 = pcm16ToBase64(original);
    const recovered = base64ToPcm16(b64);

    expect(recovered.length).toBe(original.length);
    for (let i = 0; i < original.length; i++) {
      expect(recovered[i]).toBe(original[i]);
    }
  });

  it("produces a valid base64 string", () => {
    const data = new Int16Array([1, 2, 3]);
    const b64 = pcm16ToBase64(data);
    expect(typeof b64).toBe("string");
    // Should not throw when decoded
    expect(() => atob(b64)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

describe("resample", () => {
  it("returns the same array when rates match", () => {
    const input = new Float32Array([0.1, 0.2, 0.3]);
    const result = resample(input, 48000, 48000);
    expect(result).toBe(input); // same reference
  });

  it("downsamples to the correct output length", () => {
    const input = new Float32Array(48000); // 1 second at 48 kHz
    input.fill(0.5);
    const result = resample(input, 48000, 24000);
    expect(result.length).toBe(24000);
  });

  it("upsamples to the correct output length", () => {
    const input = new Float32Array(24000); // 1 second at 24 kHz
    input.fill(0.5);
    const result = resample(input, 24000, 48000);
    expect(result.length).toBe(48000);
  });

  it("preserves constant signal value after resampling", () => {
    const input = new Float32Array(100);
    input.fill(0.75);
    const result = resample(input, 44100, 24000);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeCloseTo(0.75, 5);
    }
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("edge cases", () => {
  it("handles silence (all zeros)", () => {
    const silence = new Float32Array(100);
    const int16 = float32ToInt16(silence);
    const back = int16ToFloat32(int16);

    for (let i = 0; i < back.length; i++) {
      expect(back[i]).toBe(0);
    }
  });

  it("handles empty arrays", () => {
    const empty = new Float32Array(0);
    const int16 = float32ToInt16(empty);
    expect(int16.length).toBe(0);

    const back = int16ToFloat32(int16);
    expect(back.length).toBe(0);

    const b64 = pcm16ToBase64(int16);
    const recovered = base64ToPcm16(b64);
    expect(recovered.length).toBe(0);

    const resampled = resample(empty, 48000, 24000);
    expect(resampled.length).toBe(0);
  });

  it("handles single-sample arrays", () => {
    const single = new Float32Array([0.42]);
    const int16 = float32ToInt16(single);
    expect(int16.length).toBe(1);
    const back = int16ToFloat32(int16);
    expect(back[0]).toBeCloseTo(0.42, 2);
  });
});
