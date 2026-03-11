import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRealtimeSession } from "@/hooks/useRealtimeSession";

// ---------------------------------------------------------------------------
// Mock the realtime-client module
// ---------------------------------------------------------------------------

// We build a fake OpenAIRealtimeWebSocket that stores event listeners and
// lets us fire events imperatively.
function createMockRtClient() {
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const rawSocket = {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };

  return {
    on(event: string, cb: (...args: unknown[]) => void) {
      if (!listeners.has(event)) listeners.set(event, []);
      listeners.get(event)!.push(cb);
    },
    send: vi.fn(),
    close: vi.fn(),
    socket: rawSocket,
    // Test helper: emit an event on the mock client
    _emit(event: string, ...args: unknown[]) {
      for (const cb of listeners.get(event) ?? []) {
        cb(...args);
      }
    },
    _listeners: listeners,
  };
}

type MockRtClient = ReturnType<typeof createMockRtClient>;

let mockRt: MockRtClient;
let mockOnUnmuteCleanup: Mock;

vi.mock("@/lib/realtime-client", () => ({
  createRealtimeClient: vi.fn(async () => mockRt),
  buildSessionUpdate: vi.fn(() => ({ mocked: true })),
  onUnmuteEvent: vi.fn(() => {
    mockOnUnmuteCleanup = vi.fn();
    return mockOnUnmuteCleanup;
  }),
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const defaultSettings = {
  voice: "alloy",
  customInstructions: "",
  backendUrl: "ws://localhost:8000",
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useRealtimeSession", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockRt = createMockRtClient();
  });

  // -----------------------------------------------------------------------
  // 1. Connection lifecycle
  // -----------------------------------------------------------------------
  describe("connection lifecycle", () => {
    it("starts disconnected, transitions to connecting → connected", async () => {
      const { result } = renderHook(() => useRealtimeSession());
      expect(result.current.connectionStatus).toBe("disconnected");

      // Start connecting
      let connectPromise: Promise<void>;
      act(() => {
        connectPromise = result.current.connect(defaultSettings);
      });
      await act(async () => {
        await connectPromise;
      });

      // After createRealtimeClient resolves, status should be "connecting"
      // (it becomes "connected" only after session.updated fires)
      expect(result.current.connectionStatus).toBe("connecting");

      // Simulate session.created → session.updated cycle
      act(() => {
        mockRt._emit("session.created");
      });
      act(() => {
        mockRt._emit("session.updated");
      });

      expect(result.current.connectionStatus).toBe("connected");
    });

    it("falls back to disconnected when createRealtimeClient throws", async () => {
      const { createRealtimeClient } = await import(
        "@/lib/realtime-client"
      );
      (createRealtimeClient as Mock).mockRejectedValueOnce(
        new Error("network error"),
      );

      const { result } = renderHook(() => useRealtimeSession());

      await act(async () => {
        await result.current.connect(defaultSettings);
      });

      expect(result.current.connectionStatus).toBe("disconnected");
    });
  });

  // -----------------------------------------------------------------------
  // 2. Message accumulation from deltas
  // -----------------------------------------------------------------------
  describe("message accumulation from deltas", () => {
    async function connectHook() {
      const hook = renderHook(() => useRealtimeSession());
      await act(async () => {
        await hook.result.current.connect(defaultSettings);
      });
      return hook;
    }

    it("accumulates assistant text deltas into a single message", async () => {
      const { result } = await connectHook();

      act(() => {
        mockRt._emit("response.created", {
          response: { id: "resp-1" },
        });
      });

      // First delta creates the message
      act(() => {
        mockRt._emit("response.output_text.delta", {
          response_id: "resp-1",
          delta: "Hello",
        });
      });

      expect(result.current.messages).toHaveLength(1);
      expect(result.current.messages[0].content).toBe("Hello");
      expect(result.current.messages[0].role).toBe("assistant");
      expect(result.current.messages[0].isStreaming).toBe(true);

      // Second delta appends
      act(() => {
        mockRt._emit("response.output_text.delta", {
          response_id: "resp-1",
          delta: " world",
        });
      });

      expect(result.current.messages).toHaveLength(1);
      expect(result.current.messages[0].content).toBe("Hello world");

      // Done event finalizes
      act(() => {
        mockRt._emit("response.output_text.done", {
          response_id: "resp-1",
          text: "Hello world",
        });
      });

      expect(result.current.messages[0].isStreaming).toBe(false);
      expect(result.current.messages[0].content).toBe("Hello world");
    });

    it("accumulates user transcription deltas", async () => {
      const { result } = await connectHook();

      act(() => {
        mockRt._emit(
          "conversation.item.input_audio_transcription.delta",
          { item_id: "item-1", delta: "Hi " },
        );
      });

      expect(result.current.messages).toHaveLength(1);
      expect(result.current.messages[0].role).toBe("user");
      expect(result.current.messages[0].content).toBe("Hi ");

      act(() => {
        mockRt._emit(
          "conversation.item.input_audio_transcription.delta",
          { item_id: "item-1", delta: "there" },
        );
      });

      expect(result.current.messages[0].content).toBe("Hi there");

      act(() => {
        mockRt._emit(
          "conversation.item.input_audio_transcription.completed",
          { item_id: "item-1", transcript: "Hi there" },
        );
      });

      expect(result.current.messages[0].isStreaming).toBe(false);
    });
  });

  // -----------------------------------------------------------------------
  // 3. Disconnect cleanup
  // -----------------------------------------------------------------------
  describe("disconnect cleanup", () => {
    it("resets state and closes the client on disconnect", async () => {
      const { result } = renderHook(() => useRealtimeSession());

      await act(async () => {
        await result.current.connect(defaultSettings);
      });
      act(() => {
        mockRt._emit("session.created");
        mockRt._emit("session.updated");
      });
      expect(result.current.connectionStatus).toBe("connected");

      act(() => {
        result.current.disconnect();
      });

      expect(result.current.connectionStatus).toBe("disconnected");
      expect(result.current.isSpeechActive).toBe(false);
      expect(result.current.isResponding).toBe(false);
      expect(mockRt.close).toHaveBeenCalled();
    });

    it("cleans up unmute event listener on disconnect", async () => {
      const { result } = renderHook(() => useRealtimeSession());

      await act(async () => {
        await result.current.connect(defaultSettings);
      });

      act(() => {
        result.current.disconnect();
      });

      expect(mockOnUnmuteCleanup).toHaveBeenCalled();
    });

    it("cleans up on unmount", async () => {
      const { result, unmount } = renderHook(() => useRealtimeSession());

      await act(async () => {
        await result.current.connect(defaultSettings);
      });

      unmount();
      expect(mockRt.close).toHaveBeenCalled();
    });
  });

  // -----------------------------------------------------------------------
  // 4. Error events
  // -----------------------------------------------------------------------
  describe("error events", () => {
    it("logs errors to console without crashing", async () => {
      const consoleSpy = vi
        .spyOn(console, "error")
        .mockImplementation(() => {});

      const { result } = renderHook(() => useRealtimeSession());

      await act(async () => {
        await result.current.connect(defaultSettings);
      });

      act(() => {
        mockRt._emit("error", { message: "something went wrong" });
      });

      expect(consoleSpy).toHaveBeenCalledWith(
        "[realtime] error:",
        expect.objectContaining({ message: "something went wrong" }),
      );

      // Hook should still be functional
      expect(result.current.connectionStatus).toBe("connecting");

      consoleSpy.mockRestore();
    });
  });
});
