import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ChatMessage } from "@/components/chat/ChatMessage";
import type { ChatMessageData } from "@/components/chat/ChatMessage";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMessage(
  overrides: Partial<ChatMessageData> = {},
): ChatMessageData {
  return {
    id: "msg-1",
    role: "user",
    content: "Hello world",
    timestamp: new Date("2026-03-11T14:30:45Z"),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ChatMessage", () => {
  it("renders a user message with content", () => {
    render(<ChatMessage message={makeMessage()} />);
    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("renders an assistant message with content", () => {
    render(
      <ChatMessage
        message={makeMessage({ role: "assistant", content: "I can help" })}
      />,
    );
    expect(screen.getByText("I can help")).toBeInTheDocument();
  });

  it("right-aligns user messages", () => {
    const { container } = render(<ChatMessage message={makeMessage()} />);
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.className).toContain("justify-end");
  });

  it("left-aligns assistant messages", () => {
    const { container } = render(
      <ChatMessage message={makeMessage({ role: "assistant" })} />,
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.className).toContain("justify-start");
  });

  it("shows the streaming cursor when isStreaming is true", () => {
    const { container } = render(
      <ChatMessage message={makeMessage({ isStreaming: true })} />,
    );
    const cursor = container.querySelector(".animate-pulse");
    expect(cursor).toBeInTheDocument();
  });

  it("does not show the streaming cursor when isStreaming is false", () => {
    const { container } = render(
      <ChatMessage message={makeMessage({ isStreaming: false })} />,
    );
    const cursor = container.querySelector(".animate-pulse");
    expect(cursor).not.toBeInTheDocument();
  });

  it("displays the timestamp in HH:MM:SS format", () => {
    render(<ChatMessage message={makeMessage()} />);
    // The timestamp depends on the local timezone of the test runner,
    // so we just verify a time-like pattern is rendered.
    const timeEl = screen.getByText(/\d{2}:\d{2}:\d{2}/);
    expect(timeEl).toBeInTheDocument();
  });

  it("shows a thinking time badge when thinkingTimeMs is provided", () => {
    render(
      <ChatMessage
        message={makeMessage({
          role: "assistant",
          thinkingTimeMs: 500,
        })}
      />,
    );
    expect(screen.getByText("think 500ms")).toBeInTheDocument();
  });

  it("shows a response duration badge when responseDurationMs is provided", () => {
    render(
      <ChatMessage
        message={makeMessage({
          role: "assistant",
          responseDurationMs: 2500,
        })}
      />,
    );
    expect(screen.getByText("2.5s")).toBeInTheDocument();
  });

  it("does not show badges when timing fields are absent", () => {
    render(<ChatMessage message={makeMessage()} />);
    expect(screen.queryByText(/think/)).not.toBeInTheDocument();
  });
});
