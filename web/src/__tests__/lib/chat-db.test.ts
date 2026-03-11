import { describe, it, expect, beforeEach, vi } from "vitest";
import "fake-indexeddb/auto";
import type {
  ChatSessionRecord,
  ChatMessageRecord,
} from "@/lib/db/chat-db";

// chat-db caches its DB connection in a module-scoped `dbPromise`.
// To get a fresh DB per test we must reset both the global `indexedDB`
// and the Vitest module cache so the next dynamic `import()` creates a
// new IDB connection.

async function freshModule() {
  vi.resetModules();
  return import("@/lib/db/chat-db");
}

function makeSession(
  overrides: Partial<ChatSessionRecord> = {},
): ChatSessionRecord {
  return {
    id: overrides.id ?? crypto.randomUUID(),
    title: overrides.title ?? "Test session",
    createdAt: overrides.createdAt ?? new Date("2026-01-01T00:00:00Z"),
    updatedAt: overrides.updatedAt ?? new Date("2026-01-01T00:00:00Z"),
    messages: overrides.messages ?? [],
    config: overrides.config ?? {
      voice: "alloy",
      customInstructions: "",
      backendUrl: "ws://localhost:8000",
    },
  };
}

function makeMessage(
  overrides: Partial<ChatMessageRecord> = {},
): ChatMessageRecord {
  return {
    id: overrides.id ?? crypto.randomUUID(),
    role: overrides.role ?? "user",
    content: overrides.content ?? "hello",
    timestamp: overrides.timestamp ?? new Date(),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("chat-db CRUD operations", () => {
  beforeEach(() => {
    indexedDB = new IDBFactory();
  });

  it("creates and retrieves a session", async () => {
    const db = await freshModule();
    const session = makeSession({ id: "s1", title: "My chat" });
    const id = await db.createSession(session);
    expect(id).toBe("s1");

    const fetched = await db.getSession("s1");
    expect(fetched).toBeDefined();
    expect(fetched!.title).toBe("My chat");
    expect(fetched!.messages).toEqual([]);
  });

  it("returns undefined for a nonexistent session", async () => {
    const db = await freshModule();
    const result = await db.getSession("nonexistent");
    expect(result).toBeUndefined();
  });

  it("updates a session", async () => {
    const db = await freshModule();
    const session = makeSession({ id: "s2" });
    await db.createSession(session);

    session.title = "Updated title";
    session.updatedAt = new Date("2026-06-01T00:00:00Z");
    await db.updateSession(session);

    const fetched = await db.getSession("s2");
    expect(fetched!.title).toBe("Updated title");
  });

  it("deletes a session", async () => {
    const db = await freshModule();
    const session = makeSession({ id: "s3" });
    await db.createSession(session);
    await db.deleteSession("s3");

    const fetched = await db.getSession("s3");
    expect(fetched).toBeUndefined();
  });

  it("adds a message to an existing session", async () => {
    const db = await freshModule();
    const session = makeSession({ id: "s4" });
    await db.createSession(session);

    const msg = makeMessage({ content: "world" });
    await db.addMessage("s4", msg);

    const fetched = await db.getSession("s4");
    expect(fetched!.messages).toHaveLength(1);
    expect(fetched!.messages[0].content).toBe("world");
  });

  it("throws when adding a message to a nonexistent session", async () => {
    const db = await freshModule();
    const msg = makeMessage();
    await expect(db.addMessage("nope", msg)).rejects.toThrow(
      "Session not found: nope",
    );
  });
});

// ---------------------------------------------------------------------------
// Session ordering
// ---------------------------------------------------------------------------

describe("session ordering (getAllSessions)", () => {
  beforeEach(() => {
    indexedDB = new IDBFactory();
  });

  it("returns sessions sorted by updatedAt", async () => {
    const db = await freshModule();

    const older = makeSession({
      id: "old",
      title: "Old",
      updatedAt: new Date("2026-01-01T00:00:00Z"),
    });
    const newer = makeSession({
      id: "new",
      title: "New",
      updatedAt: new Date("2026-06-01T00:00:00Z"),
    });

    // Insert in reverse order
    await db.createSession(newer);
    await db.createSession(older);

    const all = await db.getAllSessions();
    expect(all).toHaveLength(2);
    // Index "by-updated" sorts ascending
    expect(all[0].id).toBe("old");
    expect(all[1].id).toBe("new");
  });
});

// ---------------------------------------------------------------------------
// Message append
// ---------------------------------------------------------------------------

describe("message append", () => {
  beforeEach(() => {
    indexedDB = new IDBFactory();
  });

  it("appends multiple messages and updates updatedAt", async () => {
    const db = await freshModule();
    const session = makeSession({
      id: "s5",
      updatedAt: new Date("2026-01-01T00:00:00Z"),
    });
    await db.createSession(session);

    await db.addMessage("s5", makeMessage({ content: "msg1" }));
    await db.addMessage("s5", makeMessage({ content: "msg2" }));

    const fetched = await db.getSession("s5");
    expect(fetched!.messages).toHaveLength(2);
    expect(fetched!.messages[0].content).toBe("msg1");
    expect(fetched!.messages[1].content).toBe("msg2");
    // updatedAt should have been bumped
    expect(fetched!.updatedAt.getTime()).toBeGreaterThan(
      new Date("2026-01-01T00:00:00Z").getTime(),
    );
  });
});
