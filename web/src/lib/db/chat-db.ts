import { openDB, type DBSchema, type IDBPDatabase } from "idb";

export interface SessionConfigRecord {
  voice: string;
  customInstructions: string;
  backendUrl: string;
}

export interface ChatMessageRecord {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  responseId?: string;
  thinkingTimeMs?: number;
  responseDurationMs?: number;
}

export interface ChatSessionRecord {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  messages: ChatMessageRecord[];
  config: SessionConfigRecord;
}

interface UnmuteDB extends DBSchema {
  sessions: {
    key: string;
    value: ChatSessionRecord;
    indexes: {
      "by-updated": Date;
    };
  };
}

const DB_NAME = "unmute-db";
const DB_VERSION = 1;

let dbPromise: Promise<IDBPDatabase<UnmuteDB>> | null = null;

function getDB(): Promise<IDBPDatabase<UnmuteDB>> {
  if (!dbPromise) {
    dbPromise = openDB<UnmuteDB>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        const store = db.createObjectStore("sessions", { keyPath: "id" });
        store.createIndex("by-updated", "updatedAt");
      },
    });
  }
  return dbPromise;
}

export async function createSession(
  session: ChatSessionRecord
): Promise<string> {
  const db = await getDB();
  await db.add("sessions", session);
  return session.id;
}

export async function getSession(
  id: string
): Promise<ChatSessionRecord | undefined> {
  const db = await getDB();
  return db.get("sessions", id);
}

export async function getAllSessions(): Promise<ChatSessionRecord[]> {
  const db = await getDB();
  return db.getAllFromIndex("sessions", "by-updated");
}

export async function updateSession(
  session: ChatSessionRecord
): Promise<void> {
  const db = await getDB();
  await db.put("sessions", session);
}

export async function deleteSession(id: string): Promise<void> {
  const db = await getDB();
  await db.delete("sessions", id);
}

export async function addMessage(
  sessionId: string,
  message: ChatMessageRecord
): Promise<void> {
  const db = await getDB();
  const session = await db.get("sessions", sessionId);
  if (!session) {
    throw new Error(`Session not found: ${sessionId}`);
  }
  session.messages.push(message);
  session.updatedAt = new Date();
  await db.put("sessions", session);
}
