/**
 * Vector Search
 *
 * Vector similarity search backed by the sqlite-vec (`vec0`) loadable
 * extension. sqlite-vec is the sole backend — there is no brute-force fallback
 * and no separate embedding BLOB store. When sqlite-vec cannot be loaded (a
 * platform without a prebuilt, or a `--no-optional` install), semantic search
 * is simply unavailable: queries return nothing and stores are no-ops. The
 * graph and keyword search are unaffected.
 *
 * Embeddings live only in the `vec0` virtual table. A sibling `vec_map` plain
 * table carries the node_id ↔ rowid mapping plus the model/content_hash
 * staleness metadata that drives incremental re-embedding.
 */

import { SqliteDatabase } from '../db/sqlite-adapter';

/**
 * Convert a Float32 embedding to the compact little-endian blob that vec0
 * accepts for `float[N]` columns.
 */
function embeddingToBlob(embedding: Float32Array): Buffer {
  return Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength);
}

/**
 * Decode a vec0 `embedding` column (returned as raw float32 bytes) back into a
 * Float32Array. Copies so the result doesn't alias SQLite's buffer.
 */
function blobToFloat32(buf: Buffer | Uint8Array): Float32Array {
  const view = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
  return view.slice();
}

/**
 * Options for vector search
 */
export interface VectorSearchOptions {
  /** Maximum number of results to return */
  limit?: number;

  /** Minimum similarity score (0-1) */
  minScore?: number;

  /** Node kinds to filter results */
  nodeKinds?: import('../types').Node['kind'][];
}

/**
 * Vector Search Manager
 *
 * Handles vector storage and similarity search for semantic code search.
 */
export class VectorSearchManager {
  private db: SqliteDatabase;
  private vecEnabled = false;
  private embeddingDimension: number;
  private currentModel?: string;

  constructor(db: SqliteDatabase, dimension: number = 768, currentModel?: string) {
    this.db = db;
    this.embeddingDimension = dimension;
    this.currentModel = currentModel;
  }

  /**
   * Initialize vector search.
   *
   * Loads the sqlite-vec extension and creates the vec0 + vec_map tables. If
   * sqlite-vec is unavailable on this platform/install, leaves the manager
   * disabled (`vecEnabled = false`) — every operation then no-ops and search
   * returns no results.
   */
  async initialize(): Promise<void> {
    if (!this.loadSqliteVec()) {
      this.vecEnabled = false;
      return;
    }

    // A configured model different from what's stored means a different
    // embedding space (and possibly dimension), so wipe before (re)creating the
    // vec0 table at the current dimension.
    this.clearIfModelChanged(this.currentModel);
    this.createVecTables();
    this.vecEnabled = true;
  }

  /**
   * Load the sqlite-vec loadable extension into this connection. Returns false
   * (rather than throwing) when the extension isn't available — a missing
   * platform prebuilt, a `--no-optional` install, or the package not resolving.
   */
  private loadSqliteVec(): boolean {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const sqliteVec = require('sqlite-vec') as { load: (db: unknown) => void };
      // load(db) calls db.loadExtension(getLoadablePath()); the adapter exposes
      // loadExtension and was opened with allowExtension: true.
      sqliteVec.load(this.db);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Create the vec0 virtual table (dimension baked in at CREATE time) and the
   * vec_map metadata table. Idempotent.
   */
  private createVecTables(): void {
    const exists = this.db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_items'")
      .get();

    if (!exists) {
      this.db.exec(`
        CREATE VIRTUAL TABLE vec_items USING vec0(
          embedding float[${this.embeddingDimension}] distance_metric=cosine
        );
      `);
    }

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS vec_map (
        rowid INTEGER PRIMARY KEY,
        node_id TEXT NOT NULL UNIQUE,
        model TEXT NOT NULL,
        content_hash TEXT NOT NULL DEFAULT '',
        created_at INTEGER NOT NULL
      );
    `);
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_vec_map_node ON vec_map(node_id);
    `);
  }

  /**
   * If vec_map holds rows embedded with a model other than the configured one,
   * drop the vec store so `embedAllNodes()` re-embeds into a fresh space. Safe
   * before the tables exist (first run) — it checks for vec_map first.
   */
  private clearIfModelChanged(currentModel?: string): void {
    if (!currentModel) return;

    const hasMap = this.db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_map'")
      .get();
    if (!hasMap) return;

    const stale = this.db
      .prepare('SELECT 1 FROM vec_map WHERE model != ? LIMIT 1')
      .get(currentModel);
    if (!stale) return;

    // eslint-disable-next-line no-console
    console.warn(
      `[codegraph] stored vectors were embedded with a different model than the ` +
        `currently configured "${currentModel}". Clearing the vector store; the next ` +
        `index sync will re-embed all symbols.`
    );
    this.db.exec('DROP TABLE IF EXISTS vec_items;');
    this.db.exec('DROP TABLE IF EXISTS vec_map;');
  }

  /**
   * Whether the sqlite-vec backend is loaded and usable.
   */
  isVecEnabled(): boolean {
    return this.vecEnabled;
  }

  /**
   * Look up the vec_items rowid for a node, or null if it has no vector.
   */
  private rowidForNode(nodeId: string): bigint | null {
    const row = this.db
      .prepare('SELECT rowid FROM vec_map WHERE node_id = ?')
      .get(nodeId) as { rowid: number | bigint } | undefined;
    return row ? BigInt(row.rowid) : null;
  }

  /**
   * Store a vector embedding for a node.
   */
  storeVector(nodeId: string, embedding: Float32Array, model: string, contentHash: string = ''): void {
    if (!this.vecEnabled) return;
    this.storeOne(nodeId, embedding, model, contentHash);
  }

  /**
   * Store multiple vectors in a single transaction.
   */
  storeVectorBatch(
    entries: Array<{ nodeId: string; embedding: Float32Array; contentHash?: string }>,
    model: string
  ): void {
    if (!this.vecEnabled) return;
    this.db.transaction(() => {
      for (const entry of entries) {
        this.storeOne(entry.nodeId, entry.embedding, model, entry.contentHash ?? '');
      }
    })();
  }

  /**
   * Insert or replace a single node's vector in vec0 + vec_map.
   */
  private storeOne(nodeId: string, embedding: Float32Array, model: string, contentHash: string): void {
    const now = Date.now();
    const blob = embeddingToBlob(embedding);
    const existing = this.rowidForNode(nodeId);

    if (existing !== null) {
      // Replace the vector at the existing rowid (delete + insert is the
      // portable vec0 update path) and refresh metadata.
      this.db.prepare('DELETE FROM vec_items WHERE rowid = ?').run(existing);
      this.db.prepare('INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)').run(existing, blob);
      this.db
        .prepare('UPDATE vec_map SET model = ?, content_hash = ?, created_at = ? WHERE rowid = ?')
        .run(model, contentHash, now, existing);
      return;
    }

    const maxRow = this.db
      .prepare('SELECT MAX(rowid) as max FROM vec_map')
      .get() as { max: number | bigint | null } | undefined;
    const newRowid = BigInt((maxRow?.max ?? 0)) + 1n;

    this.db.prepare('INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)').run(newRowid, blob);
    this.db
      .prepare(
        'INSERT INTO vec_map (rowid, node_id, model, content_hash, created_at) VALUES (?, ?, ?, ?, ?)'
      )
      .run(newRowid, nodeId, model, contentHash, now);
  }

  /**
   * Get the stored vector for a node, or null if it has none.
   */
  getVector(nodeId: string): Float32Array | null {
    if (!this.vecEnabled) return null;
    const rowid = this.rowidForNode(nodeId);
    if (rowid === null) return null;

    const row = this.db
      .prepare('SELECT embedding FROM vec_items WHERE rowid = ?')
      .get(rowid) as { embedding: Buffer | Uint8Array } | undefined;
    if (!row?.embedding) return null;
    return blobToFloat32(row.embedding);
  }

  /**
   * Delete the vector for a node.
   */
  deleteVector(nodeId: string): void {
    if (!this.vecEnabled) return;
    const rowid = this.rowidForNode(nodeId);
    if (rowid === null) return;
    this.db.prepare('DELETE FROM vec_items WHERE rowid = ?').run(rowid);
    this.db.prepare('DELETE FROM vec_map WHERE node_id = ?').run(nodeId);
  }

  /**
   * Search for similar vectors via vec0 KNN.
   */
  search(
    queryEmbedding: Float32Array,
    options: VectorSearchOptions = {}
  ): Array<{ nodeId: string; score: number }> {
    if (!this.vecEnabled) return [];
    const { limit = 10, minScore = 0 } = options;
    const safeLimit = Math.max(1, Math.floor(limit));

    const rows = this.db
      .prepare(
        `
        SELECT m.node_id AS node_id, v.distance AS distance
        FROM (
          SELECT rowid, distance
          FROM vec_items
          WHERE embedding MATCH ? AND k = ?
          ORDER BY distance
        ) v
        JOIN vec_map m ON m.rowid = v.rowid
      `
      )
      .all(embeddingToBlob(queryEmbedding), BigInt(safeLimit)) as Array<{
      node_id: string;
      distance: number;
    }>;

    // vec0 cosine distance is 1 - cosine_similarity, so similarity = 1 - distance.
    return rows
      .map((row) => ({ nodeId: row.node_id, score: 1 - row.distance }))
      .filter((r) => r.score >= minScore);
  }

  /**
   * Count of stored vectors.
   */
  getVectorCount(): number {
    if (!this.vecEnabled) return 0;
    const result = this.db.prepare('SELECT COUNT(*) as count FROM vec_map').get() as {
      count: number;
    };
    return result.count;
  }

  /**
   * Whether a node has any stored vector.
   */
  hasVector(nodeId: string): boolean {
    if (!this.vecEnabled) return false;
    return this.rowidForNode(nodeId) !== null;
  }

  /**
   * Whether a node has a vector matching the current model + embedded text.
   */
  hasCurrentVector(nodeId: string, model: string, contentHash: string): boolean {
    if (!this.vecEnabled) return false;
    const result = this.db
      .prepare(
        `SELECT 1 FROM vec_map WHERE node_id = ? AND model = ? AND content_hash = ? LIMIT 1`
      )
      .get(nodeId, model, contentHash);
    return !!result;
  }

  /**
   * All node IDs that have vectors.
   */
  getIndexedNodeIds(): string[] {
    if (!this.vecEnabled) return [];
    const rows = this.db.prepare('SELECT node_id FROM vec_map').all() as Array<{ node_id: string }>;
    return rows.map((r) => r.node_id);
  }

  /**
   * Delete vectors whose node no longer exists in the graph.
   */
  deleteStaleVectors(): number {
    if (!this.vecEnabled) return 0;
    const staleIds = this.db
      .prepare(
        `
        SELECT vm.node_id
        FROM vec_map vm
        LEFT JOIN nodes n ON n.id = vm.node_id
        WHERE n.id IS NULL
      `
      )
      .all() as Array<{ node_id: string }>;

    for (const row of staleIds) {
      this.deleteVector(row.node_id);
    }
    return staleIds.length;
  }

  /**
   * Clear all vectors.
   */
  clear(): void {
    if (!this.vecEnabled) return;
    this.db.prepare('DELETE FROM vec_items').run();
    this.db.prepare('DELETE FROM vec_map').run();
  }

  /**
   * Rebuild the vec0 index in place from its own current contents.
   *
   * Reads every stored vector out of vec0, drops + recreates the virtual table
   * at the current dimension/metric, and re-inserts — preserving rowids and the
   * `vec_map` mapping. This is a self-contained reindex: it does NOT re-embed
   * (embeddings are read back from vec0), so it costs no embedding-API calls.
   * Use it to recover a corrupted index or to re-materialize after changing
   * vec0 parameters. No-op when sqlite-vec is unavailable.
   */
  rebuildVecIndex(): void {
    if (!this.vecEnabled) return;

    const rows = this.db
      .prepare('SELECT rowid, embedding FROM vec_items')
      .all() as Array<{ rowid: number | bigint; embedding: Buffer | Uint8Array }>;

    this.db.transaction(() => {
      this.db.exec('DROP TABLE IF EXISTS vec_items;');
      this.db.exec(`
        CREATE VIRTUAL TABLE vec_items USING vec0(
          embedding float[${this.embeddingDimension}] distance_metric=cosine
        );
      `);
      const insert = this.db.prepare('INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)');
      for (const row of rows) {
        const buf = Buffer.from(
          row.embedding.buffer,
          row.embedding.byteOffset,
          row.embedding.byteLength
        );
        insert.run(BigInt(row.rowid), buf);
      }
    })();
  }
}

/**
 * Create a vector search manager
 */
export function createVectorSearch(
  db: SqliteDatabase,
  dimension?: number,
  currentModel?: string
): VectorSearchManager {
  return new VectorSearchManager(db, dimension, currentModel);
}
