/**
 * Vector Manager
 *
 * High-level manager that coordinates embedding generation and vector search.
 */

import * as fs from 'fs';
import { SqliteDatabase } from '../db/sqlite-adapter';
import { Node, SearchResult, SearchOptions } from '../types';
import { TextEmbedder, createEmbedder, EmbedderOptions, EmbedderStatusUpdate } from './embedder';
import { VectorSearchManager, createVectorSearch } from './search';
import { SqliteVssLoadablePaths } from './sqlite-vss-probe';
import { QueryBuilder } from '../db/queries';
import { validatePathWithinRoot } from '../utils';
import * as crypto from 'crypto';

/**
 * Progress callback for embedding generation
 */
export interface EmbeddingProgress {
  /** Current node index */
  current: number;

  /** Total nodes to embed */
  total: number;

  /** Current node being embedded */
  nodeName?: string;

  /** Optional status for non-progress updates such as quota waits */
  status?: 'embedding' | 'waiting';

  /** Human-readable detail for the current status */
  detail?: string;
}

/**
 * Options for the vector manager
 */
export interface VectorManagerOptions {
  /** Embedder options */
  embedder?: EmbedderOptions;

  /** Node kinds to embed (default: functions, methods, classes, interfaces) */
  nodeKinds?: Node['kind'][];

  /** Batch size for embedding generation */
  batchSize?: number;

  /** Init-probed sqlite-vss extension paths for ANN search */
  sqliteVssLoadablePaths?: SqliteVssLoadablePaths;

  /**
   * Project root, used to read each node's source body for embedding. When
   * omitted, embeddings fall back to metadata-only — back-compat for callers
   * that don't have a project root handy.
   */
  projectRoot?: string;
}

/**
 * Default node kinds to embed
 */
const DEFAULT_NODE_KINDS: Node['kind'][] = [
  'function',
  'method',
  'class',
  'interface',
  'type_alias',
  'module',
  'component',
];

/**
 * Vector Manager
 *
 * Provides high-level interface for semantic search:
 * - Generates embeddings for code nodes
 * - Stores embeddings in the database
 * - Performs semantic similarity search
 */
export class VectorManager {
  private embedder: TextEmbedder;
  private searchManager: VectorSearchManager;
  private queries: QueryBuilder;
  private nodeKinds: Node['kind'][];
  private batchSize: number;
  private projectRoot: string | undefined;
  private initialized = false;

  constructor(
    db: SqliteDatabase,
    queries: QueryBuilder,
    options: VectorManagerOptions = {}
  ) {
    this.embedder = createEmbedder(options.embedder);
    this.searchManager = createVectorSearch(
      db,
      this.embedder.getDimension(),
      options.sqliteVssLoadablePaths,
      this.embedder.getModelId()
    );
    this.queries = queries;
    this.nodeKinds = options.nodeKinds || DEFAULT_NODE_KINDS;
    // batchSize: user override (clamped to model max) → model default → embedder fallback
    const maxBatch = this.embedder.getMaxBatchSize();
    const requested = options.batchSize ?? this.embedder.getDefaultBatchSize();
    this.batchSize = Math.min(Math.max(1, requested), maxBatch);
    this.projectRoot = options.projectRoot;
  }

  /**
   * Read a node's source body from disk. Returns undefined when projectRoot
   * isn't set, the path escapes the root, the file can't be read, or the node
   * has no line range — caller falls back to metadata-only embedding text.
   *
   * Uses the per-call fileCache so a 50-method class reads its file once
   * instead of 50 times.
   */
  private getNodeBody(node: Node, fileCache: Map<string, string[]>): string | undefined {
    if (!this.projectRoot) return undefined;
    if (!node.startLine || !node.endLine) return undefined;

    let lines = fileCache.get(node.filePath);
    if (lines === undefined) {
      const abs = validatePathWithinRoot(this.projectRoot, node.filePath);
      if (!abs) {
        fileCache.set(node.filePath, []);
        return undefined;
      }
      try {
        lines = fs.readFileSync(abs, 'utf-8').split('\n');
      } catch {
        fileCache.set(node.filePath, []);
        return undefined;
      }
      fileCache.set(node.filePath, lines);
    }
    if (lines.length === 0) return undefined;

    const startIdx = Math.max(0, node.startLine - 1);
    const endIdx = Math.min(lines.length, node.endLine);
    if (startIdx >= endIdx) return undefined;
    return lines.slice(startIdx, endIdx).join('\n');
  }

  /**
   * Initialize the vector manager
   *
   * Validates embedding configuration and initializes vector search.
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    // Initialize embedder configuration.
    await this.embedder.initialize();

    // Initialize vector search (loads sqlite-vss if available)
    await this.searchManager.initialize();

    this.initialized = true;
  }

  /**
   * Check if the vector manager is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Generate embeddings for all eligible nodes
   *
   * @param onProgress - Optional progress callback
   * @returns Number of nodes embedded
   */
  async embedAllNodes(onProgress?: (progress: EmbeddingProgress) => void): Promise<number> {
    if (!this.initialized) {
      throw new Error('VectorManager not initialized. Call initialize() first.');
    }

    // Get all nodes that should be embedded
    const nodesToEmbed: Node[] = [];
    for (const kind of this.nodeKinds) {
      const nodes = this.queries.getNodesByKind(kind);
      nodesToEmbed.push(...nodes);
    }

    // Remove vectors for nodes no longer present, then embed nodes whose text
    // or model changed. The hash-based staleness check below already covers a
    // createNodeText format change (e.g. now including body) — when the body
    // is appended for the first time, every existing node's text hash changes
    // and they all flow back into the embed loop, no migration code needed.
    this.searchManager.deleteStaleVectors();
    const model = this.embedder.getModelId();
    // Per-call file cache, shared across the staleness scan and the embed
    // loop so a 50-method class reads its source file once total.
    const fileCache = new Map<string, string[]>();
    const newNodes = nodesToEmbed.filter((node) => {
      const text = TextEmbedder.createNodeText({ ...node, body: this.getNodeBody(node, fileCache) });
      return !this.searchManager.hasCurrentVector(
        node.id,
        model,
        this.hashText(text)
      );
    });

    if (newNodes.length === 0) {
      return 0;
    }

    // Process in batches
    let processed = 0;
    this.embedder.setStatusReporter((status) => {
      this.reportEmbedderStatus(status, processed, newNodes.length, onProgress);
    });
    try {
      for (let i = 0; i < newNodes.length; i += this.batchSize) {
        const batch = newNodes.slice(i, i + this.batchSize);

        // Create text representations
        const texts = batch.map((node) =>
          TextEmbedder.createNodeText({ ...node, body: this.getNodeBody(node, fileCache) })
        );

        // Generate embeddings
        const result = await this.embedder.embedBatch(texts, 'document');

        // Store embeddings
        const entries: Array<{ nodeId: string; embedding: Float32Array; contentHash: string }> = [];
        for (let idx = 0; idx < batch.length; idx++) {
          const node = batch[idx];
          const embedding = result.embeddings[idx];
          if (node && embedding) {
            entries.push({ nodeId: node.id, embedding, contentHash: this.hashText(texts[idx]!) });
          }
        }

        this.searchManager.storeVectorBatch(entries, model);

        processed += batch.length;

        // Report progress
        if (onProgress) {
          onProgress({
            current: processed,
            total: newNodes.length,
            nodeName: batch[batch.length - 1]?.name,
            status: 'embedding',
          });
        }
      }
    } finally {
      this.embedder.setStatusReporter(undefined);
    }

    return processed;
  }

  /**
   * Generate embedding for a single node
   *
   * @param node - Node to embed
   */
  async embedNode(node: Node): Promise<void> {
    if (!this.initialized) {
      throw new Error('VectorManager not initialized. Call initialize() first.');
    }

    const body = this.getNodeBody(node, new Map());
    const text = TextEmbedder.createNodeText({ ...node, body });
    const result = await this.embedder.embed(text);
    this.searchManager.storeVector(node.id, result.embedding, result.model, this.hashText(text));
  }

  /**
   * Semantic search for nodes matching a query
   *
   * @param query - Natural language query
   * @param options - Search options
   * @returns Array of search results with similarity scores
   */
  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    if (!this.initialized) {
      throw new Error('VectorManager not initialized. Call initialize() first.');
    }

    const { limit = 10, kinds } = options;

    // Generate query embedding
    const queryResult = await this.embedder.embedQuery(query);

    // Search for similar vectors
    const vectorResults = this.searchManager.search(queryResult.embedding, {
      limit: limit * 2, // Get more results to filter
      minScore: 0.3, // Minimum similarity threshold
    });

    // Get nodes and filter by kind if specified
    const results: SearchResult[] = [];
    for (const vr of vectorResults) {
      const node = this.queries.getNodeById(vr.nodeId);
      if (!node) {
        continue;
      }

      // Filter by node kind if specified
      if (kinds && kinds.length > 0 && !kinds.includes(node.kind)) {
        continue;
      }

      results.push({
        node,
        score: vr.score,
      });

      if (results.length >= limit) {
        break;
      }
    }

    return results;
  }

  /**
   * Find nodes similar to a given node
   *
   * @param nodeId - ID of the node to find similar nodes for
   * @param options - Search options
   * @returns Array of similar nodes with similarity scores
   */
  async findSimilar(nodeId: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    if (!this.initialized) {
      throw new Error('VectorManager not initialized. Call initialize() first.');
    }

    const { limit = 10, kinds } = options;

    // Get the node's embedding
    let embedding = this.searchManager.getVector(nodeId);

    // If no embedding exists, generate one
    if (!embedding) {
      const node = this.queries.getNodeById(nodeId);
      if (!node) {
        throw new Error(`Node not found: ${nodeId}`);
      }

      await this.embedNode(node);
      embedding = this.searchManager.getVector(nodeId);

      if (!embedding) {
        throw new Error(`Failed to generate embedding for node: ${nodeId}`);
      }
    }

    // Search for similar vectors (excluding the source node)
    const vectorResults = this.searchManager.search(embedding, {
      limit: limit + 1, // Get one extra to exclude the source
      minScore: 0.3,
    });

    // Get nodes and filter
    const results: SearchResult[] = [];
    for (const vr of vectorResults) {
      // Skip the source node
      if (vr.nodeId === nodeId) {
        continue;
      }

      const node = this.queries.getNodeById(vr.nodeId);
      if (!node) {
        continue;
      }

      // Filter by node kind if specified
      if (kinds && kinds.length > 0 && !kinds.includes(node.kind)) {
        continue;
      }

      results.push({
        node,
        score: vr.score,
      });

      if (results.length >= limit) {
        break;
      }
    }

    return results;
  }

  /**
   * Delete embedding for a node
   *
   * @param nodeId - ID of the node
   */
  deleteNodeEmbedding(nodeId: string): void {
    this.searchManager.deleteVector(nodeId);
  }

  /**
   * Get statistics about vector storage
   */
  getStats(): {
    totalVectors: number;
    vssEnabled: boolean;
    modelId: string;
  } {
    return {
      totalVectors: this.searchManager.getVectorCount(),
      vssEnabled: this.searchManager.isVssEnabled(),
      modelId: this.embedder.getModelId(),
    };
  }

  /**
   * Clear all vectors
   */
  clear(): void {
    this.searchManager.clear();
  }

  /**
   * Delete vectors for nodes that no longer exist.
   */
  deleteStaleVectors(): number {
    return this.searchManager.deleteStaleVectors();
  }

  /**
   * Rebuild the VSS index
   */
  rebuildIndex(): void {
    this.searchManager.rebuildVssIndex();
  }

  /**
   * Release resources
   */
  dispose(): void {
    this.embedder.dispose();
  }

  private hashText(text: string): string {
    return crypto.createHash('sha256').update(text).digest('hex');
  }

  private reportEmbedderStatus(
    status: EmbedderStatusUpdate,
    current: number,
    total: number,
    onProgress?: (progress: EmbeddingProgress) => void
  ): void {
    if (!onProgress || status.phase !== 'retry_wait') {
      return;
    }

    const retryInSeconds = Math.max(1, Math.ceil(status.retryInMs / 1000));
    onProgress({
      current,
      total,
      status: 'waiting',
      detail: `${retryInSeconds}s (try ${status.attempt})`,
    });
  }
}

/**
 * Create a vector manager
 */
export function createVectorManager(
  db: SqliteDatabase,
  queries: QueryBuilder,
  options?: VectorManagerOptions
): VectorManager {
  return new VectorManager(db, queries, options);
}
