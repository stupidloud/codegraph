/**
 * Text Embedder
 *
 * Generates vector embeddings through the Gemini API. Vector storage and
 * similarity search remain local.
 */

/**
 * Default Gemini embedding model.
 */
export const DEFAULT_MODEL = 'gemini-embedding-2';
export const EMBEDDING_DIMENSION = 768;
const GEMINI_EMBEDDING_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models';
const MAX_REQUESTS_PER_MINUTE = 100;
const RATE_LIMIT_WINDOW_MS = 60_000;
const MAX_RETRY_ATTEMPTS = 3;
const INITIAL_RETRY_DELAY_MS = 1_000;
const RETRY_DELAY_MULTIPLIER = 2;
const MAX_RETRY_DELAY_MS = 64_000;
const RETRYABLE_STATUS_CODES = new Set([408, 429, 500, 502, 503, 504]);

/**
 * Options for the embedder
 */
export interface EmbedderOptions {
  /** Gemini API key */
  apiKey?: string;

  /** Model ID to use (default: gemini-embedding-2) */
  modelId?: string;

  /** Embedding dimension to request from Gemini */
  outputDimensionality?: number;

  /** Whether to show progress during embedding generation */
  showProgress?: boolean;
}

/**
 * Text embedding result
 */
export interface EmbeddingResult {
  /** The embedding vector */
  embedding: Float32Array;

  /** Dimension of the embedding */
  dimension: number;

  /** Model used to generate the embedding */
  model: string;
}

/**
 * Batch embedding result
 */
export interface BatchEmbeddingResult {
  /** Array of embeddings in same order as input */
  embeddings: Float32Array[];

  /** Dimension of each embedding */
  dimension: number;

  /** Model used to generate embeddings */
  model: string;

  /** Processing time in milliseconds */
  durationMs: number;
}

type GeminiEmbeddingResponse = {
  embeddings?: Array<{ values?: number[] }>;
};

/**
 * Text Embedder using Gemini.
 *
 * Batch generation uses batchEmbedContents with one request per node text. Do
 * not combine multiple nodes into one content.parts array because that produces
 * a single aggregate embedding instead of node-level vectors.
 */
export class TextEmbedder {
  private apiKey?: string;
  private modelId: string;
  private dimension: number;
  private initialized = false;
  private requestTimestamps: Array<{ timestamp: number; cost: number }> = [];
  private rateLimitQueue: Promise<void> = Promise.resolve();

  constructor(options: EmbedderOptions = {}) {
    this.apiKey = options.apiKey;
    this.modelId = options.modelId || DEFAULT_MODEL;
    this.dimension = options.outputDimensionality || EMBEDDING_DIMENSION;
  }

  /**
   * Initialize the embedder by validating configuration.
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    if (!this.apiKey) {
      throw new Error('Gemini API key is required for semantic search');
    }

    this.initialized = true;
  }

  /**
   * Check if the embedder is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get the model ID being used
   */
  getModelId(): string {
    return this.modelId;
  }

  /**
   * Get the embedding dimension
   */
  getDimension(): number {
    return this.dimension;
  }

  /**
   * Generate embedding for a single text
   */
  async embed(text: string): Promise<EmbeddingResult> {
    const result = await this.embedBatch([text], 'document');
    const embedding = result.embeddings[0];
    if (!embedding) {
      throw new Error('Gemini embedding response did not include values');
    }
    return {
      embedding,
      dimension: embedding.length,
      model: result.model,
    };
  }

  /**
   * Generate embedding for a query.
   */
  async embedQuery(query: string): Promise<EmbeddingResult> {
    const result = await this.embedBatch([query], 'search_query');
    const embedding = result.embeddings[0];
    if (!embedding) {
      throw new Error('Gemini embedding response did not include values');
    }
    return {
      embedding,
      dimension: embedding.length,
      model: result.model,
    };
  }

  /**
   * Generate embeddings for multiple texts in node order.
   */
  async embedBatch(
    texts: string[],
    type: 'document' | 'search_query' = 'document'
  ): Promise<BatchEmbeddingResult> {
    if (!this.initialized) {
      throw new Error('Embedder not initialized. Call initialize() first.');
    }

    if (texts.length === 0) {
      return {
        embeddings: [],
        dimension: this.dimension,
        model: this.modelId,
        durationMs: 0,
      };
    }

    const startTime = Date.now();
    const embeddings = await this.embedPreparedBatch(
      texts.map((text) => this.prepareText(text)),
      type
    );

    return {
      embeddings,
      dimension: embeddings[0]?.length ?? this.dimension,
      model: this.modelId,
      durationMs: Date.now() - startTime,
    };
  }

  private async embedPreparedBatch(
    texts: string[],
    type: 'document' | 'search_query'
  ): Promise<Float32Array[]> {
    if (!this.initialized) {
      throw new Error('Embedder not initialized. Call initialize() first.');
    }

    const url = `${GEMINI_EMBEDDING_ENDPOINT}/${encodeURIComponent(this.modelId)}:batchEmbedContents`;
    const model = `models/${this.modelId}`;
    const taskType = type === 'search_query' ? 'CODE_RETRIEVAL_QUERY' : 'RETRIEVAL_DOCUMENT';
    await this.throttleRequests(texts.length);
    const response = await this.fetchWithRetry(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': this.apiKey!,
      },
      body: JSON.stringify({
        requests: texts.map((text) => ({
          model,
          content: {
            parts: [{ text }],
          },
          taskType,
          outputDimensionality: this.dimension,
        })),
      }),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new Error(`Gemini embedding request failed (${response.status}): ${body || response.statusText}`);
    }

    const json = await response.json() as GeminiEmbeddingResponse;
    const values = json.embeddings?.map((embedding) => embedding.values);
    if (!values || values.length !== texts.length) {
      throw new Error('Gemini batch embedding response count did not match request count');
    }

    const embeddings = values.map((embeddingValues) => {
      if (!embeddingValues || embeddingValues.length === 0) {
        throw new Error('Gemini embedding response did not include values');
      }
      return new Float32Array(embeddingValues);
    });

    if (embeddings.some((embedding) => embedding.length !== embeddings[0]!.length)) {
      throw new Error('Gemini batch embedding response contained mixed dimensions');
    }

    return embeddings;
  }

  private async fetchWithRetry(url: string, init: RequestInit): Promise<Response> {
    let lastNetworkError: unknown;

    for (let attempt = 0; attempt <= MAX_RETRY_ATTEMPTS; attempt++) {
      try {
        const response = await fetch(url, init);
        if (!this.isRetryableStatus(response.status) || attempt === MAX_RETRY_ATTEMPTS) {
          return response;
        }

        await this.sleep(this.getRetryDelayMs(attempt, response));
      } catch (error) {
        lastNetworkError = error;
        if (attempt === MAX_RETRY_ATTEMPTS) {
          break;
        }

        await this.sleep(this.getRetryDelayMs(attempt));
      }
    }

    throw lastNetworkError instanceof Error
      ? lastNetworkError
      : new Error(`Gemini embedding request failed: ${String(lastNetworkError)}`);
  }

  private isRetryableStatus(status: number): boolean {
    return RETRYABLE_STATUS_CODES.has(status);
  }

  private getRetryDelayMs(attempt: number, response?: Response): number {
    const retryAfterMs = this.getRetryAfterMs(response);
    const exponentialDelay = Math.min(
      INITIAL_RETRY_DELAY_MS * Math.pow(RETRY_DELAY_MULTIPLIER, attempt),
      MAX_RETRY_DELAY_MS
    );
    return Math.max(retryAfterMs ?? 0, exponentialDelay);
  }

  private getRetryAfterMs(response?: Response): number | undefined {
    const retryAfter = response?.headers?.get?.('retry-after');
    if (!retryAfter) {
      return undefined;
    }

    const seconds = Number(retryAfter);
    if (Number.isFinite(seconds) && seconds >= 0) {
      return seconds * 1000;
    }

    const retryAt = Date.parse(retryAfter);
    if (Number.isNaN(retryAt)) {
      return undefined;
    }

    return Math.max(0, retryAt - Date.now());
  }

  private async sleep(ms: number): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, ms));
  }

  private async throttleRequests(cost: number): Promise<void> {
    const reservation = this.rateLimitQueue.then(() => this.reserveRequestSlots(cost));
    this.rateLimitQueue = reservation.catch(() => undefined);
    await reservation;
  }

  private async reserveRequestSlots(cost: number): Promise<void> {
    if (cost > MAX_REQUESTS_PER_MINUTE) {
      throw new Error(
        `Gemini embedding batch size ${cost} exceeds the ${MAX_REQUESTS_PER_MINUTE} requests-per-minute limit`
      );
    }

    while (true) {
      const now = Date.now();
      this.requestTimestamps = this.requestTimestamps.filter(
        ({ timestamp }) => now - timestamp < RATE_LIMIT_WINDOW_MS
      );

      const usedSlots = this.requestTimestamps.reduce((sum, entry) => sum + entry.cost, 0);
      if (usedSlots + cost <= MAX_REQUESTS_PER_MINUTE) {
        this.requestTimestamps.push({ timestamp: now, cost });
        return;
      }

      const oldestRequestAt = this.requestTimestamps[0]!.timestamp;
      const waitMs = Math.max(1, RATE_LIMIT_WINDOW_MS - (now - oldestRequestAt));
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }

  /**
   * Prepare text for retrieval-oriented embedding.
   */
  private prepareText(text: string): string {
    const maxLength = 8192;
    return text.length > maxLength ? text.slice(0, maxLength) : text;
  }

  /**
   * Create text representation of a code node for embedding.
   */
  static createNodeText(node: {
    name: string;
    kind: string;
    qualifiedName?: string;
    signature?: string;
    docstring?: string;
    filePath: string;
  }): string {
    const parts: string[] = [];

    parts.push(`${node.kind}: ${node.name}`);

    if (node.qualifiedName && node.qualifiedName !== node.name) {
      parts.push(`path: ${node.qualifiedName}`);
    }

    parts.push(`file: ${node.filePath}`);

    if (node.signature) {
      parts.push(`signature: ${node.signature}`);
    }

    if (node.docstring) {
      parts.push(`documentation: ${node.docstring}`);
    }

    return parts.join('\n');
  }

  /**
   * Compute cosine similarity between two embeddings
   */
  static cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      const aVal = a[i]!;
      const bVal = b[i]!;
      dotProduct += aVal * bVal;
      normA += aVal * aVal;
      normB += bVal * bVal;
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (normA * normB);
  }

  /**
   * Release resources
   */
  dispose(): void {
    this.initialized = false;
  }
}

/**
 * Create a text embedder instance
 */
export function createEmbedder(options?: EmbedderOptions): TextEmbedder {
  return new TextEmbedder(options);
}
