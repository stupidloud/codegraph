/**
 * Text Embedder
 *
 * Generates vector embeddings through remote embedding APIs. Vector storage
 * and similarity search remain local.
 */

/**
 * Default embedding models.
 */
export type EmbeddingProvider = 'gemini' | 'jina' | 'siliconflow';
export const DEFAULT_PROVIDER: EmbeddingProvider = 'gemini';
export const DEFAULT_MODEL = 'gemini-embedding-2';
export const DEFAULT_GEMINI_MODEL = 'gemini-embedding-2';
export const DEFAULT_JINA_MODEL = 'jina-embeddings-v5-text-nano';
export const DEFAULT_SILICONFLOW_MODEL = 'BAAI/bge-m3';

export interface ModelCapabilities {
  dimension: number;
  defaultBatchSize: number;
  maxBatchSize: number;
}

// Per-model capabilities — single source of truth for vector dimension and
// provider-appropriate batch sizing. Token-limit overflow is handled at the
// SiliconFlow handler with a 400→shrink-10%→retry loop, not via a static
// per-text char cap (different content has wildly different token density;
// see `embedPreparedBatchWithSiliconFlow`). Gemini silently truncates and
// Jina honors `truncate: true`, so they need no client-side cap either.
export const MODEL_CAPABILITIES: Record<string, ModelCapabilities> = {
  'gemini-embedding-2':           { dimension: 768,  defaultBatchSize: 32,   maxBatchSize: 100  },
  'jina-embeddings-v5-text-nano': { dimension: 768,  defaultBatchSize: 32,   maxBatchSize: 2048 },
  'BAAI/bge-m3':                  { dimension: 1024, defaultBatchSize: 1024, maxBatchSize: 4096 },
};

const FALLBACK_DIMENSION = 768;
const FALLBACK_BATCH_SIZE = 32;

export function getModelCapabilities(modelId: string): ModelCapabilities | undefined {
  return MODEL_CAPABILITIES[modelId];
}

const GEMINI_EMBEDDING_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models';
const JINA_EMBEDDING_ENDPOINT = 'https://api.jina.ai/v1/embeddings';
const SILICONFLOW_EMBEDDING_ENDPOINT = 'https://api.siliconflow.cn/v1/embeddings';
const JINA_MAX_REQUESTS_PER_MINUTE = 500;
const RATE_LIMIT_WINDOW_MS = 60_000;
const MAX_NETWORK_RETRY_ATTEMPTS = 3;
const INITIAL_RETRY_DELAY_MS = 1_000;
const RETRY_DELAY_MULTIPLIER = 2;
const MAX_RETRY_DELAY_MS = 64_000;
const RETRYABLE_STATUS_CODES = new Set([408, 429, 500, 502, 503, 504]);

/**
 * Options for the embedder
 */
export interface EmbedderOptions {
  /** Embedding provider */
  provider?: EmbeddingProvider;

  /** Embedding provider API key */
  apiKey?: string;

  /** Model ID to use */
  modelId?: string;

  /** Whether to show progress during embedding generation */
  showProgress?: boolean;
}

export type EmbedderStatusUpdate =
  | {
      phase: 'retry_wait';
      provider: 'Gemini' | 'Jina' | 'SiliconFlow';
      retryInMs: number;
      attempt: number;
    }
  | {
      phase: 'shrink_retry';
      provider: 'SiliconFlow';
      shrunkToChars: number;
      attempt: number;
    };

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

// OpenAI-compatible response shape (used by Jina and SiliconFlow).
type OpenAICompatibleEmbeddingResponse = {
  data?: Array<{ embedding?: number[] }>;
};

/**
 * Text Embedder using a remote embedding provider.
 *
 * Gemini batch generation uses batchEmbedContents with one request per node
 * text. Do not combine multiple nodes into one content.parts array because
 * that produces a single aggregate embedding instead of node-level vectors.
 */
export class TextEmbedder {
  private provider: EmbeddingProvider;
  private apiKey?: string;
  private modelId: string;
  private initialized = false;
  private requestTimestamps: Array<{ timestamp: number; cost: number }> = [];
  private rateLimitQueue: Promise<void> = Promise.resolve();
  private statusReporter?: (status: EmbedderStatusUpdate) => void;

  constructor(options: EmbedderOptions = {}) {
    this.provider = options.provider || DEFAULT_PROVIDER;
    this.apiKey = options.apiKey;
    this.modelId = options.modelId || this.getDefaultModelForProvider(this.provider);
  }

  /**
   * Initialize the embedder by validating configuration.
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    if (!this.apiKey) {
      throw new Error(`${this.getProviderDisplayName()} API key is required for semantic search`);
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
   * Get the embedding dimension for the configured model.
   */
  getDimension(): number {
    return getModelCapabilities(this.modelId)?.dimension ?? FALLBACK_DIMENSION;
  }

  /**
   * Get the default batch size for the configured model.
   */
  getDefaultBatchSize(): number {
    return getModelCapabilities(this.modelId)?.defaultBatchSize ?? FALLBACK_BATCH_SIZE;
  }

  /**
   * Get the maximum allowed batch size for the configured model.
   */
  getMaxBatchSize(): number {
    return getModelCapabilities(this.modelId)?.maxBatchSize ?? FALLBACK_BATCH_SIZE;
  }

  setStatusReporter(reporter?: (status: EmbedderStatusUpdate) => void): void {
    this.statusReporter = reporter;
  }

  /**
   * Generate embedding for a single text
   */
  async embed(text: string): Promise<EmbeddingResult> {
    const result = await this.embedBatch([text], 'document');
    const embedding = result.embeddings[0];
    if (!embedding) {
      throw new Error(`${this.getProviderDisplayName()} embedding response did not include values`);
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
      throw new Error(`${this.getProviderDisplayName()} embedding response did not include values`);
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
        dimension: this.getDimension(),
        model: this.modelId,
        durationMs: 0,
      };
    }

    const startTime = Date.now();
    const embeddings = await this.embedPreparedBatch(texts, type);

    return {
      embeddings,
      dimension: embeddings[0]?.length ?? this.getDimension(),
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

    if (this.provider === 'jina') {
      return this.embedPreparedBatchWithJina(texts, type);
    }

    if (this.provider === 'siliconflow') {
      return this.embedPreparedBatchWithSiliconFlow(texts);
    }

    return this.embedPreparedBatchWithGemini(texts, type);
  }

  private async embedPreparedBatchWithGemini(
    texts: string[],
    type: 'document' | 'search_query'
  ): Promise<Float32Array[]> {
    const url = `${GEMINI_EMBEDDING_ENDPOINT}/${encodeURIComponent(this.modelId)}:batchEmbedContents`;
    const model = `models/${this.modelId}`;
    const taskType = type === 'search_query' ? 'CODE_RETRIEVAL_QUERY' : 'RETRIEVAL_DOCUMENT';
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
          outputDimensionality: this.getDimension(),
        })),
      }),
    });

    if (!response.ok) {
      throw new Error(await this.formatApiErrorMessage('Gemini', response));
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

  private async embedPreparedBatchWithJina(
    texts: string[],
    type: 'document' | 'search_query'
  ): Promise<Float32Array[]> {
    await this.throttleRequests(1);
    const response = await this.fetchWithRetry(JINA_EMBEDDING_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey!}`,
      },
      body: JSON.stringify({
        model: this.modelId,
        input: texts.map((text) => this.prepareJinaText(text, type)),
        normalized: true,
        embedding_type: 'float',
        truncate: true,
      }),
    });

    if (!response.ok) {
      throw new Error(await this.formatApiErrorMessage('Jina', response));
    }

    return this.parseOpenAICompatibleResponse(response, 'Jina', texts.length);
  }

  private async embedPreparedBatchWithSiliconFlow(texts: string[]): Promise<Float32Array[]> {
    // SiliconFlow returns 400 (code 20015 "The parameter is invalid") when an
    // input exceeds bge-m3's 8192-token cap, AND its documented `truncate`
    // field is not actually implemented for bge-m3 (probed: accepted on short
    // inputs but doesn't truncate oversized ones). BPE-vs-SentencePiece
    // tokenizer architectures differ by up to 40%, so any client-side
    // estimator (tiktoken etc.) is either inaccurate or heavy.
    //
    // Self-healing fix: on the specific 400+20015 response, shrink every
    // text in the batch by 10% (cap at the longest text's new length) and
    // retry. Loop converges in O(log_{1/0.9}) attempts for any input
    // density. Other 400s (auth, bad model) pass through to the caller.
    const MIN_TEXT_LENGTH = 100;
    const MAX_SHRINK_ATTEMPTS = 40;
    let current = texts;
    let attempt = 0;

    while (true) {
      const response = await this.fetchWithRetry(SILICONFLOW_EMBEDDING_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey!}`,
        },
        body: JSON.stringify({
          model: this.modelId,
          input: current,
          encoding_format: 'float',
        }),
      });

      if (response.ok) {
        return this.parseOpenAICompatibleResponse(response, 'SiliconFlow', current.length);
      }

      const errBody = await response.text().catch(() => '');
      const isTokenLimit = response.status === 400 && /"code"\s*:\s*20015/.test(errBody);
      const longest = current.reduce((m, t) => Math.max(m, t.length), 0);

      if (!isTokenLimit || attempt >= MAX_SHRINK_ATTEMPTS || longest <= MIN_TEXT_LENGTH) {
        throw new Error(this.buildSiliconFlowErrorMessage(response, errBody));
      }

      const newMax = Math.max(MIN_TEXT_LENGTH, Math.floor(longest * 0.9));
      current = current.map((t) => (t.length > newMax ? t.slice(0, newMax) : t));
      attempt++;
      this.reportShrinkRetry(newMax, attempt);
    }
  }

  private buildSiliconFlowErrorMessage(response: Response, body: string): string {
    const normalized = this.normalizeErrorBody(body);
    const status = response.statusText
      ? `${response.status} ${response.statusText}`
      : String(response.status);
    return `SiliconFlow embedding request failed (${status}): ${normalized || 'No response body'}`;
  }

  private async parseOpenAICompatibleResponse(
    response: Response,
    providerLabel: 'Jina' | 'SiliconFlow',
    expectedCount: number
  ): Promise<Float32Array[]> {
    const json = (await response.json()) as OpenAICompatibleEmbeddingResponse;
    const values = json.data?.map((embedding) => embedding.embedding);
    if (!values || values.length !== expectedCount) {
      throw new Error(`${providerLabel} batch embedding response count did not match request count`);
    }

    const embeddings = values.map((embeddingValues) => {
      if (!embeddingValues || embeddingValues.length === 0) {
        throw new Error(`${providerLabel} embedding response did not include values`);
      }
      return new Float32Array(embeddingValues);
    });

    if (embeddings.some((embedding) => embedding.length !== embeddings[0]!.length)) {
      throw new Error(`${providerLabel} batch embedding response contained mixed dimensions`);
    }

    return embeddings;
  }

  private async fetchWithRetry(url: string, init: RequestInit): Promise<Response> {
    let lastNetworkError: unknown;

    for (let attempt = 0; ; attempt++) {
      try {
        const response = await fetch(url, init);
        if (!this.isRetryableStatus(response.status)) {
          return response;
        }

        const retryDelayMs = await this.getRetryDelayMs(attempt, response);
        this.reportRetryWait(retryDelayMs, attempt + 1);
        await this.sleep(retryDelayMs);
      } catch (error) {
        lastNetworkError = error;
        if (attempt >= MAX_NETWORK_RETRY_ATTEMPTS) {
          break;
        }

        const retryDelayMs = await this.getRetryDelayMs(attempt);
        this.reportRetryWait(retryDelayMs, attempt + 1);
        await this.sleep(retryDelayMs);
      }
    }

    throw lastNetworkError instanceof Error
      ? lastNetworkError
      : new Error(`${this.getProviderDisplayName()} embedding request failed: ${String(lastNetworkError)}`);
  }

  private isRetryableStatus(status: number): boolean {
    return RETRYABLE_STATUS_CODES.has(status);
  }

  private async getRetryDelayMs(attempt: number, response?: Response): Promise<number> {
    const retryInfoDelayMs = await this.getRetryInfoDelayMs(response);
    const exponentialDelay = Math.min(
      INITIAL_RETRY_DELAY_MS * Math.pow(RETRY_DELAY_MULTIPLIER, attempt),
      MAX_RETRY_DELAY_MS
    );
    return Math.max(retryInfoDelayMs ?? 0, exponentialDelay);
  }

  private async getRetryInfoDelayMs(response?: Response): Promise<number | undefined> {
    const clonedResponse = response?.clone?.();
    const body = await clonedResponse?.text().catch(() => '');
    if (!body) {
      return undefined;
    }

    try {
      const parsed = JSON.parse(body) as {
        error?: { details?: Array<{ '@type'?: string; retryDelay?: string }> };
      };
      const retryDelay = parsed.error?.details?.find(
        (detail) => detail?.['@type'] === 'type.googleapis.com/google.rpc.RetryInfo'
      )?.retryDelay;
      return this.parseDurationMs(retryDelay);
    } catch {
      return undefined;
    }
  }

  private parseDurationMs(duration?: string): number | undefined {
    if (!duration) {
      return undefined;
    }

    const match = duration.trim().match(/^(-?\d+(?:\.\d+)?)s$/);
    if (!match) {
      return undefined;
    }

    const seconds = Number(match[1]);
    if (!Number.isFinite(seconds) || seconds < 0) {
      return undefined;
    }

    return Math.ceil(seconds * 1000);
  }

  private async sleep(ms: number): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, ms));
  }

  private reportRetryWait(retryInMs: number, attempt: number): void {
    this.statusReporter?.({
      phase: 'retry_wait',
      provider: this.getProviderDisplayName(),
      retryInMs,
      attempt,
    });
  }

  private reportShrinkRetry(shrunkToChars: number, attempt: number): void {
    this.statusReporter?.({
      phase: 'shrink_retry',
      provider: 'SiliconFlow',
      shrunkToChars,
      attempt,
    });
  }

  private async throttleRequests(cost: number): Promise<void> {
    const reservation = this.rateLimitQueue.then(() => this.reserveRequestSlots(cost));
    this.rateLimitQueue = reservation.catch(() => undefined);
    await reservation;
  }

  private async reserveRequestSlots(cost: number): Promise<void> {
    if (cost > JINA_MAX_REQUESTS_PER_MINUTE) {
      throw new Error(
        `${this.getProviderDisplayName()} embedding batch size ${cost} exceeds the ${JINA_MAX_REQUESTS_PER_MINUTE} requests-per-minute limit`
      );
    }

    while (true) {
      const now = Date.now();
      this.requestTimestamps = this.requestTimestamps.filter(
        ({ timestamp }) => now - timestamp < RATE_LIMIT_WINDOW_MS
      );

      const usedSlots = this.requestTimestamps.reduce((sum, entry) => sum + entry.cost, 0);
      if (usedSlots + cost <= JINA_MAX_REQUESTS_PER_MINUTE) {
        this.requestTimestamps.push({ timestamp: now, cost });
        return;
      }

      const oldestRequestAt = this.requestTimestamps[0]!.timestamp;
      const waitMs = Math.max(1, RATE_LIMIT_WINDOW_MS - (now - oldestRequestAt));
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }

  private prepareJinaText(text: string, type: 'document' | 'search_query'): string {
    return `${type === 'search_query' ? 'Query' : 'Document'}: ${text}`;
  }

  private async formatApiErrorMessage(providerName: string, response: Response): Promise<string> {
    const body = await response.text().catch(() => '');
    const normalizedBody = this.normalizeErrorBody(body);
    const status = response.statusText
      ? `${response.status} ${response.statusText}`
      : String(response.status);
    return `${providerName} embedding request failed (${status}): ${normalizedBody || 'No response body'}`;
  }

  private normalizeErrorBody(body: string): string {
    const trimmed = body.trim();
    if (!trimmed) {
      return '';
    }

    try {
      return JSON.stringify(JSON.parse(trimmed));
    } catch {
      return trimmed;
    }
  }

  private getDefaultModelForProvider(provider: EmbeddingProvider): string {
    switch (provider) {
      case 'jina':
        return DEFAULT_JINA_MODEL;
      case 'siliconflow':
        return DEFAULT_SILICONFLOW_MODEL;
      default:
        return DEFAULT_GEMINI_MODEL;
    }
  }

  private getProviderDisplayName(): 'Gemini' | 'Jina' | 'SiliconFlow' {
    switch (this.provider) {
      case 'jina':
        return 'Jina';
      case 'siliconflow':
        return 'SiliconFlow';
      default:
        return 'Gemini';
    }
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
    body?: string;
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

    // Body last: gemini-embedding-2 silently truncates beyond 8192 tokens, so
    // putting body at the tail means an overflow loses code-tail, not
    // signature or docstring — the bits that carry the most retrieval signal
    // per token.
    if (node.body) {
      parts.push(`code:\n${node.body}`);
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
