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
  embedding?: { values?: number[] };
  embeddings?: Array<{ values?: number[] }>;
};

/**
 * Text Embedder using Gemini.
 *
 * Gemini's embedContent endpoint returns one embedding for the provided content,
 * so batch generation intentionally calls it once per node text. Do not combine
 * multiple nodes into one content.parts array because that produces a single
 * aggregate embedding instead of node-level vectors.
 */
export class TextEmbedder {
  private apiKey?: string;
  private modelId: string;
  private dimension: number;
  private initialized = false;

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
    return this.embedPrepared(this.prepareText(text, 'document'));
  }

  /**
   * Generate embedding for a query.
   */
  async embedQuery(query: string): Promise<EmbeddingResult> {
    return this.embedPrepared(this.prepareText(query, 'search_query'));
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
    const embeddings: Float32Array[] = [];

    for (const text of texts) {
      const result = await this.embedPrepared(this.prepareText(text, type));
      embeddings.push(result.embedding);
    }

    return {
      embeddings,
      dimension: embeddings[0]?.length ?? this.dimension,
      model: this.modelId,
      durationMs: Date.now() - startTime,
    };
  }

  private async embedPrepared(text: string): Promise<EmbeddingResult> {
    if (!this.initialized) {
      throw new Error('Embedder not initialized. Call initialize() first.');
    }

    const url = `${GEMINI_EMBEDDING_ENDPOINT}/${encodeURIComponent(this.modelId)}:embedContent`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': this.apiKey!,
      },
      body: JSON.stringify({
        model: `models/${this.modelId}`,
        content: {
          parts: [{ text }],
        },
        output_dimensionality: this.dimension,
      }),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new Error(`Gemini embedding request failed (${response.status}): ${body || response.statusText}`);
    }

    const json = await response.json() as GeminiEmbeddingResponse;
    const values = json.embedding?.values ?? json.embeddings?.[0]?.values;
    if (!values || values.length === 0) {
      throw new Error('Gemini embedding response did not include values');
    }

    const embedding = new Float32Array(values);
    return {
      embedding,
      dimension: embedding.length,
      model: this.modelId,
    };
  }

  /**
   * Prepare text for retrieval-oriented embedding.
   */
  private prepareText(text: string, type: 'document' | 'search_query'): string {
    const maxLength = 8192;
    const truncatedText = text.length > maxLength ? text.slice(0, maxLength) : text;

    if (type === 'search_query') {
      return `task: code retrieval | query: ${truncatedText}`;
    }
    return `task: code retrieval | document: ${truncatedText}`;
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
