/**
 * Vectors Module
 *
 * Provides text embedding and vector similarity search for semantic code search.
 */

export {
  TextEmbedder,
  createEmbedder,
  DEFAULT_MODEL,
  DEFAULT_GEMINI_MODEL,
  DEFAULT_JINA_MODEL,
  EMBEDDING_DIMENSION,
  EmbeddingProvider,
  EmbedderOptions,
  EmbeddingResult,
  BatchEmbeddingResult,
  EmbedderStatusUpdate,
} from './embedder';

export {
  VectorSearchManager,
  createVectorSearch,
  VectorSearchOptions,
} from './search';

export {
  VectorManager,
  createVectorManager,
  VectorManagerOptions,
  EmbeddingProgress,
} from './manager';
