/**
 * Vectors Module
 *
 * Provides text embedding and vector similarity search for semantic code search.
 */

export {
  TextEmbedder,
  createEmbedder,
  DEFAULT_MODEL,
  EMBEDDING_DIMENSION,
  EmbedderOptions,
  EmbeddingResult,
  BatchEmbeddingResult,
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
