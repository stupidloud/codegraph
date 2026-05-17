/**
 * Vector Embedding Tests
 *
 * Tests for vector embedding and semantic search functionality.
 * Network embedding tests mock the Gemini API.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import CodeGraph from '../src/index';
import { TextEmbedder } from '../src/vectors/embedder';
import { VectorSearchManager, createVectorSearch } from '../src/vectors/search';
import { DatabaseConnection } from '../src/db';

describe('Vector Embeddings', () => {
  describe('TextEmbedder', () => {
    describe('createNodeText', () => {
      it('should create text representation from node', () => {
        const node = {
          name: 'processPayment',
          kind: 'function',
          qualifiedName: 'PaymentService.processPayment',
          signature: '(amount: number) => Promise<Receipt>',
          docstring: 'Process a payment and return a receipt.',
          filePath: 'src/services/payment.ts',
        };

        const text = TextEmbedder.createNodeText(node);

        expect(text).toContain('function: processPayment');
        expect(text).toContain('path: PaymentService.processPayment');
        expect(text).toContain('file: src/services/payment.ts');
        expect(text).toContain('signature: (amount: number) => Promise<Receipt>');
        expect(text).toContain('documentation: Process a payment');
      });

      it('should handle minimal node data', () => {
        const node = {
          name: 'helper',
          kind: 'function',
          filePath: 'src/utils.ts',
        };

        const text = TextEmbedder.createNodeText(node);

        expect(text).toContain('function: helper');
        expect(text).toContain('file: src/utils.ts');
        expect(text).not.toContain('signature:');
        expect(text).not.toContain('documentation:');
      });
    });

    describe('cosineSimilarity', () => {
      it('should compute similarity between identical vectors', () => {
        const vec = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
        const similarity = TextEmbedder.cosineSimilarity(vec, vec);

        expect(similarity).toBeCloseTo(1.0, 5);
      });

      it('should compute similarity between orthogonal vectors', () => {
        const vec1 = new Float32Array([1, 0, 0]);
        const vec2 = new Float32Array([0, 1, 0]);
        const similarity = TextEmbedder.cosineSimilarity(vec1, vec2);

        expect(similarity).toBeCloseTo(0.0, 5);
      });

      it('should compute similarity between opposite vectors', () => {
        const vec1 = new Float32Array([1, 0, 0]);
        const vec2 = new Float32Array([-1, 0, 0]);
        const similarity = TextEmbedder.cosineSimilarity(vec1, vec2);

        expect(similarity).toBeCloseTo(-1.0, 5);
      });

      it('should throw for vectors of different dimensions', () => {
        const vec1 = new Float32Array([1, 2, 3]);
        const vec2 = new Float32Array([1, 2]);

        expect(() => TextEmbedder.cosineSimilarity(vec1, vec2)).toThrow(
          'Embeddings must have the same dimension'
        );
      });

      it('should handle zero vectors', () => {
        const vec1 = new Float32Array([0, 0, 0]);
        const vec2 = new Float32Array([1, 2, 3]);
        const similarity = TextEmbedder.cosineSimilarity(vec1, vec2);

        expect(similarity).toBe(0);
      });
    });

    describe('Gemini API', () => {
      const originalFetch = global.fetch;

      afterEach(() => {
        global.fetch = originalFetch;
        vi.restoreAllMocks();
      });

      it('should call Gemini embedContent once per input text', async () => {
        const fetchMock = vi.fn(async () => ({
          ok: true,
          json: async () => ({ embedding: { values: [0.1, 0.2, 0.3] } }),
        })) as unknown as typeof fetch;
        global.fetch = fetchMock;

        const embedder = new TextEmbedder({
          apiKey: 'test-key',
          outputDimensionality: 3,
        });
        await embedder.initialize();

        const result = await embedder.embedBatch(['node one', 'node two'], 'document');

        expect(fetchMock).toHaveBeenCalledTimes(2);
        expect(result.embeddings).toHaveLength(2);
        const firstBody = JSON.parse((fetchMock as any).mock.calls[0][1].body);
        expect(firstBody.output_dimensionality).toBe(3);
        expect(firstBody.content.parts).toHaveLength(1);
        expect(firstBody.content.parts[0].text).toContain('task: code retrieval | document:');
      });

      it('should require an API key before initialization', async () => {
        const embedder = new TextEmbedder();
        await expect(embedder.initialize()).rejects.toThrow(/api key/i);
      });
    });
  });

  describe('VectorSearchManager', () => {
    let tempDir: string;
    let db: DatabaseConnection;
    let searchManager: VectorSearchManager;
    const TEST_DIMENSION = 3; // Use small dimension for tests

    beforeEach(() => {
      tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'codegraph-vector-test-'));
      const dbPath = path.join(tempDir, 'test.db');
      db = DatabaseConnection.initialize(dbPath);
      searchManager = createVectorSearch(db.getDb(), TEST_DIMENSION);
    });

    afterEach(() => {
      db.close();
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it('should store and retrieve vectors', async () => {
      await searchManager.initialize();

      const embedding = new Float32Array([0.1, 0.2, 0.3]);
      searchManager.storeVector('node1', embedding, 'test-model');

      const retrieved = searchManager.getVector('node1');

      expect(retrieved).not.toBeNull();
      expect(retrieved?.length).toBe(3);
      expect(retrieved?.[0]).toBeCloseTo(0.1, 5);
    });

    it('should return null for non-existent vectors', async () => {
      await searchManager.initialize();

      const retrieved = searchManager.getVector('non-existent');

      expect(retrieved).toBeNull();
    });

    it('should check if vector exists', async () => {
      await searchManager.initialize();

      const embedding = new Float32Array([0.1, 0.2, 0.3]);
      searchManager.storeVector('node1', embedding, 'test-model');

      expect(searchManager.hasVector('node1')).toBe(true);
      expect(searchManager.hasVector('node2')).toBe(false);
    });

    it('should delete vectors', async () => {
      await searchManager.initialize();

      const embedding = new Float32Array([0.1, 0.2, 0.3]);
      searchManager.storeVector('node1', embedding, 'test-model');

      expect(searchManager.hasVector('node1')).toBe(true);

      searchManager.deleteVector('node1');

      expect(searchManager.hasVector('node1')).toBe(false);
    });

    it('should count vectors', async () => {
      await searchManager.initialize();

      expect(searchManager.getVectorCount()).toBe(0);

      searchManager.storeVector('node1', new Float32Array([0.1, 0.2, 0.3]), 'test');
      searchManager.storeVector('node2', new Float32Array([0.4, 0.5, 0.6]), 'test');

      expect(searchManager.getVectorCount()).toBe(2);
    });

    it('should clear all vectors', async () => {
      await searchManager.initialize();

      searchManager.storeVector('node1', new Float32Array([0.1, 0.2, 0.3]), 'test');
      searchManager.storeVector('node2', new Float32Array([0.4, 0.5, 0.6]), 'test');

      expect(searchManager.getVectorCount()).toBe(2);

      searchManager.clear();

      expect(searchManager.getVectorCount()).toBe(0);
    });

    it('should perform brute-force similarity search', async () => {
      await searchManager.initialize();

      // Store some test vectors
      searchManager.storeVector('node1', new Float32Array([1, 0, 0]), 'test');
      searchManager.storeVector('node2', new Float32Array([0.9, 0.1, 0]), 'test');
      searchManager.storeVector('node3', new Float32Array([0, 1, 0]), 'test');

      // Search for similar to [1, 0, 0]
      const query = new Float32Array([1, 0, 0]);
      const results = searchManager.search(query, { limit: 3 });

      expect(results.length).toBe(3);
      expect(results[0].nodeId).toBe('node1'); // Most similar
      expect(results[0].score).toBeCloseTo(1.0, 5);
      expect(results[1].nodeId).toBe('node2'); // Second most similar
    });

    it('should respect minScore in search', async () => {
      await searchManager.initialize();

      searchManager.storeVector('node1', new Float32Array([1, 0, 0]), 'test');
      searchManager.storeVector('node2', new Float32Array([0, 1, 0]), 'test');

      const query = new Float32Array([1, 0, 0]);
      const results = searchManager.search(query, { limit: 10, minScore: 0.5 });

      // Only node1 should match with score >= 0.5
      expect(results.length).toBe(1);
      expect(results[0].nodeId).toBe('node1');
    });

    it('should store vectors in batch', async () => {
      await searchManager.initialize();

      // Use normalized 3-dimensional vectors
      const entries = [
        { nodeId: 'node1', embedding: new Float32Array([1.0, 0.0, 0.0]) },
        { nodeId: 'node2', embedding: new Float32Array([0.0, 1.0, 0.0]) },
        { nodeId: 'node3', embedding: new Float32Array([0.0, 0.0, 1.0]) },
      ];

      searchManager.storeVectorBatch(entries, 'test-model');

      expect(searchManager.getVectorCount()).toBe(3);
      expect(searchManager.hasVector('node1')).toBe(true);
      expect(searchManager.hasVector('node2')).toBe(true);
      expect(searchManager.hasVector('node3')).toBe(true);
    });

    it('should get indexed node IDs', async () => {
      await searchManager.initialize();

      searchManager.storeVector('node1', new Float32Array([0.1, 0.2, 0.3]), 'test');
      searchManager.storeVector('node2', new Float32Array([0.4, 0.5, 0.6]), 'test');

      const ids = searchManager.getIndexedNodeIds();

      expect(ids).toContain('node1');
      expect(ids).toContain('node2');
      expect(ids.length).toBe(2);
    });
  });

  describe('CodeGraph Embedding Integration', () => {
    let testDir: string;
    let cg: CodeGraph;

    beforeEach(() => {
      testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'codegraph-embed-integration-'));

      // Create a simple test file
      fs.writeFileSync(
        path.join(testDir, 'test.ts'),
        `
export function processData(input: string): string {
  return input.toUpperCase();
}
`
      );

      cg = CodeGraph.initSync(testDir, {
        config: {
          include: ['**/*.ts'],
          exclude: [],
        },
      });
    });

    afterEach(() => {
      if (cg) {
        cg.destroy();
      }
      if (fs.existsSync(testDir)) {
        fs.rmSync(testDir, { recursive: true, force: true });
      }
    });

    it('should report embeddings not initialized', () => {
      expect(cg.isEmbeddingsInitialized()).toBe(false);
    });

    it('should return null embedding stats when semantic search is disabled', () => {
      const stats = cg.getEmbeddingStats();
      expect(stats).toBeNull();
    });

    it('should throw when calling semanticSearch without initialization', async () => {
      await expect(cg.semanticSearch('test')).rejects.toThrow(/not initialized/i);
    });

    it('should throw when calling findSimilar without initialization', async () => {
      await expect(cg.findSimilar('test-id')).rejects.toThrow(/not initialized/i);
    });
  });
});
