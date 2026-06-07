/**
 * Multi-provider embedder tests.
 *
 * Covers the MODEL_CAPABILITIES lookup table, the SiliconFlow request /
 * response path, and the embedder's getDimension/getDefaultBatchSize accessors
 * for every supported provider. Gemini and Jina request shapes have their own
 * existing coverage in `vectors.test.ts`; here we focus on the new
 * provider + capability-table machinery.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  TextEmbedder,
  createEmbedder,
  MODEL_CAPABILITIES,
  getModelCapabilities,
  DEFAULT_SILICONFLOW_MODEL,
} from '../src/vectors/embedder';

describe('MODEL_CAPABILITIES', () => {
  it('declares dimension 768 + batch 32 for gemini-embedding-2', () => {
    const caps = getModelCapabilities('gemini-embedding-2')!;
    expect(caps.dimension).toBe(768);
    expect(caps.defaultBatchSize).toBe(32);
  });

  it('declares dimension 768 + batch 32 for jina-embeddings-v5-text-nano', () => {
    const caps = getModelCapabilities('jina-embeddings-v5-text-nano')!;
    expect(caps.dimension).toBe(768);
    expect(caps.defaultBatchSize).toBe(32);
  });

  it('declares dimension 1024 + batch 1024 + max 4096 for BAAI/bge-m3', () => {
    const caps = getModelCapabilities('BAAI/bge-m3')!;
    expect(caps.dimension).toBe(1024);
    expect(caps.defaultBatchSize).toBe(1024);
    expect(caps.maxBatchSize).toBe(4096);
  });

  it('returns undefined for unknown models', () => {
    expect(getModelCapabilities('nonexistent-model-xyz')).toBeUndefined();
  });

  it('every known model satisfies defaultBatchSize <= maxBatchSize', () => {
    for (const [model, caps] of Object.entries(MODEL_CAPABILITIES)) {
      expect(
        caps.defaultBatchSize,
        `${model} default (${caps.defaultBatchSize}) must be <= max (${caps.maxBatchSize})`
      ).toBeLessThanOrEqual(caps.maxBatchSize);
    }
  });
});

describe('TextEmbedder accessors per provider/model', () => {
  it('Gemini → 768-d, batch 32', () => {
    const e = createEmbedder({ provider: 'gemini', apiKey: 'k' });
    expect(e.getDimension()).toBe(768);
    expect(e.getDefaultBatchSize()).toBe(32);
    expect(e.getMaxBatchSize()).toBe(100);
  });

  it('Jina → 768-d, batch 32', () => {
    const e = createEmbedder({ provider: 'jina', apiKey: 'k' });
    expect(e.getDimension()).toBe(768);
    expect(e.getDefaultBatchSize()).toBe(32);
    expect(e.getMaxBatchSize()).toBe(2048);
  });

  it('SiliconFlow → 1024-d, batch 1024, max 4096', () => {
    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'k' });
    expect(e.getDimension()).toBe(1024);
    expect(e.getDefaultBatchSize()).toBe(1024);
    expect(e.getMaxBatchSize()).toBe(4096);
  });


  it('siliconflow with default model resolves to BAAI/bge-m3', () => {
    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'k' });
    expect(e.getModelId()).toBe(DEFAULT_SILICONFLOW_MODEL);
  });
});

describe('SiliconFlow embed request', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // 1024-d dummy vectors so the dimension check passes.
    const fakeEmbedding = (seed: number) => {
      const arr = new Array(1024);
      for (let i = 0; i < 1024; i++) arr[i] = ((seed + i) % 100) / 100;
      return arr;
    };
    const fakeResponse = (n: number) => ({
      data: Array.from({ length: n }, (_, i) => ({
        embedding: fakeEmbedding(i),
        index: i,
        object: 'embedding',
      })),
      model: 'BAAI/bge-m3',
      usage: { prompt_tokens: n * 5, total_tokens: n * 5 },
    });

    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => {
      const body = JSON.parse((init as RequestInit).body as string);
      const count = Array.isArray(body.input) ? body.input.length : 1;
      return new Response(JSON.stringify(fakeResponse(count)), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    });
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it('POSTs OpenAI-compatible body to the SiliconFlow embeddings endpoint', async () => {
    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'sk-test-key' });
    await e.initialize();
    const result = await e.embedBatch(['alpha', 'beta', 'gamma'], 'document');

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe('https://api.siliconflow.cn/v1/embeddings');
    expect((init as RequestInit).method).toBe('POST');
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers['Authorization']).toBe('Bearer sk-test-key');
    expect(headers['Content-Type']).toBe('application/json');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.model).toBe('BAAI/bge-m3');
    expect(body.input).toEqual(['alpha', 'beta', 'gamma']);
    expect(body.encoding_format).toBe('float');

    expect(result.embeddings).toHaveLength(3);
    expect(result.embeddings[0]).toBeInstanceOf(Float32Array);
    expect(result.embeddings[0]!.length).toBe(1024);
    expect(result.dimension).toBe(1024);
    expect(result.model).toBe('BAAI/bge-m3');
  });

  it('throws when apiKey is missing', async () => {
    const e = createEmbedder({ provider: 'siliconflow' });
    await expect(e.initialize()).rejects.toThrow(/SiliconFlow API key is required/);
  });

  it('embedQuery hits the same endpoint with a single-element input', async () => {
    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'sk-x' });
    await e.initialize();
    await e.embedQuery('how does auth work');

    const [, init] = fetchSpy.mock.calls[0]!;
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.input).toEqual(['how does auth work']);
  });

});

describe('SiliconFlow 400 shrink-retry self-healing', () => {
  function tokenLimitResponse() {
    return new Response(
      JSON.stringify({ code: 20015, message: 'The parameter is invalid. Please check again.', data: null }),
      { status: 400, statusText: 'Bad Request', headers: { 'Content-Type': 'application/json' } }
    );
  }
  function okResponse(inputCount: number) {
    const data = Array.from({ length: inputCount }, (_, i) => ({
      embedding: new Array(1024).fill(0).map((_, j) => ((i + j) % 100) / 100),
      index: i,
      object: 'embedding',
    }));
    return new Response(JSON.stringify({ data, model: 'BAAI/bge-m3', usage: { total_tokens: 100 } }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  let fetchSpy: ReturnType<typeof vi.spyOn>;
  afterEach(() => { fetchSpy?.mockRestore(); });

  it('shrinks each text by 10% and retries when SiliconFlow returns 400 code 20015', async () => {
    fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(tokenLimitResponse())
      .mockResolvedValueOnce(tokenLimitResponse())
      .mockResolvedValueOnce(okResponse(2));

    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'sk-x' });
    await e.initialize();
    const big = 'x'.repeat(10000);
    const small = 'short';
    const result = await e.embedBatch([big, small], 'document');

    expect(fetchSpy).toHaveBeenCalledTimes(3);
    const body1 = JSON.parse((fetchSpy.mock.calls[0]![1] as RequestInit).body as string);
    const body2 = JSON.parse((fetchSpy.mock.calls[1]![1] as RequestInit).body as string);
    const body3 = JSON.parse((fetchSpy.mock.calls[2]![1] as RequestInit).body as string);
    expect(body1.input[0]).toHaveLength(10000);
    expect(body2.input[0]).toHaveLength(9000);
    expect(body3.input[0]).toHaveLength(8100);
    // Smaller text is left untouched throughout
    expect(body1.input[1]).toBe('short');
    expect(body2.input[1]).toBe('short');
    expect(body3.input[1]).toBe('short');

    expect(result.embeddings).toHaveLength(2);
  });

  it('does NOT retry on a 400 that is not the token-limit error (e.g. auth)', async () => {
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify({ code: 50001, message: 'Invalid API key' }), {
        status: 400,
        statusText: 'Bad Request',
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'sk-bad' });
    await e.initialize();
    await expect(e.embedBatch(['hello'], 'document')).rejects.toThrow(/SiliconFlow.*400/);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it('gives up when texts have been shrunk below the minimum length', async () => {
    // Each fetch call must return a fresh Response — bodies are one-shot streams.
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => tokenLimitResponse());

    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'sk-x' });
    await e.initialize();
    // Start at 1000 chars: log(100/1000)/log(0.9) ≈ 22 attempts to hit floor.
    await expect(e.embedBatch(['x'.repeat(1000)], 'document')).rejects.toThrow(/SiliconFlow.*400/);
    expect(fetchSpy.mock.calls.length).toBeGreaterThan(15);
    expect(fetchSpy.mock.calls.length).toBeLessThan(40);
  });

  it('reports shrink_retry phase via statusReporter so progress UI can show it', async () => {
    fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(tokenLimitResponse())
      .mockResolvedValueOnce(okResponse(1));

    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'sk-x' });
    await e.initialize();
    const updates: Array<{ phase: string; shrunkToChars?: number; attempt?: number }> = [];
    e.setStatusReporter((s) => updates.push(s as any));
    await e.embedBatch(['x'.repeat(5000)], 'document');

    expect(updates).toHaveLength(1);
    expect(updates[0].phase).toBe('shrink_retry');
    expect(updates[0].shrunkToChars).toBe(4500);
    expect(updates[0].attempt).toBe(1);
  });
});

describe('SiliconFlow embed error handling', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  afterEach(() => {
    fetchSpy?.mockRestore();
  });

  it('surfaces a friendly error including provider name + status + body', async () => {
    fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(
        JSON.stringify({ message: 'invalid api key', code: 401 }),
        { status: 401, statusText: 'Unauthorized', headers: { 'Content-Type': 'application/json' } }
      )
    );

    const e = createEmbedder({ provider: 'siliconflow', apiKey: 'sk-bad' });
    await e.initialize();
    await expect(e.embedBatch(['hello'], 'document')).rejects.toThrow(/SiliconFlow.*401/);
  });
});

describe('createNodeText still works (regression sanity)', () => {
  it('builds text from a node', () => {
    const text = TextEmbedder.createNodeText({
      name: 'foo',
      kind: 'function',
      filePath: 'src/foo.ts',
    });
    expect(text).toContain('function: foo');
    expect(text).toContain('file: src/foo.ts');
  });
});
