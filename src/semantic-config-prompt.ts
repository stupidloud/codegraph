import type { CodeGraphConfig } from './types';

type ClackPrompts = typeof import('@clack/prompts');

/**
 * Ask whether to enable remote-provider-backed semantic search during project init.
 */
export async function promptSemanticSearchConfig(
  clack: ClackPrompts
): Promise<Pick<CodeGraphConfig, 'semanticSearch'> | undefined> {
  const enableSemanticSearch = await clack.confirm({
    message: 'Enable semantic search with remote embeddings?',
    active: 'Yes',
    inactive: 'No',
    initialValue: false,
  });

  if (clack.isCancel(enableSemanticSearch)) {
    clack.cancel('Initialization cancelled.');
    process.exit(0);
  }

  if (!enableSemanticSearch) {
    return undefined;
  }

  const provider = await clack.select({
    message: 'Embedding provider',
    options: [
      { value: 'gemini', label: 'Gemini', hint: 'gemini-embedding-2, 768-d, batch 32' },
      { value: 'jina', label: 'Jina', hint: 'jina-embeddings-v5-text-nano, 768-d, batch 32' },
      { value: 'siliconflow', label: 'SiliconFlow', hint: 'BAAI/bge-m3, 1024-d, batch 1024, generous free tier' },
    ],
    initialValue: 'gemini',
  });

  if (clack.isCancel(provider)) {
    clack.cancel('Initialization cancelled.');
    process.exit(0);
  }

  const selectedProvider = provider as 'gemini' | 'jina' | 'siliconflow';
  const providerName =
    selectedProvider === 'jina' ? 'Jina' :
    selectedProvider === 'siliconflow' ? 'SiliconFlow' : 'Gemini';
  const keyPlaceholder =
    selectedProvider === 'jina' ? 'jina_...' :
    selectedProvider === 'siliconflow' ? 'sk-...' : 'AIza...';
  const apiKey = await clack.password({
    message: `${providerName} API key`,
    placeholder: keyPlaceholder,
    validate(value) {
      if (!value.trim()) {
        return `${providerName} API key is required when semantic search is enabled`;
      }
      return undefined;
    },
  });

  if (clack.isCancel(apiKey)) {
    clack.cancel('Initialization cancelled.');
    process.exit(0);
  }

  const model =
    selectedProvider === 'jina' ? 'jina-embeddings-v5-text-nano' :
    selectedProvider === 'siliconflow' ? 'BAAI/bge-m3' :
    'gemini-embedding-2';
  const dimension = selectedProvider === 'siliconflow' ? 1024 : 768;
  const defaultBatchSize = selectedProvider === 'siliconflow' ? 1024 : 32;
  clack.log.info(
    `${providerName} semantic search will use ${model} with ${dimension} dimensions, batch ${defaultBatchSize}.`
  );

  return {
    semanticSearch: {
      enabled: true,
      provider: selectedProvider,
      apiKey: apiKey.trim(),
      model,
      // batchSize intentionally omitted — let the embedder pull its default
      // from MODEL_CAPABILITIES so config files stay terse and future-proof.
    },
  };
}
