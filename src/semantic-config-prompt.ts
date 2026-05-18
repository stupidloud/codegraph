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
      { value: 'gemini', label: 'Gemini', hint: 'gemini-embedding-001, 768 dimensions' },
      { value: 'jina', label: 'Jina', hint: 'jina-embeddings-v5-text-nano, 768 dimensions' },
    ],
    initialValue: 'gemini',
  });

  if (clack.isCancel(provider)) {
    clack.cancel('Initialization cancelled.');
    process.exit(0);
  }

  const selectedProvider = provider as 'gemini' | 'jina';
  const providerName = selectedProvider === 'jina' ? 'Jina' : 'Gemini';
  const apiKey = await clack.password({
    message: `${providerName} API key`,
    placeholder: selectedProvider === 'jina' ? 'jina_...' : 'AIza...',
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

  const model = selectedProvider === 'jina' ? 'jina-embeddings-v5-text-nano' : 'gemini-embedding-001';
  clack.log.info(`${providerName} semantic search will use ${model} with 768 dimensions.`);

  return {
    semanticSearch: {
      enabled: true,
      provider: selectedProvider,
      apiKey: apiKey.trim(),
      model,
      outputDimensionality: 768,
      batchSize: 32,
    },
  };
}
