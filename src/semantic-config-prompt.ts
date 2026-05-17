import type { CodeGraphConfig } from './types';

type ClackPrompts = typeof import('@clack/prompts');

/**
 * Ask whether to enable Gemini-backed semantic search during project init.
 */
export async function promptSemanticSearchConfig(
  clack: ClackPrompts
): Promise<Pick<CodeGraphConfig, 'semanticSearch'> | undefined> {
  const enableSemanticSearch = await clack.confirm({
    message: 'Enable Gemini semantic search?',
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

  const apiKey = await clack.password({
    message: 'Gemini API key',
    placeholder: 'AIza...',
    validate(value) {
      if (!value.trim()) {
        return 'Gemini API key is required when semantic search is enabled';
      }
      return undefined;
    },
  });

  if (clack.isCancel(apiKey)) {
    clack.cancel('Initialization cancelled.');
    process.exit(0);
  }

  clack.log.info('Gemini semantic search will use gemini-embedding-2 with 768 dimensions.');

  return {
    semanticSearch: {
      enabled: true,
      provider: 'gemini',
      apiKey: apiKey.trim(),
      model: 'gemini-embedding-2',
      outputDimensionality: 768,
      batchSize: 32,
    },
  };
}
