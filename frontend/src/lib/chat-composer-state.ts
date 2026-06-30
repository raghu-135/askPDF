export type ChatComposerIndexingStatus = 'checking' | 'indexing' | 'ready' | 'blocked' | 'error';

export type ChatComposerStatus =
  | 'sending'
  | 'no_llm_selected'
  | 'llm_checking'
  | 'llm_unavailable'
  | 'llm_tools_unsupported'
  | 'embed_checking'
  | 'embed_unavailable'
  | 'index_error'
  | 'indexing'
  | 'ready';

export interface ChatComposerStateInput {
  loading: boolean;
  llmModel: string;
  isLlmModelValid: boolean | null;
  isLlmToolsSupported: boolean | null;
  isEmbedModelValid: boolean | null;
  indexingStatus: ChatComposerIndexingStatus;
  hasInput: boolean;
}

export interface ChatComposerState {
  status: ChatComposerStatus;
  disabled: boolean;
  busy: boolean;
  placeholder: string;
}

function locked(
  status: Exclude<ChatComposerStatus, 'ready'>,
  placeholder: string,
  busy = false
): ChatComposerState {
  return {
    status,
    disabled: true,
    busy,
    placeholder,
  };
}

export function getChatComposerState(input: ChatComposerStateInput): ChatComposerState {
  if (input.loading) {
    return locked('sending', 'Sending...', true);
  }

  if (!input.llmModel) {
    return locked('no_llm_selected', 'Select LLM model...');
  }

  if (input.isLlmModelValid === null) {
    return locked('llm_checking', 'Checking LLM model...', true);
  }

  if (input.isLlmModelValid === false) {
    return locked('llm_unavailable', 'Selected LLM model is unavailable.');
  }

  if (input.isLlmToolsSupported === false) {
    return locked('llm_tools_unsupported', 'Selected LLM does not support tools.');
  }

  if (input.isEmbedModelValid === null) {
    return locked('embed_checking', 'Checking embedding model...', true);
  }

  if (input.isEmbedModelValid === false || input.indexingStatus === 'blocked') {
    return locked('embed_unavailable', 'Blocked: selected embedding model is unavailable on server.');
  }

  if (input.indexingStatus === 'error') {
    return locked('index_error', 'Connection error. Please refresh to retry.');
  }

  if (input.indexingStatus !== 'ready') {
    return locked('indexing', 'Indexing your sources. This may take a moment...', true);
  }

  return {
    status: 'ready',
    disabled: false,
    busy: false,
    placeholder: `Ask a question about your documents...${input.hasInput ? '\n(Shift+Enter for new line)' : ''}`,
  };
}
