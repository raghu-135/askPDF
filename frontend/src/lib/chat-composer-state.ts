import {
  ChatComposerIndexingStatus,
  ChatComposerStatus,
  type ChatComposerIndexingStatus as ChatComposerIndexingStatusValue,
  type ChatComposerStatus as ChatComposerStatusValue,
} from './enums.ts';

export type { ChatComposerIndexingStatusValue as ChatComposerIndexingStatus };
export type { ChatComposerStatusValue as ChatComposerStatus };

export interface ChatComposerStateInput {
  loading: boolean;
  llmModel: string;
  isLlmModelValid: boolean | null;
  isLlmToolsSupported: boolean | null;
  isEmbeddingModelValid: boolean | null;
  indexingStatus: ChatComposerIndexingStatusValue;
  hasInput: boolean;
}

export interface ChatComposerState {
  status: ChatComposerStatusValue;
  disabled: boolean;
  busy: boolean;
  placeholder: string;
}

function locked(
  status: Exclude<ChatComposerStatusValue, typeof ChatComposerStatus.Ready>,
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
    return locked(ChatComposerStatus.Sending, 'Sending...', true);
  }

  if (!input.llmModel) {
    return locked(ChatComposerStatus.NoLlmSelected, 'Select LLM model...');
  }

  if (input.isLlmModelValid === null) {
    return locked(ChatComposerStatus.LlmChecking, 'Checking LLM model...', true);
  }

  if (input.isLlmModelValid === false) {
    return locked(ChatComposerStatus.LlmUnavailable, 'Selected LLM model is unavailable.');
  }

  if (input.isLlmToolsSupported === false) {
    return locked(ChatComposerStatus.LlmToolsUnsupported, 'Selected LLM does not support tools.');
  }

  if (input.isEmbeddingModelValid === null) {
    return locked(ChatComposerStatus.EmbeddingChecking, 'Checking embedding model...', true);
  }

  if (input.isEmbeddingModelValid === false || input.indexingStatus === ChatComposerIndexingStatus.Blocked) {
    return locked(ChatComposerStatus.EmbeddingUnavailable, 'Blocked: selected embedding model is unavailable on server.');
  }

  if (input.indexingStatus === ChatComposerIndexingStatus.Error) {
    return locked(ChatComposerStatus.IndexError, 'Connection error. Please refresh to retry.');
  }

  if (input.indexingStatus !== ChatComposerIndexingStatus.Ready) {
    return locked(ChatComposerStatus.Indexing, 'Indexing your sources. This may take a moment...', true);
  }

  return {
    status: ChatComposerStatus.Ready,
    disabled: false,
    busy: false,
    placeholder: `Ask a question about your documents...${input.hasInput ? '\n(Shift+Enter for new line)' : ''}`,
  };
}
