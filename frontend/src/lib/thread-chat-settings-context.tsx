import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type { ChatSettingsFormProps } from '../components/ChatSettingsForm';

export type ThreadChatSettingsValue = ChatSettingsFormProps & {
  saving: boolean;
  saveLabel?: string;
  onSave: () => void;
  onReset: () => void;
};

type ThreadChatSettingsContextValue = {
  settings: ThreadChatSettingsValue | null;
  registerSettings: (value: ThreadChatSettingsValue | null) => void;
};

const ThreadChatSettingsContext = createContext<ThreadChatSettingsContextValue | null>(null);

export function ThreadChatSettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<ThreadChatSettingsValue | null>(null);

  const registerSettings = useCallback((value: ThreadChatSettingsValue | null) => {
    setSettings(value);
  }, []);

  const contextValue = useMemo(
    () => ({ settings, registerSettings }),
    [registerSettings, settings],
  );

  return (
    <ThreadChatSettingsContext.Provider value={contextValue}>
      {children}
    </ThreadChatSettingsContext.Provider>
  );
}

export function useThreadChatSettingsRegistry() {
  const context = useContext(ThreadChatSettingsContext);
  if (!context) {
    throw new Error('useThreadChatSettingsRegistry must be used within ThreadChatSettingsProvider');
  }
  return context;
}

export function useThreadChatSettingsRegistryOptional() {
  return useContext(ThreadChatSettingsContext);
}

export function useThreadChatSettings() {
  const context = useContext(ThreadChatSettingsContext);
  return context?.settings ?? null;
}
