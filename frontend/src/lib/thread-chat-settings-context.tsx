import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type { ChatSettingsFormProps } from '../components/ChatSettingsForm';

export type ThreadChatSettingsValue = ChatSettingsFormProps & {
  saving: boolean;
  saveLabel?: string;
  onSave: () => void;
  onReset: () => void;
};

type ThreadChatSettingsRegistryValue = {
  registerSettings: (value: ThreadChatSettingsValue | null) => void;
};

const ThreadChatSettingsRegistryContext = createContext<ThreadChatSettingsRegistryValue | null>(null);
const ThreadChatSettingsStateContext = createContext<ThreadChatSettingsValue | null>(null);

export function ThreadChatSettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<ThreadChatSettingsValue | null>(null);

  const registerSettings = useCallback((value: ThreadChatSettingsValue | null) => {
    setSettings(value);
  }, []);

  const registryValue = useMemo(
    () => ({ registerSettings }),
    [registerSettings],
  );

  return (
    <ThreadChatSettingsRegistryContext.Provider value={registryValue}>
      <ThreadChatSettingsStateContext.Provider value={settings}>
        {children}
      </ThreadChatSettingsStateContext.Provider>
    </ThreadChatSettingsRegistryContext.Provider>
  );
}

export function useThreadChatSettingsRegistry() {
  const context = useContext(ThreadChatSettingsRegistryContext);
  if (!context) {
    throw new Error('useThreadChatSettingsRegistry must be used within ThreadChatSettingsProvider');
  }
  return context;
}

export function useThreadChatSettingsRegistryOptional() {
  return useContext(ThreadChatSettingsRegistryContext);
}

export function useThreadChatSettings() {
  return useContext(ThreadChatSettingsStateContext);
}
