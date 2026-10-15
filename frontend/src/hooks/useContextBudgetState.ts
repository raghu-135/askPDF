import { useCallback, useEffect, useState } from 'react';
import { getPromptTools } from '../lib/api';
import {
  type ContextBudgetRatios,
  contextBudgetFromDefaults,
  readStoredContextBudget,
  writeStoredContextBudget,
} from '../lib/context-budget';

export function useContextBudgetState() {
  const [defaultContextBudget, setDefaultContextBudget] = useState<ContextBudgetRatios | null>(null);
  const [contextBudget, setContextBudgetState] = useState<ContextBudgetRatios | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setReady(false);
    setError(null);
    getPromptTools()
      .then((res) => {
        if (cancelled) return;
        const defaults = contextBudgetFromDefaults(res.defaults.context_budget);
        setDefaultContextBudget(defaults);
        const stored = readStoredContextBudget();
        setContextBudgetState(stored ?? defaults);
        setReady(true);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : 'Failed to load context budget defaults';
        setError(message);
        setDefaultContextBudget(null);
        setContextBudgetState(null);
        setReady(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const setContextBudget = useCallback((budget: ContextBudgetRatios) => {
    setContextBudgetState(budget);
    writeStoredContextBudget(budget);
  }, []);

  return {
    contextBudget,
    defaultContextBudget,
    contextBudgetReady: ready && contextBudget !== null && defaultContextBudget !== null,
    contextBudgetError: error,
    setContextBudget,
  };
}
