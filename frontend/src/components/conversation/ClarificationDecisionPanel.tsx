import React from 'react';
import { DecisionChoiceList } from './DecisionChoiceList';
import { ResizableDecisionPanel } from './ResizableDecisionPanel';
import type { ClarificationChoice } from '../../lib/clarification-options';

export function ClarificationDecisionPanel({
  choices,
  disabled = false,
  rootRef,
  onSelectOption,
  onCustomSubmit,
  onClose,
}: {
  choices: ClarificationChoice[];
  disabled?: boolean;
  rootRef?: React.RefObject<HTMLElement | null>;
  onSelectOption: (text: string) => void;
  onCustomSubmit: (text: string) => void;
  onClose?: () => void;
}) {
  if (!choices.length) {
    return null;
  }

  return (
    <ResizableDecisionPanel
      title="I need a bit more clarification. Did you mean:"
      variant="clarification"
      rootRef={rootRef}
      horizontalInset={1}
      minHeight={80}
      onClose={onClose}
    >
      <DecisionChoiceList
        presentation="buttons"
        choices={choices.map((choice, index) => ({
          id: String(index),
          label: choice.text,
        }))}
        disabled={disabled}
        onSelect={(_selectedChoice, text) => onSelectOption(text)}
        onCustomSubmit={onCustomSubmit}
        customPlaceholder="Type your own question"
      />
    </ResizableDecisionPanel>
  );
}