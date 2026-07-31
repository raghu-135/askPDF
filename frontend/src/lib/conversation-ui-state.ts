export const clampDecisionPanelRatio = (
  ratio: number,
  min = 0.16,
  max = 0.58,
) => Math.max(min, Math.min(max, ratio));

export type ConversationComposerButtonState = {
  mode: 'send' | 'stop';
  disabled: boolean;
  spinning: boolean;
};

export const getConversationComposerButtonState = ({
  disabled,
  busy,
  showStop,
  canStop,
  stopping,
  hasDraft,
  disableWhenEmpty,
}: {
  disabled: boolean;
  busy: boolean;
  showStop: boolean;
  canStop: boolean;
  stopping: boolean;
  hasDraft: boolean;
  disableWhenEmpty: boolean;
}): ConversationComposerButtonState => {
  if (showStop) {
    return {
      mode: 'stop',
      disabled: !canStop || stopping,
      spinning: stopping,
    };
  }
  return {
    mode: 'send',
    disabled: disabled || busy || (disableWhenEmpty && !hasDraft),
    spinning: busy,
  };
};
