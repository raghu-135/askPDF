/**
 * Chat layout when a human-review interrupt is open (tool approval vs other reviews).
 */

export function isToolApprovalInterrupt(interrupt: { type?: string } | null | undefined): boolean {
  return interrupt?.type === 'tool_approval';
}

/** Tool-approval bottom sheet replaces the composer; only non-tool reviews lock the composer. */
export function composerHumanReviewPending(
  pendingInterrupt: { type?: string } | null | undefined,
): boolean {
  return Boolean(pendingInterrupt) && !isToolApprovalInterrupt(pendingInterrupt);
}

/** Non-tool human review uses the decision column and caps transcript height. */
export function chatHumanReviewUsesStackedDecisionLayout(
  pendingInterrupt: { type?: string } | null | undefined,
): boolean {
  return Boolean(pendingInterrupt) && !isToolApprovalInterrupt(pendingInterrupt);
}

/** Tool approval HITL renders in the composer slot at the bottom of the thread panel. */
export function chatToolApprovalReplacesComposer(
  pendingInterrupt: { type?: string } | null | undefined,
): boolean {
  return isToolApprovalInterrupt(pendingInterrupt);
}

/** Dismiss the in-thread tool-approval sheet as soon as the user chooses an action. */
export function shouldDismissToolApprovalOnAction(
  interrupt: { type?: string } | null | undefined,
): boolean {
  return isToolApprovalInterrupt(interrupt);
}