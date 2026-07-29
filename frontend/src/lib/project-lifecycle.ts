export function defaultProjectCloneName(projectName: string): string {
  return `${projectName} (Copy)`;
}

export function projectDeletionConfirmed(
  confirmation: string,
  projectName: string,
): boolean {
  return confirmation === projectName;
}
