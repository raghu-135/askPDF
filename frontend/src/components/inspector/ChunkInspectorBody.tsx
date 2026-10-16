import React from 'react';
import { ConversationDisclosure } from '../conversation/ConversationDisclosure';
import { InspectorTextPreview } from './InspectorTextPreview';
import { JsonPreview } from './JsonPreview';

export function ChunkInspectorBody({
  text,
  metadata,
  metadataMaxHeight = 220,
  textMaxHeight = false,
  defaultExpanded = true,
}: {
  text: string;
  metadata?: unknown;
  metadataMaxHeight?: number | false;
  textMaxHeight?: number | false;
  defaultExpanded?: boolean;
}) {
  const hasMetadata = metadata != null
    && (Array.isArray(metadata) ? metadata.length > 0 : typeof metadata === 'object'
      ? Object.keys(metadata as Record<string, unknown>).length > 0
      : true);

  return (
    <>
      <ConversationDisclosure label="Text" defaultExpanded={defaultExpanded}>
        <InspectorTextPreview text={text} maxHeight={textMaxHeight} />
      </ConversationDisclosure>
      {hasMetadata ? (
        <ConversationDisclosure label="Metadata" defaultExpanded={defaultExpanded}>
          <JsonPreview value={metadata} maxHeight={metadataMaxHeight} />
        </ConversationDisclosure>
      ) : null}
    </>
  );
}
