export type ChatActor = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  id: string;
  actor: ChatActor;
  content: string;
  timestamp: Date;
  previewImageUrl?: string;
  prediction?: {
    plantId: string;
    plantName: string;
    confidence: number;
    classConfidences: Array<{ plantId: string; plantName: string; value: number }>;
    breakdown: Array<{ plantId: string; plantName: string; value: number }>;
  };
  status?: 'pending' | 'complete' | 'error';
  error?: string;
}
