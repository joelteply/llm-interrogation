// Project
export interface Project {
  name: string;
  topic?: string;
  angles?: string[];
  ground_truth?: string[];
  hidden_entities?: string[];    // Negative prompts - entities to avoid
  promoted_entities?: string[];  // Positive prompts - entities to focus on
  created_at?: string;
  corpus_count?: number;
  narrative?: string;            // Interrogator's working theory
}

// Probe configuration
export interface ProbeConfig {
  project: string;
  topic: string;
  angles: string[];
  models: string[];
  questions: (string | null)[];  // null = generate
  runs_per_question: number;
  questions_count: number;
  technique_preset: TechniquePreset;
  negative_entities?: string[];  // Entities to avoid in questioning
  positive_entities?: string[];  // Entities to focus on
  infinite_mode?: boolean;       // Keep running until manually stopped
  accumulate?: boolean;          // Accumulate from existing corpus (default true)
  auto_curate?: boolean;         // Let AI clean noise while running
}

export type TechniquePreset = 'balanced' | 'aggressive' | 'subtle';

// Question with technique annotation
export interface GeneratedQuestion {
  question: string;
  technique: TechniqueType;
  target_entity?: string;
}

export type TechniqueType =
  | 'scharff_illusion'
  | 'scharff_confirmation'
  | 'scharff_ignore'
  | 'fbi_false_statement'
  | 'fbi_bracketing'
  | 'fbi_macro_to_micro'
  | 'fbi_disbelief'
  | 'fbi_flattery'
  | 'fbi_half_sentence'
  | 'cognitive_context'
  | 'cognitive_perspective'
  | 'cognitive_reverse';

// Response from a single model run
export interface ProbeResponse {
  question_index: number;
  question: string;
  model: string;
  run_index: number;
  response: string;
  entities?: string[];
  is_refusal?: boolean;
}

// Scored entity with frequency and score
export interface ScoredEntity {
  entity: string;
  score: number;
  frequency: number;
}

// Findings aggregate
export interface Findings {
  entities: Record<string, number>;  // entity -> count
  scored_entities?: ScoredEntity[];  // entities with scores (frequency × specificity × connections)
  by_model: Record<string, Record<string, number>>;  // model -> entity -> count
  by_question: Record<number, Record<string, number>>;  // question_index -> entity -> count
  warmth_scores?: Record<string, number>;  // entity -> warmth score (if ground truth set)
  corpus_size: number;
  refusal_rate: number;
  // Dead-end detection: entities leading only to public/generic info
  dead_ends?: string[];
  // Live threads: entities with high-quality connections to pursue
  live_threads?: string[];
}

// SSE event types
export type SSEEventType =
  | 'generating'
  | 'questions'
  | 'run_start'
  | 'response'
  | 'batch_complete'
  | 'findings_update'
  | 'pivot_suggested'
  | 'complete'
  | 'status'
  | 'error'
  | 'phase'
  | 'curate_ban'
  | 'curate_promote'
  | 'narrative'
  | 'cycle_start'
  | 'cycle_complete'
  | 'prompts'
  | 'validate_done'
  | 'grow_done'
  | 'init';

export interface SSEEvent {
  type: SSEEventType;
  data: unknown;
}

// Available models
export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
}

export const AVAILABLE_MODELS: ModelInfo[] = [
  { id: 'groq/llama-3.1-8b-instant', name: 'Llama 3.1 8B', provider: 'Groq' },
  { id: 'groq/llama-3.1-70b-versatile', name: 'Llama 3.1 70B', provider: 'Groq' },
  { id: 'deepseek/deepseek-chat', name: 'DeepSeek Chat', provider: 'DeepSeek' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini', provider: 'OpenAI' },
];

// Technique display info
export const TECHNIQUE_INFO: Record<TechniqueType, { name: string; description: string }> = {
  scharff_illusion: { name: 'Illusion of Knowing', description: 'Frame as confirming known facts' },
  scharff_confirmation: { name: 'Confirmation', description: 'Present claims to verify' },
  scharff_ignore: { name: 'Ignore Reveals', description: 'Downplay to get more' },
  fbi_false_statement: { name: 'False Statement', description: 'Wrong info to trigger correction' },
  fbi_bracketing: { name: 'Bracketing', description: 'Range estimates to get specifics' },
  fbi_macro_to_micro: { name: 'Macro to Micro', description: 'Start broad, narrow focus' },
  fbi_disbelief: { name: 'Disbelief', description: 'Question validity for elaboration' },
  fbi_flattery: { name: 'Flattery', description: 'Appeal to authority/expertise' },
  fbi_half_sentence: { name: 'Half Sentence', description: 'Incomplete to prompt completion' },
  cognitive_context: { name: 'Context Reinstatement', description: 'Imagine reviewing records' },
  cognitive_perspective: { name: 'Change Perspective', description: 'Ask from other viewpoints' },
  cognitive_reverse: { name: 'Reverse Order', description: 'Outcomes first, then causes' },
};
