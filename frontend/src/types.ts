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
  narrative_updated?: string;    // ISO timestamp when narrative was last updated
  user_notes?: string;           // User's personal notes/hunches (fed back to AI)
  selected_models?: string[];    // Models to use for probing
  questions?: GeneratedQuestion[];  // Current question queue
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

// 'auto' = random mix, or any template ID from /api/techniques
export type TechniquePreset = 'auto' | string;

// Question with technique annotation
export interface GeneratedQuestion {
  question: string;
  technique: TechniqueType | string;  // Can be custom technique name
  template?: string;  // Which template was used (from templates/techniques/*.yaml)
  color?: string;     // Template color from YAML (e.g., "#f85149")
  target_entity?: string;
}

// Technique type is now dynamic - loaded from templates/techniques/*.yaml
// Any string is valid since techniques are defined in YAML files
export type TechniqueType = string;

// Response from a single model run
export interface ProbeResponse {
  question_index: number;
  question: string;
  model: string;
  run_index: number;
  response: string;
  entities?: string[];               // All entities (for backwards compat)
  discovered_entities?: string[];    // Genuine findings NOT in user query
  introduced_entities?: string[];    // Entities echoed from user query (less valuable)
  is_refusal?: boolean;
}

// Scored entity with frequency and score
export interface ScoredEntity {
  entity: string;
  score: number;
  frequency: number;
}

// Entity match details for hover display
export interface EntityMatch {
  model: string;
  question: string;
  context: string;
  is_refusal: boolean;
  is_first_mention?: boolean;  // True if this was a genuine reveal (not an echo from context)
}

// Findings aggregate
export interface Findings {
  entities: Record<string, number>;  // entity -> count
  entity_matches?: Record<string, EntityMatch[]>;  // entity -> detailed matches for hover
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
  | 'models_selected'
  | 'cycle_start'
  | 'cycle_complete'
  | 'prompts'
  | 'validate_done'
  | 'grow_done'
  | 'init'
  | 'model_active'
  | 'entity_verification';

// Entity verification result - PUBLIC (found on web) vs PRIVATE (only in training data)
export interface EntityVerification {
  verified: Array<{ entity: string; url?: string; snippet?: string }>;  // PUBLIC - found on web
  unverified: Array<{ entity: string }>;  // PRIVATE - not found on web (interesting!)
  unknown: string[];  // No search results
  summary: string;
}

export interface SSEEvent {
  type: SSEEventType;
  data: unknown;
}

// Available models (fetched dynamically from /api/models)
export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
}

// Technique Template System
export interface TechniqueConfig {
  weight: number;           // 0.0-1.0
  prompt: string;           // Instructions for AI
  examples?: string[];      // Example questions
  triggers?: string[];      // When to use
}

export interface TechniqueTemplate {
  name: string;
  description: string;
  techniques: Record<string, TechniqueConfig>;
}

export interface TechniqueListItem {
  id: string;
  name: string;
  description: string;
  color: string;
  technique_count: number;
}

// Technique display info - legacy map for nice display names
// NOTE: Techniques now loaded dynamically from YAML files. This is a fallback for display.
// Unknown techniques will show their raw ID (e.g., "helpfulness_trap" instead of "Helpfulness Trap")
export const TECHNIQUE_INFO: Record<string, { name: string; description: string }> = {
  // Classic techniques (for backwards compatibility with saved data)
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
