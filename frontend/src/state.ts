import type { Project, ProbeResponse, Findings, GeneratedQuestion, TechniquePreset, EntityVerification } from './types';

// Simple reactive state using custom events
class State<T> {
  private value: T;
  private listeners = new Set<(value: T) => void>();

  constructor(initial: T) {
    this.value = initial;
  }

  get(): T {
    return this.value;
  }

  set(value: T): void {
    this.value = value;
    this.notify();
  }

  update(fn: (current: T) => T): void {
    this.value = fn(this.value);
    this.notify();
  }

  subscribe(listener: (value: T) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener(this.value);
    }
  }
}

// App state
export interface AppState {
  currentView: 'projects' | 'project';
  currentProject: string | null;
  projects: Project[];
  isLoading: boolean;
  error: string | null;
}

export const appState = new State<AppState>({
  currentView: 'projects',
  currentProject: null,
  projects: [],
  isLoading: false,
  error: null,
});

// Probe state
export interface ProbeState {
  topic: string;
  angles: string[];
  selectedModels: string[];
  runsPerQuestion: number;
  questionCount: number;
  techniquePreset: TechniquePreset;
  isRunning: boolean;
  isPaused: boolean;
  isGenerating: boolean;
  infiniteMode: boolean;
  questions: GeneratedQuestion[];
  responses: ProbeResponse[];
  findings: Findings | null;
  abortController: AbortController | null;
  hiddenEntities: string[];      // Negatives - avoid these
  promotedEntities: string[];    // Positives - focus on these
  shouldAutostart: boolean;      // Auto-run probe when project loads
  autoCurate: boolean;           // Let AI clean up noise while running
  narrative: string;             // Interrogator's working theory
  narrativeUpdated: number | null; // Timestamp of last narrative update
  userNotes: string;             // User's personal notes/wild theories (fed back to AI)
  narrativeHistory: string[];    // Past working theories (so AI doesn't lose context)
  activeModel: string | null;    // Currently queried model (for highlighting)
  currentQuestionIndex: number;  // Index of question being executed (-1 = none)
  entityVerification: EntityVerification | null;  // PUBLIC vs PRIVATE entity verification
}

const initialProbeState: ProbeState = {
  topic: '',
  angles: [],
  selectedModels: [],  // Empty = auto-survey all models
  runsPerQuestion: 20,
  questionCount: 5,
  techniquePreset: 'auto',
  isRunning: false,
  isPaused: false,
  isGenerating: false,
  infiniteMode: false,
  questions: [],
  responses: [],
  findings: null,
  abortController: null,
  hiddenEntities: [],
  promotedEntities: [],
  shouldAutostart: false,
  autoCurate: true,  // Default ON
  narrative: '',
  narrativeUpdated: null,
  userNotes: '',
  narrativeHistory: [],
  activeModel: null,
  currentQuestionIndex: -1,
  entityVerification: null,
};

export const probeState = new State<ProbeState>({ ...initialProbeState });

// Reset probe state to initial values (call when switching projects)
export function resetProbeState(): void {
  probeState.set({ ...initialProbeState });
}

// Ground truth state (hidden from probed models)
export interface GroundTruthState {
  facts: string[];
}

export const groundTruthState = new State<GroundTruthState>({
  facts: [],
});

// Navigation
export function navigateTo(view: 'projects' | 'project', projectName?: string, autostart?: boolean): void {
  appState.update((s) => ({
    ...s,
    currentView: view,
    currentProject: projectName ?? null,
  }));

  // Signal autostart via probe state
  if (autostart) {
    probeState.update(s => ({ ...s, shouldAutostart: true }));
  }

  // Update URL without reload
  const url = view === 'projects' ? '/' : `/project/${encodeURIComponent(projectName!)}`;
  history.pushState({ view, projectName }, '', url);
}

// Handle browser back/forward
window.addEventListener('popstate', (e) => {
  const state = e.state as { view: 'projects' | 'project'; projectName?: string } | null;
  if (state) {
    appState.update((s) => ({
      ...s,
      currentView: state.view,
      currentProject: state.projectName ?? null,
    }));
  }
});

// Initialize from URL
export function initFromUrl(): void {
  const path = window.location.pathname;
  const match = path.match(/^\/project\/(.+)$/);
  if (match) {
    const projectName = decodeURIComponent(match[1]);
    appState.update((s) => ({
      ...s,
      currentView: 'project',
      currentProject: projectName,
    }));
  }
}
