import type { Project, ProbeConfig, Findings, SSEEvent } from './types';

const API_BASE = '/api';

// Models
export interface ModelConfig {
  provider: string;
  model: string;
  env_key?: string;
  temperature?: number;
}

export interface ModelsResponse {
  default: string;
  models: Record<string, ModelConfig>;
}

export async function getModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

// Projects
export async function getProjects(): Promise<Project[]> {
  const res = await fetch(`${API_BASE}/projects`);
  if (!res.ok) throw new Error('Failed to fetch projects');
  return res.json();
}

export async function createProject(name: string, topic?: string): Promise<Project> {
  const res = await fetch(`${API_BASE}/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, topic: topic || name }),
  });
  if (!res.ok) throw new Error('Failed to create project');
  return res.json();
}

export async function getProject(name: string): Promise<Project> {
  const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error('Project not found');
  return res.json();
}

export async function updateProject(name: string, updates: Partial<Project>): Promise<Project> {
  const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  if (!res.ok) throw new Error('Failed to update project');
  return res.json();
}

export async function deleteProject(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}/delete`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete project');
}

// Findings
export async function getFindings(projectName: string): Promise<Findings> {
  const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/findings`);
  if (!res.ok) throw new Error('Failed to fetch findings');
  return res.json();
}

// Transcript (past interrogations)
export interface TranscriptQuestion {
  question_index: number;
  question: string;
  technique: string;
  responses: Array<{
    model: string;
    run_index: number;
    response: string;
    entities: string[];
    is_refusal: boolean;
  }>;
}

export interface TranscriptResponse {
  transcript: TranscriptQuestion[];
  total_responses: number;
  total_questions: number;
}

export async function getTranscript(projectName: string): Promise<TranscriptResponse> {
  const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectName)}/transcript`);
  if (!res.ok) throw new Error('Failed to fetch transcript');
  return res.json();
}

// Probe (SSE streaming)
export function startProbe(
  config: ProbeConfig,
  onEvent: (event: SSEEvent) => void,
  onError: (error: Error) => void
): AbortController {
  const controller = new AbortController();

  fetch(`${API_BASE}/probe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        throw new Error(`Probe failed: ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6)) as SSEEvent;
              onEvent(event);
            } catch {
              // Ignore parse errors
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err);
      }
    });

  return controller;
}

// Generate questions only (without running)
export async function generateQuestions(
  topic: string,
  angles: string[],
  count: number,
  technique_preset: string,
  entities_found?: string[],
  project?: string  // Project name for loading banned/promoted entities
): Promise<{ questions: Array<{ question: string; technique: string }> }> {
  const res = await fetch(`${API_BASE}/generate-questions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      topic,
      angles,
      count,
      technique_preset,
      entities_found: entities_found || [],
      project,
    }),
  });
  if (!res.ok) throw new Error('Failed to generate questions');
  return res.json();
}

// Cluster entities using k-means with silhouette score
export interface ClusterResult {
  id: number;
  entities: Array<{ entity: string; count: number }>;
  total_count: number;
  size: number;
}

export interface ClusterResponse {
  clusters: ClusterResult[];
  optimal_k: number;
  silhouette_score: number;
  silhouette_scores: Array<[number, number]>;
  total_entities: number;
  total_mentions: number;
}

export async function clusterEntities(
  projectName: string,
  minCount = 2,
  maxK = 10
): Promise<ClusterResponse> {
  const res = await fetch(`${API_BASE}/cluster-entities`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      project: projectName,
      min_count: minCount,
      max_k: maxK,
    }),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.error || 'Failed to cluster entities');
  }
  return res.json();
}

// Drill into a specific cluster (SSE streaming)
export function drillCluster(
  config: {
    project: string;
    topic: string;
    entities: string[];
    models: string[];
    runs_per_question?: number;
  },
  onEvent: (event: SSEEvent) => void,
  onError: (error: Error) => void
): AbortController {
  const controller = new AbortController();

  fetch(`${API_BASE}/drill-cluster`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        throw new Error(`Drill cluster failed: ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6)) as SSEEvent;
              onEvent(event);
            } catch {
              // Ignore parse errors
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err);
      }
    });

  return controller;
}
