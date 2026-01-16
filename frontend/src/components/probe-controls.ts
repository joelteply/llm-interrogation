import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { probeState, groundTruthState, type ProbeState } from '../state';
import { startProbe, updateProject, generateQuestions, clusterEntities, drillCluster, getAvailableModels, getApiKeysError, getTechniques, type ClusterResult } from '../api';
import { type TechniquePreset, type SSEEvent, type GeneratedQuestion, type ModelInfo, type TechniqueListItem } from '../types';

@customElement('probe-controls')
export class ProbeControls extends LitElement {
  static styles = css`
    :host {
      display: block;
    }

    .card {
      background: var(--bg-secondary, #161b22);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 8px;
      padding: 16px;
    }

    h3 {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary, #c9d1d9);
      margin-bottom: 16px;
    }

    .field {
      margin-bottom: 16px;
    }

    label {
      display: block;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary, #8b949e);
      margin-bottom: 6px;
    }

    input[type="text"],
    textarea {
      width: 100%;
      padding: 8px 12px;
      background: var(--bg-primary, #0d1117);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 6px;
      color: var(--text-primary, #c9d1d9);
      font-size: 14px;
      font-family: inherit;
    }

    input:focus,
    textarea:focus {
      outline: none;
      border-color: var(--accent-blue, #58a6ff);
    }

    textarea {
      resize: vertical;
      min-height: 60px;
    }

    .label-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }

    .label-row label {
      margin-bottom: 0;
    }

    .label-row button {
      font-size: 11px;
      padding: 2px 8px;
      background: var(--bg-tertiary, #21262d);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 4px;
      color: var(--text-secondary, #8b949e);
      cursor: pointer;
    }

    .label-row button:hover {
      background: var(--accent-blue, #58a6ff);
      color: white;
      border-color: var(--accent-blue, #58a6ff);
    }

    .models {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 200px;
      overflow-y: auto;
    }

    .model-option {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .model-option input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--accent-blue, #58a6ff);
    }

    .model-option span {
      font-size: 13px;
      color: var(--text-primary, #c9d1d9);
    }

    .api-keys-error {
      padding: 12px;
      background: rgba(248, 81, 73, 0.1);
      border: 1px solid var(--accent-red, #f85149);
      border-radius: 6px;
      font-size: 13px;
    }
    .api-keys-error p { margin: 0 0 8px 0; }
    .api-keys-error ul { margin: 8px 0 0 0; padding-left: 20px; }
    .api-keys-error li { margin: 4px 0; }
    .api-keys-error code { background: var(--bg-tertiary, #21262d); padding: 2px 6px; border-radius: 4px; font-size: 12px; }
    .api-keys-error a { color: var(--accent-blue, #58a6ff); }
    .api-keys-error em { color: var(--text-muted, #6e7681); font-size: 11px; }

    .model-option .provider {
      font-size: 11px;
      color: var(--text-muted, #6e7681);
    }

    .slider-field {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    input[type="range"] {
      flex: 1;
      height: 4px;
      accent-color: var(--accent-blue, #58a6ff);
    }

    .slider-value {
      min-width: 32px;
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary, #c9d1d9);
      text-align: right;
    }

    .infinity-toggle {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      background: var(--bg-tertiary, #21262d);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 6px;
      cursor: pointer;
      transition: all 150ms ease;
      font-size: 12px;
      color: var(--text-secondary, #8b949e);
    }

    .infinity-toggle:hover {
      border-color: var(--accent-blue, #58a6ff);
    }

    .infinity-toggle.active {
      background: rgba(88, 166, 255, 0.15);
      border-color: var(--accent-blue, #58a6ff);
      color: var(--accent-blue, #58a6ff);
    }

    .infinity-toggle .icon {
      font-size: 14px;
    }

    .technique-select {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .technique-select select {
      width: 100%;
      padding: 10px 12px;
      background: var(--bg-primary, #0d1117);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 6px;
      color: var(--text-primary, #c9d1d9);
      font-size: 14px;
      cursor: pointer;
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='%238b949e' viewBox='0 0 16 16'%3E%3Cpath d='M4.427 7.427l3.396 3.396a.25.25 0 00.354 0l3.396-3.396A.25.25 0 0011.396 7H4.604a.25.25 0 00-.177.427z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 12px center;
      background-size: 16px;
      padding-right: 36px;
    }

    .technique-select select:focus {
      outline: none;
      border-color: var(--accent-blue, #58a6ff);
    }

    .technique-select select:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .technique-description {
      font-size: 11px;
      color: var(--text-muted, #6e7681);
      font-style: italic;
    }

    .technique-link {
      font-size: 11px;
      color: var(--accent-blue, #58a6ff);
      text-decoration: none;
      cursor: pointer;
    }

    .technique-link:hover {
      text-decoration: underline;
    }

    .actions {
      display: flex;
      gap: 8px;
      margin-top: 16px;
    }

    .actions button {
      flex: 1;
      padding: 10px 16px;
      border: none;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 150ms ease;
    }

    .btn-primary {
      background: var(--accent-green, #3fb950);
      color: white;
    }

    .btn-primary:hover:not(:disabled) {
      background: #46c258;
    }

    .btn-secondary {
      background: var(--bg-tertiary, #21262d);
      border: 1px solid var(--border-default, #30363d);
      color: var(--text-primary, #c9d1d9);
    }

    .btn-secondary:hover:not(:disabled) {
      border-color: var(--text-muted, #6e7681);
    }

    .btn-danger {
      background: var(--accent-red, #f85149);
      color: white;
    }

    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .btn-large {
      padding: 14px 20px !important;
      font-size: 16px !important;
      font-weight: 600 !important;
    }

    .action-hint {
      margin-top: 12px;
      padding: 10px;
      background: rgba(210, 153, 34, 0.1);
      border: 1px solid rgba(210, 153, 34, 0.3);
      border-radius: 6px;
      font-size: 12px;
      color: #d29922;
      text-align: center;
    }

    .ground-truth-section {
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--border-default, #30363d);
    }

    .ground-truth-section h4 {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary, #8b949e);
      margin-bottom: 8px;
    }

    .hint {
      font-size: 11px;
      color: var(--text-muted, #6e7681);
      margin-top: 4px;
    }

    .cluster-section {
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--border-default, #30363d);
    }

    .cluster-section h4 {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary, #8b949e);
      margin-bottom: 8px;
    }

    .cluster-info {
      font-size: 11px;
      color: var(--text-muted, #6e7681);
      margin-top: 8px;
    }

    .clusters {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 12px;
    }

    .cluster {
      padding: 10px;
      background: var(--bg-primary, #0d1117);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 6px;
      cursor: pointer;
      transition: all 150ms ease;
    }

    .cluster:hover {
      border-color: var(--accent-blue, #58a6ff);
      background: rgba(88, 166, 255, 0.1);
    }

    .cluster-header {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-primary, #c9d1d9);
      margin-bottom: 6px;
    }

    .cluster-count {
      font-weight: 400;
      color: var(--text-muted, #6e7681);
      margin-left: 8px;
    }

    .cluster-entities {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }

    .cluster-entity {
      padding: 2px 6px;
      background: var(--bg-tertiary, #21262d);
      border-radius: 4px;
      font-size: 10px;
      color: var(--text-secondary, #8b949e);
    }

    .cluster-more {
      font-size: 10px;
      color: var(--text-muted, #6e7681);
      font-style: italic;
    }
  `;

  @property({ type: String })
  projectName: string | null = null;

  @state()
  private _probeState: ProbeState = probeState.get();

  @state()
  private _groundTruth: string[] = groundTruthState.get().facts;

  @state()
  private isGenerating = false;

  @state()
  private isClustering = false;

  @state()
  private clusters: ClusterResult[] = [];

  @state()
  private optimalK: number | null = null;

  @state()
  private availableModels: ModelInfo[] = [];

  @state()
  private availableTechniques: TechniqueListItem[] = [];

  @state()
  private apiKeysError: { message: string; supported_keys: Array<{key: string; provider: string; url: string; note?: string}> } | null = null;

  @state()
  private silhouetteScore: number | null = null;

  @state()
  private isDrilling = false;

  private _unsubscribes: Array<() => void> = [];

  connectedCallback() {
    super.connectedCallback();
    // Fetch available models from API (not hardcoded)
    getAvailableModels().then(models => {
      this.availableModels = models;
      // Check if no API keys configured
      this.apiKeysError = getApiKeysError();
    }).catch(err => {
      console.error('Failed to fetch models:', err);
    });
    // Fetch available technique templates
    getTechniques().then(techniques => {
      this.availableTechniques = techniques;
    }).catch(err => {
      console.error('Failed to fetch techniques:', err);
    });
    this._unsubscribes.push(
      probeState.subscribe((s) => {
        this._probeState = s;
      }),
      groundTruthState.subscribe((s) => {
        this._groundTruth = s.facts;
      })
    );

    // Listen for regenerate-questions event from question-queue
    this._regenerateHandler = () => this.handleGenerate();
    window.addEventListener('regenerate-questions', this._regenerateHandler);
  }

  private _regenerateHandler?: () => void;

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribes.forEach((unsub) => unsub());
    if (this._regenerateHandler) {
      window.removeEventListener('regenerate-questions', this._regenerateHandler);
    }
  }

  private updateTopic(e: Event) {
    const topic = (e.target as HTMLInputElement).value;
    probeState.update((s) => ({ ...s, topic }));
  }

  private handleAnglesChanged(e: CustomEvent<{ tags: string[] }>) {
    probeState.update((s) => ({ ...s, angles: e.detail.tags }));
  }

  private handleGroundTruthChanged(e: CustomEvent<{ tags: string[] }>) {
    groundTruthState.update((s) => ({ ...s, facts: e.detail.tags }));
  }

  private toggleModel(modelId: string) {
    probeState.update((s) => {
      const selected = s.selectedModels.includes(modelId)
        ? s.selectedModels.filter((m) => m !== modelId)
        : [...s.selectedModels, modelId];
      // Save to project so hot-reload picks up mid-run additions
      if (this.projectName) {
        updateProject(this.projectName, { selected_models: selected });
      }
      return { ...s, selectedModels: selected };
    });
  }

  private selectAllModels() {
    const allModelIds = this.availableModels.map((m) => m.id);
    probeState.update((s) => {
      if (this.projectName) {
        updateProject(this.projectName, { selected_models: allModelIds });
      }
      return { ...s, selectedModels: allModelIds };
    });
  }

  private clearAllModels() {
    probeState.update((s) => {
      if (this.projectName) {
        updateProject(this.projectName, { selected_models: [] });
      }
      return { ...s, selectedModels: [] };
    });
  }

  private updateRuns(e: Event) {
    const runs = parseInt((e.target as HTMLInputElement).value, 10);
    probeState.update((s) => ({ ...s, runsPerQuestion: runs }));
  }

  private toggleInfiniteMode() {
    probeState.update((s) => ({ ...s, infiniteMode: !s.infiniteMode }));
  }

  private updateQuestionCount(e: Event) {
    const count = parseInt((e.target as HTMLInputElement).value, 10);
    probeState.update((s) => ({ ...s, questionCount: count }));
  }

  private setTechniquePreset(preset: TechniquePreset) {
    probeState.update((s) => ({ ...s, techniquePreset: preset }));
  }

  private async handleGenerate() {
    if (!this._probeState.topic.trim()) return;

    this.isGenerating = true;
    try {
      const result = await generateQuestions(
        this._probeState.topic,
        this._probeState.angles,
        this._probeState.questionCount,
        this._probeState.techniquePreset,
        this._probeState.findings?.entities
          ? Object.keys(this._probeState.findings.entities)
          : undefined,
        this.projectName || undefined  // Pass project for banned/promoted entity context
      );
      probeState.update((s) => ({
        ...s,
        questions: result.questions as GeneratedQuestion[],
      }));
    } catch (err) {
      console.error('Failed to generate questions:', err);
    } finally {
      this.isGenerating = false;
    }
  }

  private handleRun() {
    if (!this.projectName || !this._probeState.topic.trim()) return;
    // Empty models = auto-survey all available models
    // if (this._probeState.selectedModels.length === 0) return;

    // Save project state
    updateProject(this.projectName, {
      topic: this._probeState.topic,
      angles: this._probeState.angles,
      ground_truth: this._groundTruth,
    }).catch(console.error);

    const controller = startProbe(
      {
        project: this.projectName,
        topic: this._probeState.topic,
        angles: this._probeState.angles,
        models: this._probeState.selectedModels,
        questions: this._probeState.questions.length
          ? this._probeState.questions  // Full objects with technique preserved
          : Array(this._probeState.questionCount).fill(null),
        runs_per_question: this._probeState.runsPerQuestion,
        questions_count: this._probeState.questionCount,
        technique_preset: this._probeState.techniquePreset,
        negative_entities: this._probeState.hiddenEntities,
        positive_entities: this._probeState.promotedEntities,
        infinite_mode: this._probeState.infiniteMode,
        accumulate: true,  // Always accumulate from existing corpus
      },
      (event: SSEEvent) => this.handleSSEEvent(event),
      (error: Error) => {
        console.error('Probe error:', error);
        probeState.update((s) => ({
          ...s,
          isRunning: false,
          abortController: null,
        }));
      }
    );

    probeState.update((s) => ({
      ...s,
      isRunning: true,
      // Don't clear responses - accumulate from existing
      abortController: controller,
    }));
  }

  private handleSSEEvent(event: SSEEvent) {
    switch (event.type) {
      case 'questions':
        // Merge new questions - only update if we're getting more or starting fresh
        const newQs = event.data as GeneratedQuestion[];
        probeState.update((s) => {
          if (newQs.length > s.questions.length || s.questions.length === 0) {
            return { ...s, questions: newQs };
          }
          return s;
        });
        break;
      case 'run_start':
        // Starting to execute a question - track the index
        probeState.update((s) => ({
          ...s,
          currentQuestionIndex: (event.data as any).question_index ?? -1,
        }));
        break;
      case 'model_active':
        // Currently querying this model
        probeState.update((s) => ({
          ...s,
          activeModel: (event.data as any).model ?? null,
        }));
        break;
      case 'response':
        probeState.update((s) => ({
          ...s,
          responses: [...s.responses, event.data as any],
        }));
        break;
      case 'findings_update':
        probeState.update((s) => ({
          ...s,
          findings: event.data as any,
        }));
        break;
      case 'complete':
        probeState.update((s) => ({
          ...s,
          isRunning: false,
          abortController: null,
          currentQuestionIndex: -1,
          activeModel: null,
        }));
        // Auto-continue if in infinite mode
        if (this._probeState.infiniteMode) {
          setTimeout(() => this.handleRun(), 1000);  // Brief pause then continue
        }
        break;
      case 'error':
        console.error('Probe error:', event.data);
        probeState.update((s) => ({
          ...s,
          isRunning: false,
          abortController: null,
          currentQuestionIndex: -1,
          activeModel: null,
        }));
        break;
    }
  }

  private handleStop() {
    const controller = this._probeState.abortController;
    if (controller) {
      controller.abort();
      probeState.update((s) => ({
        ...s,
        isRunning: false,
        abortController: null,
        currentQuestionIndex: -1,
        activeModel: null,
      }));
    }
  }

  private async handleCluster() {
    if (!this.projectName) return;

    this.isClustering = true;
    this.clusters = [];
    try {
      const result = await clusterEntities(this.projectName, 2, 10);
      this.clusters = result.clusters;
      this.optimalK = result.optimal_k;
      this.silhouetteScore = result.silhouette_score;
    } catch (err) {
      console.error('Clustering failed:', err);
    } finally {
      this.isClustering = false;
    }
  }

  private handleDrillCluster(cluster: ClusterResult) {
    if (!this.projectName || this.isDrilling) return;

    const entities = cluster.entities.map((e) => e.entity);

    this.isDrilling = true;
    const controller = drillCluster(
      {
        project: this.projectName,
        topic: this._probeState.topic,
        entities,
        models: this._probeState.selectedModels,
        runs_per_question: Math.min(this._probeState.runsPerQuestion, 5),
      },
      (event: SSEEvent) => this.handleSSEEvent(event),
      (error: Error) => {
        console.error('Drill error:', error);
        this.isDrilling = false;
      }
    );

    probeState.update((s) => ({
      ...s,
      isRunning: true,
      abortController: controller,
    }));
  }

  render() {
    const { topic, angles, selectedModels, runsPerQuestion, questionCount, techniquePreset, isRunning } =
      this._probeState;

    // Allow running with 0 models = auto-survey
    const canRun = topic.trim() && !isRunning;

    return html`
      <div class="card">
        <h3>Probe Configuration</h3>

        <div class="field">
          <label>Topic / Target</label>
          <textarea
            placeholder="e.g., Theranos board members 2015"
            .value=${topic}
            @input=${this.updateTopic}
            ?disabled=${isRunning}
          ></textarea>
        </div>

        <div class="field">
          <label>Investigation Angles</label>
          <tag-input
            .tags=${angles}
            placeholder="Add angles (GitHub, private repos, etc.)"
            @tags-changed=${this.handleAnglesChanged}
          ></tag-input>
        </div>

        <div class="field">
          <div class="label-row">
            <label>Models</label>
            <button @click=${this.selectAllModels} title="Select all models">All</button>
            <button @click=${this.clearAllModels} title="Clear selection">Clear</button>
          </div>
          <div class="models">
            ${this.apiKeysError ? html`
              <div class="api-keys-error">
                <p>${this.apiKeysError.message}</p>
                <p>Add one of these to your <code>.env</code> file:</p>
                <ul>
                  ${this.apiKeysError.supported_keys.map(k => html`
                    <li>
                      <code>${k.key}</code> - <a href="${k.url}" target="_blank">${k.provider}</a>
                      ${k.note ? html` <em>(${k.note})</em>` : ''}
                    </li>
                  `)}
                </ul>
              </div>
            ` : this.availableModels.map(
              (model) => html`
                <label class="model-option">
                  <input
                    type="checkbox"
                    .checked=${selectedModels.includes(model.id)}
                    @change=${() => this.toggleModel(model.id)}
                  />
                  <span>${model.name}</span>
                  <span class="provider">(${model.provider})</span>
                </label>
              `
            )}
          </div>
        </div>

        <div class="field">
          <label>Runs per Question</label>
          <div class="slider-field">
            <input
              type="range"
              min="1"
              max="100"
              .value=${String(runsPerQuestion)}
              @input=${this.updateRuns}
              ?disabled=${isRunning}
            />
            <span class="slider-value">${runsPerQuestion}</span>
            <button
              class="infinity-toggle ${this._probeState.infiniteMode ? 'active' : ''}"
              @click=${this.toggleInfiniteMode}
              ?disabled=${isRunning}
              title="Keep running until stopped"
            >
              <span class="icon">∞</span>
            </button>
          </div>
        </div>

        <div class="field">
          <label>Questions per Batch</label>
          <div class="slider-field">
            <input
              type="range"
              min="1"
              max="20"
              .value=${String(questionCount)}
              @input=${this.updateQuestionCount}
              ?disabled=${isRunning}
            />
            <span class="slider-value">${questionCount}</span>
          </div>
        </div>

        <div class="field">
          <label>Interrogation Strategy</label>
          <div class="technique-select">
            <select
              .value=${techniquePreset}
              @change=${(e: Event) => this.setTechniquePreset((e.target as HTMLSelectElement).value as TechniquePreset)}
              ?disabled=${isRunning}
            >
              <option value="auto">Auto (Random Mix)</option>
              <optgroup label="Templates">
                ${this.availableTechniques.map(t => html`
                  <option value=${t.id}>${t.name}</option>
                `)}
              </optgroup>
            </select>
            ${techniquePreset !== 'auto' ? html`
              <span class="technique-description">
                ${this.availableTechniques.find(t => t.id === techniquePreset)?.description || ''}
              </span>
            ` : html`
              <span class="technique-description">
                Randomly picks from all techniques each round
              </span>
            `}
          </div>
        </div>

        <div class="actions">
          ${isRunning
            ? html`
                <button class="btn-danger btn-large" @click=${this.handleStop}>
                  ⏹ Stop Probing
                </button>
              `
            : html`
                <button
                  class="btn-secondary"
                  @click=${this.handleGenerate}
                  ?disabled=${!topic.trim() || this.isGenerating}
                >
                  ${this.isGenerating ? 'Generating...' : 'Preview Questions'}
                </button>
                <button
                  class="btn-primary btn-large"
                  @click=${this.handleRun}
                  ?disabled=${!canRun}
                >
                  ▶ Run Probe
                </button>
              `}
        </div>
        ${!isRunning && !canRun ? html`
          <div class="action-hint">
            ${!topic.trim() ? 'Enter a topic above to start probing' : ''}
            ${topic.trim() && selectedModels.length === 0 ? '(will auto-survey all models)' : ''}
          </div>
        ` : null}

        <div class="ground-truth-section">
          <h4>Hidden Ground Truth</h4>
          <tag-input
            .tags=${this._groundTruth}
            placeholder="Add known facts (hidden from model)"
            @tags-changed=${this.handleGroundTruthChanged}
          ></tag-input>
          <div class="hint">
            These facts are never shown to the probed model. Used for warmth detection.
          </div>
        </div>

        ${this._probeState.findings?.entities && Object.keys(this._probeState.findings.entities).length > 3
          ? html`
              <div class="cluster-section">
                <h4>Cluster Analysis</h4>
                <button
                  class="btn-secondary"
                  @click=${this.handleCluster}
                  ?disabled=${this.isClustering || isRunning}
                >
                  ${this.isClustering ? 'Clustering...' : 'Find Clusters (k-means)'}
                </button>
                ${this.optimalK !== null
                  ? html`
                      <div class="cluster-info">
                        Optimal k=${this.optimalK}, silhouette=${this.silhouetteScore?.toFixed(3)}
                      </div>
                    `
                  : null}
                ${this.clusters.length > 0
                  ? html`
                      <div class="clusters">
                        ${this.clusters.map(
                          (cluster) => html`
                            <div
                              class="cluster"
                              @click=${() => this.handleDrillCluster(cluster)}
                              title="Click to drill into this cluster"
                            >
                              <div class="cluster-header">
                                Cluster ${cluster.id + 1}
                                <span class="cluster-count">${cluster.size} entities, ${cluster.total_count} mentions</span>
                              </div>
                              <div class="cluster-entities">
                                ${cluster.entities.slice(0, 5).map(
                                  (e) => html`<span class="cluster-entity">${e.entity} (${e.count})</span>`
                                )}
                                ${cluster.entities.length > 5
                                  ? html`<span class="cluster-more">+${cluster.entities.length - 5} more</span>`
                                  : null}
                              </div>
                            </div>
                          `
                        )}
                      </div>
                    `
                  : null}
              </div>
            `
          : null}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'probe-controls': ProbeControls;
  }
}
