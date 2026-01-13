import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { navigateTo, probeState, groundTruthState, type ProbeState } from '../state';
import { getProject, getFindings, startProbe, generateQuestions, updateProject } from '../api';
import type { Project, Findings, SSEEvent, GeneratedQuestion, ProbeResponse } from '../types';

@customElement('project-view')
export class ProjectView extends LitElement {
  static styles = css`
    :host {
      display: block;
      background: #0d1117;
      min-height: 100vh;
    }

    /* Top bar with controls */
    .top-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 24px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .top-left {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .back-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      background: transparent;
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #8b949e;
      cursor: pointer;
      font-size: 16px;
    }

    .back-btn:hover {
      border-color: #58a6ff;
      color: #f0f6fc;
    }

    h1 {
      font-size: 18px;
      font-weight: 600;
      color: #f0f6fc;
      margin: 0;
    }

    .topic-input {
      font-size: 18px;
      font-weight: 600;
      color: #f0f6fc;
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      padding: 4px 0;
      width: 300px;
      outline: none;
      transition: border-color 150ms;
    }

    .topic-input:hover {
      border-bottom-color: #30363d;
    }

    .topic-input:focus {
      border-bottom-color: #58a6ff;
    }

    .topic-input::placeholder {
      color: #6e7681;
    }

    .top-controls {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* Main layout: Single column, natural scroll */
    .main-layout {
      display: flex;
      flex-direction: column;
      height: calc(100vh - 65px);
      overflow: hidden;
    }

    resizable-panel {
      flex: 1;
      overflow: hidden;
    }

    /* Top: Interactive findings/word cloud */
    .findings-top {
      padding: 8px;
      background: #161b22;
      height: 100%;
      box-sizing: border-box;
      overflow: hidden;
    }

    .findings-top findings-panel {
      height: 100%;
      display: block;
    }

    /* Below: Two columns - questions queue + response stream */
    .interrogation-area {
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 24px;
      padding: 24px;
      align-items: start;
      height: 100%;
      overflow: auto;
    }

    @media (max-width: 900px) {
      .interrogation-area {
        grid-template-columns: 1fr;
      }
    }

    .questions-column {
      position: sticky;
      top: 0;
    }

    .responses-column {
      /* Natural flow, no constraints */
    }

    /* Config drawer */
    .config-drawer {
      position: fixed;
      top: 65px;
      left: 0;
      width: 350px;
      height: calc(100vh - 65px);
      background: #161b22;
      border-right: 1px solid #30363d;
      transform: translateX(-100%);
      transition: transform 200ms ease;
      z-index: 40;
      overflow-y: auto;
      padding: 16px;
    }

    .config-drawer.open {
      transform: translateX(0);
    }

    .config-btn {
      padding: 8px 16px;
      background: #238636;
      border: none;
      border-radius: 6px;
      color: white;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }

    .config-btn:hover {
      background: #2ea043;
    }

    .config-btn.secondary {
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
    }

    /* Android Studio style run controls */
    .run-controls {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      background: #21262d;
      border-radius: 6px;
    }

    .run-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      background: transparent;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 150ms;
    }

    .run-btn:hover {
      background: #30363d;
    }

    .run-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .run-btn.play {
      color: #3fb950;
    }

    .run-btn.play:hover:not(:disabled) {
      background: rgba(63, 185, 80, 0.2);
    }

    .run-btn.stop {
      color: #f85149;
    }

    .run-btn.stop:hover:not(:disabled) {
      background: rgba(248, 81, 73, 0.2);
    }

    .run-btn.preview {
      color: #58a6ff;
    }

    .run-btn.preview:hover:not(:disabled) {
      background: rgba(88, 166, 255, 0.2);
    }

    .run-btn svg {
      width: 16px;
      height: 16px;
      fill: currentColor;
    }

    .divider {
      width: 1px;
      height: 20px;
      background: #30363d;
      margin: 0 4px;
    }

    .run-counter {
      font-size: 13px;
      font-weight: 600;
      color: #6e7681;
      font-family: monospace;
      min-width: 50px;
      text-align: center;
      padding: 4px 8px;
      border-radius: 4px;
      transition: all 150ms;
    }

    .run-counter:hover {
      background: #30363d;
    }

    .run-counter.active {
      color: #3fb950;
      animation: pulse-counter 1s ease-in-out infinite;
    }

    .run-counter.infinite {
      color: #58a6ff;
    }

    @keyframes pulse-counter {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }

    /* Recycle bin */
    .recycle-bin {
      position: relative;
    }

    .bin-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      background: transparent;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      color: #6e7681;
      font-size: 16px;
      transition: all 150ms;
    }

    .bin-btn:hover {
      background: #30363d;
      color: #f85149;
    }

    .bin-btn.has-items {
      color: #f0883e;
    }

    .bin-count {
      position: absolute;
      top: -4px;
      right: -4px;
      min-width: 16px;
      height: 16px;
      padding: 0 4px;
      background: #f85149;
      border-radius: 8px;
      font-size: 10px;
      font-weight: 600;
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .bin-dropdown {
      position: absolute;
      top: 100%;
      right: 0;
      margin-top: 8px;
      width: 280px;
      max-height: 300px;
      overflow-y: auto;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      z-index: 100;
    }

    .bin-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border-bottom: 1px solid #21262d;
      font-size: 12px;
      color: #8b949e;
    }

    .bin-empty-btn {
      background: none;
      border: none;
      color: #f85149;
      font-size: 11px;
      cursor: pointer;
    }

    .bin-empty-btn:hover {
      text-decoration: underline;
    }

    .bin-items {
      padding: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .bin-item {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      font-size: 12px;
      color: #8b949e;
      cursor: pointer;
      user-select: none;
      transition: all 150ms;
    }

    .bin-item:hover {
      background: #30363d;
      color: #c9d1d9;
      border-color: #58a6ff;
    }

    .bin-item:active {
      background: rgba(248, 81, 73, 0.2);
      border-color: #f85149;
    }

    .bin-hint {
      padding: 8px 12px;
      font-size: 10px;
      color: #484f58;
      text-align: center;
      border-top: 1px solid #21262d;
    }

    .loading {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
      color: #8b949e;
    }
  `;

  @property({ type: String })
  projectName: string | null = null;

  @state()
  private project: Project | null = null;

  @state()
  private findings: Findings | null = null;

  @state()
  private isLoading = true;

  @state()
  private _probeState: ProbeState = probeState.get();

  private _unsubscribe?: () => void;

  async connectedCallback() {
    super.connectedCallback();
    this._unsubscribe = probeState.subscribe((s) => {
      this._probeState = s;
    });
    document.addEventListener('click', this.handleClickOutside);
    await this.loadProject();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribe?.();
    document.removeEventListener('click', this.handleClickOutside);
  }

  async updated(changed: Map<string, unknown>) {
    if (changed.has('projectName') && this.projectName) {
      await this.loadProject();
    }
  }

  async loadProject() {
    if (!this.projectName) return;

    this.isLoading = true;
    try {
      this.project = await getProject(this.projectName);

      // Load saved state into probe state
      // Default topic from project name if not set
      const defaultTopic = this.projectName
        .replace(/-/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase());

      probeState.update((s) => ({
        ...s,
        topic: this.project!.topic || defaultTopic
      }));

      if (this.project.angles) {
        probeState.update((s) => ({ ...s, angles: this.project!.angles || [] }));
      }
      if (this.project.ground_truth) {
        groundTruthState.update((s) => ({
          ...s,
          facts: this.project!.ground_truth || [],
        }));
      }

      // Load findings
      try {
        this.findings = await getFindings(this.projectName);
        probeState.update((s) => ({ ...s, findings: this.findings }));
      } catch {
        // No findings yet, that's fine
      }
    } catch (err) {
      console.error('Failed to load project:', err);
    } finally {
      this.isLoading = false;
    }
  }

  @state()
  private showConfig = false;

  @state()
  private showBin = false;

  @state()
  private binPressStart = 0;

  private _abortController: AbortController | null = null;

  private async handlePreview() {
    const state = probeState.get();
    if (!state.topic) return;

    probeState.update(s => ({ ...s, isGenerating: true }));
    try {
      const result = await generateQuestions(
        state.topic,
        state.angles,
        state.questionCount,
        state.techniquePreset,
        Object.keys(state.findings?.entities || {}).slice(0, 20)
      );
      probeState.update(s => ({ ...s, questions: result.questions as GeneratedQuestion[], isGenerating: false }));
    } catch (err) {
      console.error('Failed to generate questions:', err);
      probeState.update(s => ({ ...s, isGenerating: false }));
    }
  }

  private async handleRun() {
    const state = probeState.get();
    if (!this.projectName || !state.topic) return;

    // Save config
    await updateProject(this.projectName, {
      topic: state.topic,
      angles: state.angles,
      ground_truth: groundTruthState.get().facts,
    });

    probeState.update(s => ({ ...s, isRunning: true, responses: [] }));

    this._abortController = startProbe(
      {
        project: this.projectName,
        topic: state.topic,
        angles: state.angles,
        models: state.selectedModels,
        runs_per_question: state.runsPerQuestion,
        questions_count: state.questionCount,
        technique_preset: state.techniquePreset,
        questions: state.questions.length > 0 ? state.questions.map(q => q.question) : Array(state.questionCount).fill(null),
        negative_entities: state.hiddenEntities,    // Entities to AVOID
        positive_entities: state.promotedEntities,  // Entities to FOCUS ON
      },
      (event: SSEEvent) => {
        if (event.type === 'questions') {
          probeState.update(s => ({ ...s, questions: event.data as GeneratedQuestion[] }));
        } else if (event.type === 'response') {
          probeState.update(s => ({ ...s, responses: [...s.responses, event.data as ProbeResponse] }));
        } else if (event.type === 'findings_update') {
          probeState.update(s => ({ ...s, findings: event.data as Findings }));
          this.findings = event.data as Findings;
        } else if (event.type === 'complete') {
          probeState.update(s => ({ ...s, isRunning: false }));
          this._abortController = null;
        }
      },
      (error) => {
        console.error('Probe error:', error);
        probeState.update(s => ({ ...s, isRunning: false }));
        this._abortController = null;
      }
    );
  }

  private handleStop() {
    if (this._abortController) {
      this._abortController.abort();
      this._abortController = null;
      probeState.update(s => ({ ...s, isRunning: false }));
    }
  }

  private handleBinItemMouseDown(entity: string) {
    this.binPressStart = Date.now();
  }

  private handleBinItemMouseUp(entity: string) {
    const duration = Date.now() - this.binPressStart;
    if (duration < 300) {
      // Tap - restore
      probeState.update(s => ({
        ...s,
        hiddenEntities: s.hiddenEntities.filter(e => e !== entity)
      }));
    } else {
      // Hold - permanently delete (just remove from hidden, it's gone)
      probeState.update(s => ({
        ...s,
        hiddenEntities: s.hiddenEntities.filter(e => e !== entity)
      }));
    }
    this.binPressStart = 0;
  }

  private emptyBin() {
    probeState.update(s => ({ ...s, hiddenEntities: [] }));
    this.showBin = false;
  }

  private handleClickOutside = (e: MouseEvent) => {
    if (this.showBin) {
      const bin = this.shadowRoot?.querySelector('.recycle-bin');
      if (bin && !bin.contains(e.target as Node)) {
        this.showBin = false;
      }
    }
  };

  render() {
    if (this.isLoading) {
      return html`<div class="loading">Loading...</div>`;
    }

    if (!this.project) {
      return html`<div class="loading">Project not found</div>`;
    }

    const isRunning = this._probeState.isRunning;

    return html`
      <!-- Top bar -->
      <div class="top-bar">
        <div class="top-left">
          <button class="back-btn" @click=${() => navigateTo('projects')}>←</button>
          <input
            class="topic-input"
            type="text"
            .value=${this._probeState.topic || this.project.name}
            placeholder="Investigation topic..."
            @input=${(e: Event) => {
              const value = (e.target as HTMLInputElement).value;
              probeState.update(s => ({ ...s, topic: value }));
            }}
            @blur=${() => {
              // Auto-save topic on blur
              if (this.projectName) {
                updateProject(this.projectName, { topic: this._probeState.topic });
              }
            }}
          />
        </div>
        <div class="top-controls">
          <!-- Android Studio style run controls -->
          <div class="run-controls">
            <button
              class="run-btn preview"
              title="Preview Questions"
              ?disabled=${isRunning || this._probeState.isGenerating}
              @click=${this.handlePreview}
            >
              <svg viewBox="0 0 16 16"><path d="M8 3a5 5 0 100 10A5 5 0 008 3zM1 8a7 7 0 1114 0A7 7 0 011 8zm8-1V5H7v2H5v2h2v2h2V9h2V7H9z"/></svg>
            </button>
            <div class="divider"></div>
            <button
              class="run-btn play"
              title="Run Probe"
              ?disabled=${isRunning}
              @click=${this.handleRun}
            >
              <svg viewBox="0 0 16 16"><path d="M4 2l10 6-10 6V2z"/></svg>
            </button>
            <span
              class="run-counter ${isRunning ? 'active' : ''} ${this._probeState.infiniteMode ? 'infinite' : ''}"
              title="Click to toggle infinite mode"
              @click=${() => probeState.update(s => ({ ...s, infiniteMode: !s.infiniteMode }))}
              style="cursor: pointer;"
            >
              ${this._probeState.responses.length}:${this._probeState.infiniteMode ? '∞' : this._probeState.runsPerQuestion * this._probeState.questionCount * this._probeState.selectedModels.length}
            </span>
            <button
              class="run-btn stop"
              title="Stop"
              ?disabled=${!isRunning}
              @click=${this.handleStop}
            >
              <svg viewBox="0 0 16 16"><rect x="3" y="3" width="10" height="10" rx="1"/></svg>
            </button>
            <div class="divider"></div>
            <!-- Recycle bin -->
            <div class="recycle-bin">
              <button
                class="bin-btn ${this._probeState.hiddenEntities.length > 0 ? 'has-items' : ''}"
                title="Removed entities"
                @click=${() => this.showBin = !this.showBin}
              >
                🗑️
                ${this._probeState.hiddenEntities.length > 0 ? html`
                  <span class="bin-count">${this._probeState.hiddenEntities.length}</span>
                ` : null}
              </button>
              ${this.showBin ? html`
                <div class="bin-dropdown">
                  <div class="bin-header">
                    <span>Removed (${this._probeState.hiddenEntities.length})</span>
                    ${this._probeState.hiddenEntities.length > 0 ? html`
                      <button class="bin-empty-btn" @click=${this.emptyBin}>Empty all</button>
                    ` : null}
                  </div>
                  <div class="bin-items">
                    ${this._probeState.hiddenEntities.length === 0 ? html`
                      <span style="color: #484f58; font-size: 12px; padding: 12px;">No removed entities</span>
                    ` : this._probeState.hiddenEntities.map(entity => html`
                      <span
                        class="bin-item"
                        @mousedown=${() => this.handleBinItemMouseDown(entity)}
                        @mouseup=${() => this.handleBinItemMouseUp(entity)}
                        @mouseleave=${() => this.binPressStart = 0}
                      >
                        ${entity}
                      </span>
                    `)}
                  </div>
                  <div class="bin-hint">tap = restore · hold = delete permanently</div>
                </div>
              ` : null}
            </div>
          </div>
          <button
            class="config-btn secondary"
            @click=${() => this.showConfig = !this.showConfig}
          >
            ⚙️ Config
          </button>
        </div>
      </div>

      <!-- Main layout: resizable panels -->
      <div class="main-layout">
        <resizable-panel .initialHeight=${280} .minHeight=${150} .maxHeight=${600}>
          <!-- TOP: Interactive word cloud -->
          <div slot="top" class="findings-top">
            <findings-panel .findings=${this.findings}></findings-panel>
          </div>

          <!-- BOTTOM: Questions + Responses side by side -->
          <div slot="bottom" class="interrogation-area">
            <div class="questions-column">
              <question-queue></question-queue>
            </div>
            <div class="responses-column">
              <response-stream .projectName=${this.projectName} @stop-probe=${this.handleStop}></response-stream>
            </div>
          </div>
        </resizable-panel>
      </div>

      <!-- Config drawer (slides from left) -->
      <div class="config-drawer ${this.showConfig ? 'open' : ''}">
        <probe-controls .projectName=${this.projectName}></probe-controls>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'project-view': ProjectView;
  }
}
