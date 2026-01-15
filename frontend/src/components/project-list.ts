import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { getProjects, createProject, getFindings, startProbe } from '../api';
import type { Project, Findings, SSEEvent } from '../types';
import './word-cloud';

const MODELS = [
  { id: 'groq/llama-3.1-8b-instant', name: 'Llama 3.1 8B', provider: 'Groq' },
  { id: 'deepseek/deepseek-chat', name: 'DeepSeek Chat', provider: 'DeepSeek' },
];

@customElement('project-list')
export class ProjectList extends LitElement {
  static styles = css`
    :host {
      display: block;
      min-height: 100vh;
      background: #0d1117;
      color: #c9d1d9;
    }

    .app {
      display: grid;
      grid-template-columns: 1fr 400px;
      min-height: 100vh;
    }

    /* Left side - main action area */
    .main {
      padding: 32px;
      overflow-y: auto;
    }

    h1 {
      font-size: 32px;
      font-weight: 700;
      margin: 0 0 8px 0;
      background: linear-gradient(135deg, #58a6ff, #a371f7);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .subtitle {
      color: #8b949e;
      margin-bottom: 32px;
    }

    /* Controls */
    .controls {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }

    .field {
      margin-bottom: 20px;
    }

    .field:last-child {
      margin-bottom: 0;
    }

    label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: #8b949e;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    input[type="text"], textarea {
      width: 100%;
      padding: 12px 16px;
      font-size: 15px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      color: #c9d1d9;
      font-family: inherit;
    }

    input:focus, textarea:focus {
      outline: none;
      border-color: #58a6ff;
    }

    textarea {
      min-height: 80px;
      resize: vertical;
    }

    .row {
      display: flex;
      gap: 16px;
    }

    .row .field {
      flex: 1;
    }

    /* Models */
    .models {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    .model-btn {
      padding: 10px 16px;
      background: #21262d;
      border: 2px solid #30363d;
      border-radius: 8px;
      color: #8b949e;
      font-size: 13px;
      cursor: pointer;
      transition: all 150ms;
    }

    .model-btn:hover {
      border-color: #58a6ff;
    }

    .model-btn.selected {
      background: rgba(88, 166, 255, 0.15);
      border-color: #58a6ff;
      color: #58a6ff;
    }

    /* Sliders */
    .slider-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    input[type="range"] {
      flex: 1;
      height: 6px;
      -webkit-appearance: none;
      background: #30363d;
      border-radius: 3px;
    }

    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      background: #58a6ff;
      border-radius: 50%;
      cursor: pointer;
    }

    .slider-value {
      min-width: 40px;
      text-align: right;
      font-weight: 600;
      color: #c9d1d9;
    }

    /* Big run button */
    .run-btn {
      width: 100%;
      padding: 16px;
      font-size: 18px;
      font-weight: 700;
      background: linear-gradient(135deg, #3fb950, #2ea043);
      border: none;
      border-radius: 10px;
      color: white;
      cursor: pointer;
      transition: all 200ms;
      margin-top: 8px;
    }

    .run-btn:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(63, 185, 80, 0.3);
    }

    .run-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .run-btn.running {
      background: linear-gradient(135deg, #f85149, #da3633);
    }

    /* Results area */
    .results {
      margin-top: 24px;
    }

    .results h2 {
      font-size: 18px;
      margin: 0 0 16px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .results h2::before {
      content: '';
      width: 10px;
      height: 10px;
      background: #3fb950;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    .response-feed {
      max-height: 400px;
      overflow-y: auto;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
    }

    .response-item {
      padding: 16px;
      border-bottom: 1px solid #21262d;
    }

    .response-item:last-child {
      border-bottom: none;
    }

    .response-meta {
      display: flex;
      gap: 12px;
      margin-bottom: 8px;
      font-size: 12px;
      color: #8b949e;
    }

    .response-model {
      color: #58a6ff;
      font-weight: 600;
    }

    .response-text {
      font-size: 14px;
      line-height: 1.5;
      color: #c9d1d9;
    }

    .response-entities {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }

    .entity-tag {
      padding: 3px 8px;
      background: #21262d;
      border-radius: 12px;
      font-size: 11px;
      color: #8b949e;
    }

    .entity-tag.hot {
      background: rgba(255, 107, 107, 0.2);
      color: #ff6b6b;
    }

    /* Right side - findings panel */
    .sidebar {
      background: #161b22;
      border-left: 1px solid #30363d;
      padding: 24px;
      overflow-y: auto;
    }

    .sidebar h2 {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #8b949e;
      margin: 0 0 16px 0;
    }

    .cloud-box {
      margin-bottom: 24px;
      border-radius: 12px;
      overflow: hidden;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 24px;
    }

    .stat-box {
      background: #0d1117;
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }

    .stat-num {
      font-size: 28px;
      font-weight: 700;
      color: #c9d1d9;
    }

    .stat-label {
      font-size: 11px;
      color: #6e7681;
      text-transform: uppercase;
    }

    .signals-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .signal-row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      background: #0d1117;
      border-radius: 8px;
    }

    .signal-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .signal-dot.hot { background: #ff6b6b; box-shadow: 0 0 8px #ff6b6b; }
    .signal-dot.warm { background: #ffd43b; }
    .signal-dot.cool { background: #69db7c; }

    .signal-name {
      flex: 1;
      font-size: 13px;
    }

    .signal-count {
      font-size: 12px;
      font-weight: 600;
      color: #6e7681;
    }

    .empty-msg {
      color: #6e7681;
      font-size: 13px;
      text-align: center;
      padding: 40px 20px;
    }
  `;

  @state() private topic = '';
  @state() private angles = '';
  @state() private selectedModels: string[] = [];  // Empty = auto-survey all
  @state() private runs = 10;
  @state() private questions = 3;
  @state() private isRunning = false;
  @state() private responses: Array<{model: string; text: string; entities: string[]}> = [];
  @state() private findings: Findings | null = null;
  @state() private projectName = 'default';
  @state() private abortController: AbortController | null = null;

  async connectedCallback() {
    super.connectedCallback();
    // Try to load existing findings
    try {
      const projects = await getProjects();
      if (projects.length > 0) {
        const active = projects.sort((a, b) => (b.corpus_count || 0) - (a.corpus_count || 0))[0];
        this.projectName = active.name;
        this.topic = active.topic || '';
        this.findings = await getFindings(active.name);
      }
    } catch (e) {
      console.error(e);
    }
  }

  toggleModel(id: string) {
    if (this.selectedModels.includes(id)) {
      this.selectedModels = this.selectedModels.filter(m => m !== id);
    } else {
      this.selectedModels = [...this.selectedModels, id];
    }
  }

  async runProbe() {
    if (!this.topic.trim()) return;
    // Empty models = auto-survey all available models

    // Create project if needed
    if (this.projectName === 'default') {
      const slug = this.topic.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 30).replace(/-+$/, '');
      try {
        await createProject(slug);
        this.projectName = slug;
      } catch (e) {
        // Project might exist
        this.projectName = slug;
      }
    }

    this.isRunning = true;
    this.responses = [];

    const controller = startProbe(
      {
        project: this.projectName,
        topic: this.topic,
        angles: this.angles.split(',').map(s => s.trim()).filter(Boolean),
        models: this.selectedModels,
        questions: Array(this.questions).fill(null),
        runs_per_question: this.runs,
        questions_count: this.questions,
        technique_preset: 'balanced',
      },
      (event: SSEEvent) => this.handleEvent(event),
      (err) => {
        console.error(err);
        this.isRunning = false;
      }
    );

    this.abortController = controller;
  }

  handleEvent(event: SSEEvent) {
    if (event.type === 'response') {
      const data = event.data as any;
      this.responses = [...this.responses, {
        model: data.model,
        text: data.response,
        entities: data.entities || [],
      }];
    } else if (event.type === 'findings_update') {
      this.findings = event.data as Findings;
    } else if (event.type === 'complete') {
      this.isRunning = false;
      this.abortController = null;
    }
  }

  stopProbe() {
    if (this.abortController) {
      this.abortController.abort();
      this.isRunning = false;
      this.abortController = null;
    }
  }

  getSignalLevel(count: number): string {
    if (count >= 15) return 'hot';
    if (count >= 8) return 'warm';
    return 'cool';
  }

  getTopSignals() {
    if (!this.findings?.entities) return [];
    const skip = ['Based', 'Here', 'However', 'Therefore', 'For', 'The', 'His', 'Public', 'Some', 'Without'];
    return Object.entries(this.findings.entities)
      .filter(([name]) => !skip.includes(name))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12);
  }

  render() {
    const signals = this.getTopSignals();
    const entityCount = this.findings?.entities ? Object.keys(this.findings.entities).length : 0;
    const signalCount = signals.filter(([,c]) => c >= 5).length;

    return html`
      <div class="app">
        <div class="main">
          <h1>LLM Interrogator</h1>
          <p class="subtitle">Extract training data through statistical repetition</p>

          <div class="controls">
            <div class="field">
              <label>Target / Topic</label>
              <textarea
                placeholder="Who or what to investigate (e.g., 'Enron executive compensation 2001')"
                .value=${this.topic}
                @input=${(e: Event) => this.topic = (e.target as HTMLTextAreaElement).value}
              ></textarea>
            </div>

            <div class="field">
              <label>Angles (optional, comma-separated)</label>
              <input
                type="text"
                placeholder="e.g., Garmin, BMW, startups, GitHub"
                .value=${this.angles}
                @input=${(e: Event) => this.angles = (e.target as HTMLInputElement).value}
              />
            </div>

            <div class="field">
              <label>Models</label>
              <div class="models">
                ${MODELS.map(m => html`
                  <button
                    class="model-btn ${this.selectedModels.includes(m.id) ? 'selected' : ''}"
                    @click=${() => this.toggleModel(m.id)}
                  >
                    ${m.name}
                  </button>
                `)}
              </div>
            </div>

            <div class="row">
              <div class="field">
                <label>Runs per question</label>
                <div class="slider-row">
                  <input
                    type="range"
                    min="1"
                    max="50"
                    .value=${String(this.runs)}
                    @input=${(e: Event) => this.runs = parseInt((e.target as HTMLInputElement).value)}
                  />
                  <span class="slider-value">${this.runs}</span>
                </div>
              </div>
              <div class="field">
                <label>Questions</label>
                <div class="slider-row">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    .value=${String(this.questions)}
                    @input=${(e: Event) => this.questions = parseInt((e.target as HTMLInputElement).value)}
                  />
                  <span class="slider-value">${this.questions}</span>
                </div>
              </div>
            </div>

            <button
              class="run-btn ${this.isRunning ? 'running' : ''}"
              @click=${this.isRunning ? () => this.stopProbe() : () => this.runProbe()}
              ?disabled=${!this.topic.trim()}
            >
              ${this.isRunning ? 'STOP' : 'RUN INTERROGATION'}
            </button>
          </div>

          ${this.responses.length > 0 ? html`
            <div class="results">
              <h2>Live Responses (${this.responses.length})</h2>
              <div class="response-feed">
                ${this.responses.slice().reverse().slice(0, 20).map(r => html`
                  <div class="response-item">
                    <div class="response-meta">
                      <span class="response-model">${r.model.split('/').pop()}</span>
                    </div>
                    <div class="response-text">${r.text.slice(0, 300)}${r.text.length > 300 ? '...' : ''}</div>
                    ${r.entities.length > 0 ? html`
                      <div class="response-entities">
                        ${r.entities.slice(0, 8).map(e => html`
                          <span class="entity-tag">${e}</span>
                        `)}
                      </div>
                    ` : null}
                  </div>
                `)}
              </div>
            </div>
          ` : null}
        </div>

        <div class="sidebar">
          <h2>Findings</h2>

          ${this.findings?.entities && Object.keys(this.findings.entities).length > 0 ? html`
            <div class="cloud-box">
              <word-cloud .entities=${this.findings.entities} .signalThreshold=${5}></word-cloud>
            </div>
          ` : null}

          <div class="stats-grid">
            <div class="stat-box">
              <div class="stat-num">${this.findings?.corpus_size || 0}</div>
              <div class="stat-label">Responses</div>
            </div>
            <div class="stat-box">
              <div class="stat-num">${entityCount}</div>
              <div class="stat-label">Entities</div>
            </div>
            <div class="stat-box">
              <div class="stat-num">${signalCount}</div>
              <div class="stat-label">Signals</div>
            </div>
            <div class="stat-box">
              <div class="stat-num">${this.findings?.refusal_rate ? Math.round(this.findings.refusal_rate * 100) : 0}%</div>
              <div class="stat-label">Refusals</div>
            </div>
          </div>

          <h2>Top Signals</h2>
          ${signals.length > 0 ? html`
            <div class="signals-list">
              ${signals.map(([name, count]) => html`
                <div class="signal-row">
                  <span class="signal-dot ${this.getSignalLevel(count)}"></span>
                  <span class="signal-name">${name}</span>
                  <span class="signal-count">${count}x</span>
                </div>
              `)}
            </div>
          ` : html`
            <div class="empty-msg">Run an interrogation to find signals</div>
          `}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'project-list': ProjectList;
  }
}
