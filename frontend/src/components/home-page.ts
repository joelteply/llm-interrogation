import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { getProjects, createProject, getFindings, deleteProject, getAvailableModels, getInvestigationConfig, type InvestigationConfig } from '../api';
import { navigateTo } from '../state';
import type { Findings, ModelInfo } from '../types';
import './project-card';
import type { ProjectCardData } from './project-card';

@customElement('home-page')
export class HomePage extends LitElement {
  static styles = css`
    :host {
      display: block;
      min-height: 100vh;
      background: #0d1117;
      color: #c9d1d9;
      padding: 32px;
    }

    .header {
      max-width: 600px;
      margin: 0 auto 40px auto;
      text-align: center;
    }

    h1 {
      font-size: 36px;
      font-weight: 800;
      margin: 0 0 8px 0;
      background: linear-gradient(135deg, #58a6ff, #a371f7);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .subtitle {
      color: #8b949e;
      margin-bottom: 24px;
    }

    /* New investigation form */
    .new-form {
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-width: 600px;
      margin: 0 auto;
    }

    .new-form textarea {
      width: 100%;
      min-height: 100px;
      padding: 16px 18px;
      font-size: 16px;
      font-family: inherit;
      background: #161b22;
      border: 2px solid #30363d;
      border-radius: 12px;
      color: #c9d1d9;
      resize: vertical;
    }

    .new-form textarea:focus {
      outline: none;
      border-color: #58a6ff;
    }

    .new-form textarea::placeholder {
      color: #6e7681;
    }

    .source-section {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .source-label {
      font-size: 12px;
      color: #8b949e;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .source-input-wrapper {
      display: flex;
      gap: 8px;
    }

    .source-input {
      flex: 1;
      padding: 12px 14px;
      font-size: 14px;
      font-family: inherit;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      color: #c9d1d9;
    }

    .source-input:focus {
      outline: none;
      border-color: #d4a574;
    }

    .source-input::placeholder {
      color: #6e7681;
    }

    .browse-btn {
      padding: 12px 16px;
      font-size: 12px;
      font-weight: 600;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 8px;
      color: #8b949e;
      cursor: pointer;
      white-space: nowrap;
    }

    .browse-btn:hover {
      background: #30363d;
      color: #c9d1d9;
    }

    .source-hint {
      font-size: 11px;
      color: #6e7681;
    }

    .goal-indicator {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: rgba(88, 166, 255, 0.1);
      border: 1px solid rgba(88, 166, 255, 0.3);
      border-radius: 12px;
      font-size: 11px;
      color: #58a6ff;
      margin-top: 8px;
    }

    .goal-indicator.find_leaks {
      background: rgba(248, 81, 73, 0.1);
      border-color: rgba(248, 81, 73, 0.3);
      color: #f85149;
    }

    .goal-indicator.competitive_intel {
      background: rgba(163, 113, 247, 0.1);
      border-color: rgba(163, 113, 247, 0.3);
      color: #a371f7;
    }

    .goal-indicator.find_connections {
      background: rgba(63, 185, 80, 0.1);
      border-color: rgba(63, 185, 80, 0.3);
      color: #3fb950;
    }

    .go-btn {
      padding: 14px 28px;
      font-size: 16px;
      font-weight: 700;
      background: linear-gradient(135deg, #3fb950, #2ea043);
      border: none;
      border-radius: 12px;
      color: white;
      cursor: pointer;
      transition: all 200ms;
    }

    .go-btn:hover:not(:disabled) {
      transform: scale(1.05);
    }

    .go-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* Project grid */
    .section-title {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #6e7681;
      margin: 48px 0 20px 0;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
      gap: 32px;
      max-width: 1400px;
      margin: 0 auto;
    }

    .empty {
      text-align: center;
      padding: 60px 20px;
      color: #6e7681;
    }

    .empty h2 {
      color: #c9d1d9;
      margin-bottom: 8px;
    }
  `;

  @state() private projects: ProjectCardData[] = [];
  @state() private newQuery = '';
  @state() private seedValue = '';
  @state() private isLoading = true;
  @state() private allModels: ModelInfo[] = [];
  @state() private config: InvestigationConfig | null = null;

  private detectGoal(query: string): string {
    const q = query.toLowerCase();

    // Use config-based goal detection if available
    if (this.config?.goals) {
      for (const [goalId, goalConfig] of Object.entries(this.config.goals)) {
        if (goalConfig.keywords?.some(kw => q.includes(kw.toLowerCase()))) {
          return goalId;
        }
      }
    }

    // Fallback to hardcoded detection
    if (['leak', 'training data', 'in the model', 'our code', 'our data', 'private'].some(w => q.includes(w))) {
      return 'find_leaks';
    } else if (['competitor', 'opposition', 'rival', 'they know', 'using our'].some(w => q.includes(w))) {
      return 'competitive_intel';
    } else if (['connection', 'between', 'relationship', 'linked'].some(w => q.includes(w))) {
      return 'find_connections';
    }
    return 'research';
  }

  private getGoalLabel(goal: string): string {
    // Use config-based labels if available
    if (this.config?.goals?.[goal]) {
      const g = this.config.goals[goal];
      return `${g.icon} ${g.label}`;
    }

    // Fallback
    switch (goal) {
      case 'find_leaks': return '🔍 Leak Detection';
      case 'competitive_intel': return '🕵️ Competitive Intel';
      case 'find_connections': return '🔗 Find Connections';
      default: return '📚 Research';
    }
  }

  async connectedCallback() {
    super.connectedCallback();
    await Promise.all([this.loadProjects(), this.loadModels(), this.loadConfig()]);
  }

  async loadConfig() {
    try {
      this.config = await getInvestigationConfig();
    } catch (e) {
      console.error('Failed to load config:', e);
    }
  }

  async loadModels() {
    try {
      this.allModels = await getAvailableModels();
    } catch (e) {
      console.error('Failed to load models:', e);
    }
  }

  private getPlaceholderText(): string {
    const examples = this.config?.examples || [
      "Is our vHSM code in GPT's training data?",
      "Find connections between Company X and Person Y",
      "What does competitor Z know about our API?",
      "Case law related to Smith v. Jones 2023"
    ];
    return `What are you investigating?\n\nExamples:\n${examples.map(e => `• ${e}`).join('\n')}`;
  }

  private getSourcePlaceholder(): string {
    return this.config?.source_hints?.placeholder || '/path/to/repo, https://..., or paste content';
  }

  private getSourceDescription(): string {
    return this.config?.source_hints?.description || 'Point to private code, documents, or data you want to probe for in LLM training data';
  }

  async loadProjects() {
    this.isLoading = true;
    try {
      const rawProjects = await getProjects();

      // Load findings for EVERY project - always try
      const projectsWithFindings = await Promise.all(
        rawProjects.map(async (p) => {
          let entities: Record<string, number> = {};
          let corpusCount = 0;
          try {
            const findings = await getFindings(p.name);
            entities = findings.entities || {};
            corpusCount = findings.corpus_size || Object.values(entities).reduce((a, b) => a + b, 0);
          } catch (e) {
            // Project might not have findings yet
          }
          return {
            name: p.name,
            topic: p.topic || p.name.replace(/-/g, ' '),
            corpus_count: corpusCount,
            entities,
            narrative: p.narrative,
            questions: p.questions,
          };
        })
      );

      // Keep backend's sort order (most recent first)
      this.projects = projectsWithFindings;
    } catch (e) {
      console.error(e);
    } finally {
      this.isLoading = false;
    }
  }

  async handleCreate() {
    if (!this.newQuery.trim()) return;
    const query = this.newQuery.trim();
    const allModelIds = this.allModels.map(m => m.id);

    try {
      // Use smart creation API
      const response = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          seed_value: this.seedValue.trim(),
          seed_type: 'auto',
          selected_models: allModelIds
        })
      });

      if (!response.ok) {
        const err = await response.json();
        if (err.error === 'Project already exists') {
          // Navigate to existing project
          const slug = query.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 60).replace(/-+$/, '');
          navigateTo('project', slug, true);
          return;
        }
        throw new Error(err.error);
      }

      const project = await response.json();
      navigateTo('project', project.name, true);
    } catch (e) {
      console.error('Failed to create project:', e);
    }
  }

  handleSelect(e: CustomEvent) {
    navigateTo('project', e.detail, true);  // autostart on resume
  }

  async handleDelete(e: CustomEvent) {
    const name = e.detail;
    if (!confirm(`Delete "${name}"? This cannot be undone.`)) return;
    try {
      await deleteProject(name);
      this.projects = this.projects.filter(p => p.name !== name);
    } catch (err) {
      console.error('Failed to delete:', err);
    }
  }

  render() {
    return html`
      <div class="header">
        <h1>LLM Interrogator</h1>
        <p class="subtitle">Extract training data through statistical repetition</p>

        <div class="new-form">
          <textarea
            .placeholder=${this.getPlaceholderText()}
            .value=${this.newQuery}
            @input=${(e: Event) => this.newQuery = (e.target as HTMLTextAreaElement).value}
            @keydown=${(e: KeyboardEvent) => e.key === 'Enter' && e.metaKey && this.handleCreate()}
          ></textarea>

          ${this.newQuery.trim() ? html`
            <span class="goal-indicator ${this.detectGoal(this.newQuery)}">
              ${this.getGoalLabel(this.detectGoal(this.newQuery))}
            </span>
          ` : ''}

          <div class="source-section">
            <label class="source-label">
              📎 Source material <span style="color: #6e7681">(optional)</span>
            </label>
            <div class="source-input-wrapper">
              <input
                type="text"
                class="source-input"
                .placeholder=${this.getSourcePlaceholder()}
                .value=${this.seedValue}
                @input=${(e: Event) => this.seedValue = (e.target as HTMLInputElement).value}
              />
            </div>
            <div class="source-hint">
              ${this.getSourceDescription()}
            </div>
          </div>

          <button
            class="go-btn"
            @click=${this.handleCreate}
            ?disabled=${!this.newQuery.trim()}
          >Start Investigation (${this.allModels.length} models)</button>
        </div>
      </div>

      ${this.projects.length > 0 ? html`
        <div class="section-title">Your Investigations</div>
        <div class="grid">
          ${this.projects.map(p => html`
            <project-card
              .data=${p}
              @select=${this.handleSelect}
              @delete=${this.handleDelete}
            ></project-card>
          `)}
        </div>
      ` : !this.isLoading ? html`
        <div class="empty">
          <h2>No investigations yet</h2>
          <p>Enter a name or topic above to start your first investigation</p>
        </div>
      ` : null}
    `;
  }
}
