import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { getProjects, createProject, getFindings } from '../api';
import { navigateTo } from '../state';
import type { Findings } from '../types';
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
      gap: 12px;
      max-width: 500px;
      margin: 0 auto;
    }

    .new-form input {
      flex: 1;
      padding: 14px 18px;
      font-size: 16px;
      background: #161b22;
      border: 2px solid #30363d;
      border-radius: 12px;
      color: #c9d1d9;
    }

    .new-form input:focus {
      outline: none;
      border-color: #58a6ff;
    }

    .new-form input::placeholder {
      color: #6e7681;
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
  @state() private newName = '';
  @state() private isLoading = true;

  async connectedCallback() {
    super.connectedCallback();
    await this.loadProjects();
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
          };
        })
      );

      // Sort by total entity mentions (most data first)
      this.projects = projectsWithFindings.sort((a, b) => {
        const aTotal = Object.values(a.entities).reduce((sum, c) => sum + c, 0);
        const bTotal = Object.values(b.entities).reduce((sum, c) => sum + c, 0);
        return bTotal - aTotal;
      });
    } catch (e) {
      console.error(e);
    } finally {
      this.isLoading = false;
    }
  }

  async handleCreate() {
    if (!this.newName.trim()) return;
    const slug = this.newName.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40);
    try {
      await createProject(slug);
      navigateTo('project', slug);
    } catch (e) {
      // might exist
      navigateTo('project', slug);
    }
  }

  handleSelect(e: CustomEvent) {
    navigateTo('project', e.detail);
  }

  render() {
    return html`
      <div class="header">
        <h1>LLM Interrogator</h1>
        <p class="subtitle">Extract training data through statistical repetition</p>

        <div class="new-form">
          <input
            type="text"
            placeholder="New investigation (e.g., John Smith CEO)"
            .value=${this.newName}
            @input=${(e: Event) => this.newName = (e.target as HTMLInputElement).value}
            @keydown=${(e: KeyboardEvent) => e.key === 'Enter' && this.handleCreate()}
          />
          <button
            class="go-btn"
            @click=${this.handleCreate}
            ?disabled=${!this.newName.trim()}
          >GO</button>
        </div>
      </div>

      ${this.projects.length > 0 ? html`
        <div class="section-title">Your Investigations</div>
        <div class="grid">
          ${this.projects.map(p => html`
            <project-card
              .data=${p}
              @select=${this.handleSelect}
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
