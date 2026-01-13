import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import './word-cloud';

export interface ProjectCardData {
  name: string;
  topic?: string;
  corpus_count: number;
  entities: Record<string, number>;
}

@customElement('project-card')
export class ProjectCard extends LitElement {
  static styles = css`
    :host {
      display: block;
    }

    .card {
      background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
      border: 2px solid #30363d;
      border-radius: 24px;
      padding: 32px;
      cursor: pointer;
      transition: all 300ms ease;
      display: flex;
      flex-direction: column;
    }

    .card:hover {
      border-color: #58a6ff;
      transform: translateY(-8px);
      box-shadow: 0 24px 64px rgba(88, 166, 255, 0.15), 0 0 0 1px rgba(88, 166, 255, 0.1);
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 24px;
    }

    .title-section {
      flex: 1;
    }

    .name {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #58a6ff;
      margin-bottom: 8px;
      font-weight: 600;
    }

    .topic {
      font-size: 28px;
      font-weight: 700;
      color: #f0f6fc;
      line-height: 1.2;
      background: linear-gradient(135deg, #f0f6fc 0%, #c9d1d9 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .stats-badge {
      background: rgba(88, 166, 255, 0.1);
      border: 1px solid rgba(88, 166, 255, 0.3);
      border-radius: 12px;
      padding: 12px 16px;
      text-align: center;
    }

    .stats-number {
      font-size: 24px;
      font-weight: 700;
      color: #58a6ff;
    }

    .stats-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #6e7681;
      margin-top: 2px;
    }

    .cloud-area {
      min-height: 500px;
      margin-bottom: 24px;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid #21262d;
    }

    .signals-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }

    .signal {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      background: rgba(255, 255, 255, 0.02);
      border-radius: 12px;
      border: 1px solid #21262d;
    }

    .signal:hover {
      background: rgba(255, 255, 255, 0.05);
      border-color: #30363d;
    }

    .dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .dot.hot {
      background: #ff6b6b;
      box-shadow: 0 0 12px #ff6b6b, 0 0 24px rgba(255, 107, 107, 0.5);
    }
    .dot.warm {
      background: #ffd43b;
      box-shadow: 0 0 8px rgba(255, 212, 59, 0.5);
    }
    .dot.cool {
      background: #69db7c;
      box-shadow: 0 0 8px rgba(105, 219, 124, 0.5);
    }

    .signal-name {
      flex: 1;
      font-size: 16px;
      font-weight: 600;
      color: #f0f6fc;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .signal-count {
      font-size: 18px;
      font-weight: 700;
      color: #8b949e;
    }

    .empty {
      min-height: 300px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: #6e7681;
      font-size: 18px;
      border: 2px dashed #30363d;
      border-radius: 16px;
      margin: 24px 0;
    }

    .empty-icon {
      font-size: 48px;
      margin-bottom: 16px;
      opacity: 0.5;
    }

    .actions {
      display: flex;
      gap: 12px;
      margin-top: 24px;
      padding-top: 24px;
      border-top: 1px solid #21262d;
    }

    .action-btn {
      flex: 1;
      padding: 14px 20px;
      border-radius: 12px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 200ms;
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .action-btn.primary {
      background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
      color: white;
    }

    .action-btn.primary:hover {
      transform: scale(1.02);
      box-shadow: 0 8px 24px rgba(46, 160, 67, 0.3);
    }

    .action-btn.secondary {
      background: rgba(88, 166, 255, 0.1);
      border: 1px solid rgba(88, 166, 255, 0.3);
      color: #58a6ff;
    }

    .action-btn.secondary:hover {
      background: rgba(88, 166, 255, 0.2);
    }

    .footer-stats {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid #21262d;
      font-size: 13px;
      color: #6e7681;
    }

    .stat-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .stat-value {
      font-weight: 600;
      color: #8b949e;
    }
  `;

  @property({ type: Object }) data!: ProjectCardData;

  private getTopSignals() {
    if (!this.data.entities) return [];
    const skip = ['Based', 'Here', 'However', 'Therefore', 'For', 'The', 'His', 'Public', 'Some', 'Without'];
    return Object.entries(this.data.entities)
      .filter(([name]) => !skip.includes(name))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4);
  }

  private getSignalLevel(count: number): string {
    if (count >= 15) return 'hot';
    if (count >= 8) return 'warm';
    return 'cool';
  }

  private handleClick() {
    this.dispatchEvent(new CustomEvent('select', { detail: this.data.name }));
  }

  render() {
    const signals = this.getTopSignals();
    const entityCount = Object.keys(this.data.entities || {}).length;
    const hasData = entityCount > 0;
    const totalMentions = Object.values(this.data.entities || {}).reduce((a, b) => a + b, 0);

    return html`
      <div class="card" @click=${this.handleClick}>
        <div class="header">
          <div class="title-section">
            <div class="name">${this.data.name}</div>
            <div class="topic">${this.data.topic || this.data.name.replace(/-/g, ' ')}</div>
          </div>
          ${hasData ? html`
            <div class="stats-badge">
              <div class="stats-number">${entityCount}</div>
              <div class="stats-label">Entities Found</div>
            </div>
          ` : null}
        </div>

        ${hasData ? html`
          <div class="cloud-area">
            <word-cloud
              .entities=${this.data.entities}
              .signalThreshold=${3}
              style="--height: 500px;"
            ></word-cloud>
          </div>

          <div class="signals-grid">
            ${signals.map(([name, count]) => html`
              <div class="signal">
                <span class="dot ${this.getSignalLevel(count)}"></span>
                <span class="signal-name">${name}</span>
                <span class="signal-count">${count}x</span>
              </div>
            `)}
          </div>

          <div class="actions">
            <button class="action-btn primary" @click=${(e: Event) => { e.stopPropagation(); this.handleClick(); }}>
              ▶ Continue Probing
            </button>
            <button class="action-btn secondary" @click=${(e: Event) => { e.stopPropagation(); this.handleClick(); }}>
              📊 View Responses
            </button>
          </div>

          <div class="footer-stats">
            <div class="stat-item">
              <span class="stat-value">${this.data.corpus_count || 0}</span> responses
            </div>
            <div class="stat-item">
              <span class="stat-value">${totalMentions}</span> total mentions
            </div>
            <div class="stat-item">
              <span class="stat-value">${signals.filter(([,c]) => c >= 10).length}</span> hot signals
            </div>
          </div>
        ` : html`
          <div class="empty">
            <div class="empty-icon">🔍</div>
            <div>No data yet</div>
            <div style="font-size: 14px; margin-top: 8px; color: #484f58;">Click to start investigation</div>
          </div>

          <div class="actions">
            <button class="action-btn primary" @click=${(e: Event) => { e.stopPropagation(); this.handleClick(); }}>
              🚀 Start Investigation
            </button>
          </div>
        `}
      </div>
    `;
  }
}
