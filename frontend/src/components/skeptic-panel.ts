import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface SkepticFeedback {
  weakest_link?: string;
  alternative_explanation?: string;
  circular_evidence?: string[];
  counter_questions?: string[];
  missing_research?: string;
  confidence?: string;
  updated?: number;
}

@customElement('skeptic-panel')
export class SkepticPanel extends LitElement {
  @property({ type: String }) projectName = '';
  @state() private feedback: SkepticFeedback | null = null;
  @state() private loading = false;
  @state() private expanded = false;

  static styles = css`
    :host {
      display: block;
      font-family: system-ui, sans-serif;
    }

    .skeptic-header {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: linear-gradient(135deg, #2d1f1f 0%, #1a1a1a 100%);
      border: 1px solid #e74c3c44;
      border-radius: 6px;
      cursor: pointer;
      user-select: none;
    }

    .skeptic-header:hover {
      background: linear-gradient(135deg, #3d2f2f 0%, #2a2a2a 100%);
    }

    .devil-icon {
      font-size: 16px;
    }

    .title {
      color: #e74c3c;
      font-weight: 600;
      font-size: 13px;
      flex: 1;
    }

    .confidence {
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
    }

    .confidence.LOW { background: #e74c3c33; color: #e74c3c; }
    .confidence.MEDIUM { background: #f39c1233; color: #f39c12; }
    .confidence.HIGH { background: #27ae6033; color: #27ae60; }

    .expand-icon {
      color: #888;
      transition: transform 0.2s;
    }

    .expand-icon.expanded {
      transform: rotate(180deg);
    }

    .content {
      display: none;
      padding: 12px;
      background: #1a1a1a;
      border: 1px solid #333;
      border-top: none;
      border-radius: 0 0 6px 6px;
    }

    .content.expanded {
      display: block;
    }

    .section {
      margin-bottom: 12px;
    }

    .section:last-child {
      margin-bottom: 0;
    }

    .section-title {
      color: #e74c3c;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      margin-bottom: 4px;
    }

    .section-content {
      color: #ccc;
      font-size: 12px;
      line-height: 1.4;
    }

    .counter-questions {
      list-style: none;
      padding: 0;
      margin: 0;
    }

    .counter-questions li {
      padding: 6px 8px;
      background: #2a1a1a;
      border-left: 2px solid #e74c3c;
      margin-bottom: 4px;
      font-size: 12px;
      color: #ddd;
    }

    .circular-list {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }

    .circular-item {
      background: #3a2a2a;
      color: #e74c3c;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
    }

    .no-data {
      color: #666;
      font-style: italic;
      font-size: 12px;
      padding: 8px;
      text-align: center;
    }

    .updated {
      color: #555;
      font-size: 10px;
      margin-top: 8px;
      text-align: right;
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    this.loadFeedback();
    // Refresh every 30 seconds
    this._interval = setInterval(() => this.loadFeedback(), 30000);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this._interval) clearInterval(this._interval);
  }

  private _interval: any;

  async loadFeedback() {
    if (!this.projectName) return;

    try {
      const resp = await fetch(`/api/projects/${this.projectName}/skeptic`);
      if (resp.ok) {
        this.feedback = await resp.json();
      }
    } catch (e) {
      console.error('Failed to load skeptic feedback:', e);
    }
  }

  private formatTime(ts: number): string {
    if (!ts) return '';
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString();
  }

  render() {
    const hasData = this.feedback && (
      this.feedback.weakest_link ||
      this.feedback.counter_questions?.length
    );

    return html`
      <div
        class="skeptic-header"
        @click=${() => this.expanded = !this.expanded}
      >
        <span class="devil-icon">&#128520;</span>
        <span class="title">Devil's Advocate</span>
        ${this.feedback?.confidence ? html`
          <span class="confidence ${this.feedback.confidence}">${this.feedback.confidence}</span>
        ` : null}
        <span class="expand-icon ${this.expanded ? 'expanded' : ''}">&#9660;</span>
      </div>

      <div class="content ${this.expanded ? 'expanded' : ''}">
        ${!hasData ? html`
          <div class="no-data">No skeptical analysis yet. Run the probe to generate findings.</div>
        ` : html`
          ${this.feedback?.weakest_link ? html`
            <div class="section">
              <div class="section-title">Weakest Link</div>
              <div class="section-content">${this.feedback.weakest_link}</div>
            </div>
          ` : null}

          ${this.feedback?.alternative_explanation ? html`
            <div class="section">
              <div class="section-title">Alternative Explanation</div>
              <div class="section-content">${this.feedback.alternative_explanation}</div>
            </div>
          ` : null}

          ${this.feedback?.counter_questions?.length ? html`
            <div class="section">
              <div class="section-title">Counter-Questions</div>
              <ul class="counter-questions">
                ${this.feedback.counter_questions.map(q => html`<li>${q}</li>`)}
              </ul>
            </div>
          ` : null}

          ${this.feedback?.circular_evidence?.length ? html`
            <div class="section">
              <div class="section-title">Circular Evidence</div>
              <div class="circular-list">
                ${this.feedback.circular_evidence.map(e => html`
                  <span class="circular-item">${e}</span>
                `)}
              </div>
            </div>
          ` : null}

          ${this.feedback?.missing_research ? html`
            <div class="section">
              <div class="section-title">Missing Research</div>
              <div class="section-content">${this.feedback.missing_research}</div>
            </div>
          ` : null}

          ${this.feedback?.updated ? html`
            <div class="updated">Updated: ${this.formatTime(this.feedback.updated)}</div>
          ` : null}
        `}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'skeptic-panel': SkepticPanel;
  }
}
