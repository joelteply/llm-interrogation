import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { probeState, type ProbeState } from '../state';
import { getTranscript, type TranscriptQuestion } from '../api';
import type { ProbeResponse } from '../types';

@customElement('response-stream')
export class ResponseStream extends LitElement {
  @property({ type: String })
  projectName: string | null = null;
  static styles = css`
    :host {
      display: block;
    }

    .card {
      background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
      border: 2px solid #30363d;
      border-radius: 12px;
      padding: 0;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px;
      position: sticky;
      top: 0;
      background: linear-gradient(180deg, rgba(13, 17, 23, 0.98) 0%, rgba(13, 17, 23, 0.95) 100%);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid #21262d;
      z-index: 10;
    }

    .header-title {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    h3 {
      font-size: 16px;
      font-weight: 700;
      color: #f0f6fc;
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .live-badge {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: rgba(248, 81, 73, 0.2);
      border: 1px solid rgba(248, 81, 73, 0.4);
      border-radius: 20px;
      font-size: 10px;
      font-weight: 600;
      color: #f85149;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .live-dot {
      width: 6px;
      height: 6px;
      background: #f85149;
      border-radius: 50%;
      animation: pulse-live 1.5s ease-in-out infinite;
    }

    @keyframes pulse-live {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(0.8); }
    }

    .model-tags {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }

    .model-tag {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 8px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      font-size: 11px;
      color: #8b949e;
    }

    .model-tag .name {
      color: #c9d1d9;
    }

    .model-tag .close {
      cursor: pointer;
      opacity: 0.6;
      font-size: 14px;
      line-height: 1;
    }

    .model-tag .close:hover {
      opacity: 1;
      color: #f85149;
    }

    .model-tag.groq { border-color: rgba(251, 146, 60, 0.4); }
    .model-tag.deepseek { border-color: rgba(88, 166, 255, 0.4); }
    .model-tag.openai { border-color: rgba(16, 185, 129, 0.4); }

    .stats {
      display: flex;
      gap: 16px;
      font-size: 12px;
    }

    .stat {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 6px;
    }

    .stat-label {
      color: #6e7681;
    }

    .stat-value {
      color: #f0f6fc;
      font-weight: 700;
      font-size: 14px;
    }

    .content {
      padding: 20px;
    }

    .empty {
      padding: 60px 32px;
      text-align: center;
      color: #6e7681;
      font-size: 14px;
    }

    .empty-icon {
      font-size: 48px;
      margin-bottom: 16px;
      opacity: 0.5;
    }

    .running-banner {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 16px;
      background: linear-gradient(90deg, rgba(88, 166, 255, 0.1) 0%, rgba(163, 113, 247, 0.1) 100%);
      border: 1px solid rgba(88, 166, 255, 0.3);
      border-radius: 8px;
      margin-bottom: 20px;
      animation: glow-pulse 2s ease-in-out infinite;
    }

    @keyframes glow-pulse {
      0%, 100% { box-shadow: 0 0 20px rgba(88, 166, 255, 0.1); }
      50% { box-shadow: 0 0 40px rgba(88, 166, 255, 0.2); }
    }

    .spinner {
      width: 20px;
      height: 20px;
      border: 3px solid #21262d;
      border-top-color: #58a6ff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .running-text {
      font-size: 14px;
      font-weight: 600;
      color: #58a6ff;
    }

    /* Interrogation transcript style */
    .transcript {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .exchange {
      border-left: 3px solid #30363d;
      padding-left: 20px;
      animation: fade-in 0.3s ease-out;
    }

    @keyframes fade-in {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .exchange.active {
      border-left-color: #58a6ff;
    }

    .exchange.past {
      opacity: 0.8;
    }

    .exchange.past .question-text {
      background: rgba(110, 118, 129, 0.1);
      border-color: rgba(110, 118, 129, 0.2);
    }

    .interrogator {
      margin-bottom: 16px;
    }

    .interrogator-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 10px;
      font-weight: 700;
      color: #f85149;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 8px;
    }

    .question-text {
      font-size: 15px;
      font-weight: 500;
      color: #f0f6fc;
      line-height: 1.5;
      padding: 12px 16px;
      background: rgba(248, 81, 73, 0.05);
      border: 1px solid rgba(248, 81, 73, 0.2);
      border-radius: 8px;
    }

    .responses-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .subject-response {
      padding: 16px;
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 8px;
      transition: all 200ms;
    }

    .subject-response:hover {
      border-color: #30363d;
      background: #1c2128;
    }

    .subject-response.new {
      animation: highlight-new 1s ease-out;
    }

    @keyframes highlight-new {
      0% { background: rgba(88, 166, 255, 0.2); border-color: #58a6ff; }
      100% { background: #161b22; border-color: #21262d; }
    }

    .response-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }

    .subject-label {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .model-badge {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: #21262d;
      border-radius: 6px;
    }

    .model-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .model-indicator.groq { background: #3fb950; box-shadow: 0 0 8px rgba(63, 185, 80, 0.5); }
    .model-indicator.deepseek { background: #58a6ff; box-shadow: 0 0 8px rgba(88, 166, 255, 0.5); }
    .model-indicator.openai { background: #a371f7; box-shadow: 0 0 8px rgba(163, 113, 247, 0.5); }

    .model-name {
      font-size: 12px;
      font-weight: 600;
      color: #8b949e;
    }

    .run-badge {
      font-size: 10px;
      color: #6e7681;
      padding: 2px 6px;
      background: #0d1117;
      border-radius: 4px;
    }

    .response-status {
      font-size: 10px;
      padding: 4px 8px;
      border-radius: 4px;
    }

    .response-status.talking {
      background: rgba(63, 185, 80, 0.15);
      color: #3fb950;
    }

    .response-status.refused {
      background: rgba(248, 81, 73, 0.15);
      color: #f85149;
    }

    .response-text {
      font-size: 14px;
      color: #c9d1d9;
      line-height: 1.6;
    }

    .response-text.refusal {
      color: #6e7681;
      font-style: italic;
    }

    .extracted-intel {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #21262d;
    }

    .intel-label {
      font-size: 10px;
      color: #6e7681;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-right: 4px;
    }

    .intel-item {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      background: rgba(63, 185, 80, 0.1);
      border: 1px solid rgba(63, 185, 80, 0.3);
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      color: #3fb950;
    }

    .intel-item.hot {
      background: rgba(248, 81, 73, 0.1);
      border-color: rgba(248, 81, 73, 0.3);
      color: #f85149;
      animation: pulse-hot 1.5s ease-in-out infinite;
    }

    @keyframes pulse-hot {
      0%, 100% { box-shadow: 0 0 0 rgba(248, 81, 73, 0); }
      50% { box-shadow: 0 0 12px rgba(248, 81, 73, 0.3); }
    }
  `;

  @state()
  private _probeState: ProbeState = probeState.get();

  @state()
  private pastTranscript: TranscriptQuestion[] = [];

  @state()
  private isLoadingTranscript = false;

  private _unsubscribe?: () => void;

  connectedCallback() {
    super.connectedCallback();
    this._unsubscribe = probeState.subscribe((s) => {
      this._probeState = s;
    });
    this.loadPastTranscript();
  }

  async updated(changed: Map<string, unknown>) {
    if (changed.has('projectName') && this.projectName) {
      await this.loadPastTranscript();
    }
  }

  private async loadPastTranscript() {
    if (!this.projectName) return;
    this.isLoadingTranscript = true;
    try {
      const data = await getTranscript(this.projectName);
      this.pastTranscript = data.transcript;
    } catch (e) {
      console.error('Failed to load transcript:', e);
    } finally {
      this.isLoadingTranscript = false;
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribe?.();
  }

  private removeModel(model: string) {
    probeState.update(s => ({
      ...s,
      selectedModels: s.selectedModels.filter(m => m !== model)
    }));
  }

  private getModelProvider(model: string): string {
    if (model.includes('groq')) return 'groq';
    if (model.includes('deepseek')) return 'deepseek';
    if (model.includes('openai')) return 'openai';
    return 'other';
  }

  private getModelDisplayName(model: string): string {
    const parts = model.split('/');
    return parts[parts.length - 1].split('-').slice(0, 2).join(' ');
  }

  private groupResponsesByQuestion(responses: ProbeResponse[]): Map<number, ProbeResponse[]> {
    const groups = new Map<number, ProbeResponse[]>();
    for (const response of responses) {
      const existing = groups.get(response.question_index) || [];
      groups.set(response.question_index, [...existing, response]);
    }
    return groups;
  }

  private isHotEntity(entity: string): boolean {
    // Count how many times this entity appears across all responses
    const count = this._probeState.responses.filter(r =>
      r.entities?.includes(entity)
    ).length;
    return count >= 3;
  }

  render() {
    const { responses, questions, isRunning } = this._probeState;
    const grouped = this.groupResponsesByQuestion(responses);

    // Combine current session + past transcripts
    const hasCurrentSession = responses.length > 0;
    const hasPastData = this.pastTranscript.length > 0;
    const totalResponses = responses.length + this.pastTranscript.reduce((sum, q) => sum + q.responses.length, 0);

    return html`
      <div class="card">
        <div class="header">
          <div class="header-title">
            <h3>${isRunning ? 'Interrogating' : 'Interrogation'}</h3>
            ${isRunning ? html`
              <div class="live-badge">
                <span class="live-dot"></span>
                Live
              </div>
            ` : null}
            <div class="model-tags">
              ${this._probeState.selectedModels.map(model => {
                const provider = this.getModelProvider(model);
                const name = this.getModelDisplayName(model);
                return html`
                  <span class="model-tag ${provider}">
                    <span class="name">${name}</span>
                    <span class="close" @click=${() => this.removeModel(model)}>×</span>
                  </span>
                `;
              })}
            </div>
          </div>
          <div class="stats">
            <div class="stat">
              <span class="stat-label">Responses</span>
              <span class="stat-value">${totalResponses}</span>
            </div>
            <div class="stat">
              <span class="stat-label">Questions</span>
              <span class="stat-value">${questions.length + this.pastTranscript.length}</span>
            </div>
          </div>
        </div>

        <div class="content">
          ${isRunning ? html`
            <div class="running-banner">
              <div class="spinner"></div>
              <span class="running-text">Interrogating subjects...</span>
            </div>
          ` : null}

          ${!hasCurrentSession && !hasPastData && !isRunning && !this.isLoadingTranscript ? html`
            <div class="empty">
              <div class="empty-icon">🎯</div>
              <div>Click "Run Probe" to begin interrogation</div>
              <div style="font-size: 12px; margin-top: 8px; color: #484f58;">
                Questions will be generated and fired at the selected models
              </div>
            </div>
          ` : null}

          ${this.isLoadingTranscript ? html`
            <div class="empty">
              <div class="spinner"></div>
              <div style="margin-top: 16px;">Loading past interrogations...</div>
            </div>
          ` : null}

          <div class="transcript">
            <!-- Current session responses (newest first) -->
            ${Array.from(grouped.entries()).reverse().map(([questionIndex, questionResponses]) => {
              const question = questions[questionIndex];
              const isActive = isRunning && questionIndex === Math.max(...Array.from(grouped.keys()));
              return this.renderExchange(question?.question || `Question ${questionIndex + 1}`, questionResponses, isActive, true);
            })}

            <!-- Past transcript -->
            ${this.pastTranscript.slice().reverse().map(q =>
              this.renderExchange(q.question, q.responses.map(r => ({
                ...r,
                question: q.question,
                question_index: q.question_index
              })), false, false)
            )}
          </div>
        </div>
      </div>
    `;
  }

  private renderExchange(question: string, responses: any[], isActive: boolean, isCurrent: boolean) {
    return html`
      <div class="exchange ${isActive ? 'active' : ''} ${isCurrent ? 'current' : 'past'}">
        <div class="interrogator">
          <div class="interrogator-label">
            <span>⚡</span> Interrogator ${!isCurrent ? html`<span style="opacity: 0.5; font-size: 9px;">(past)</span>` : ''}
          </div>
          <div class="question-text">
            "${question}"
          </div>
        </div>

        <div class="responses-list">
          ${responses.slice(0, 10).map((r: any, i: number) => {
            const isNew = i === 0 && isActive && isCurrent;
            return html`
              <div class="subject-response ${isNew ? 'new' : ''}">
                <div class="response-header">
                  <div class="subject-label">
                    <div class="model-badge">
                      <span class="model-indicator ${this.getModelProvider(r.model)}"></span>
                      <span class="model-name">${this.getModelDisplayName(r.model)}</span>
                    </div>
                    <span class="run-badge">Run #${(r.run_index || 0) + 1}</span>
                  </div>
                  <span class="response-status ${r.is_refusal ? 'refused' : 'talking'}">
                    ${r.is_refusal ? '🚫 Refused' : '💬 Talking'}
                  </span>
                </div>
                <div class="response-text ${r.is_refusal ? 'refusal' : ''}">
                  ${r.response}
                </div>
                ${r.entities && r.entities.length ? html`
                  <div class="extracted-intel">
                    <span class="intel-label">Extracted:</span>
                    ${r.entities.slice(0, 8).map((e: string) => html`
                      <span class="intel-item ${this.isHotEntity(e) ? 'hot' : ''}">${e}</span>
                    `)}
                    ${r.entities.length > 8 ? html`<span class="intel-item" style="opacity: 0.5;">+${r.entities.length - 8} more</span>` : ''}
                  </div>
                ` : null}
              </div>
            `;
          })}
          ${responses.length > 10 ? html`
            <div style="text-align: center; padding: 8px; color: #6e7681; font-size: 12px;">
              +${responses.length - 10} more responses...
            </div>
          ` : null}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'response-stream': ResponseStream;
  }
}
