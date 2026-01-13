import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { probeState, type ProbeState } from '../state';
import { getTranscript, updateProject, type TranscriptQuestion } from '../api';
import type { ProbeResponse } from '../types';
import { AVAILABLE_MODELS } from '../types';

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

    .model-tag.groq { border-color: rgba(63, 185, 80, 0.4); }
    .model-tag.deepseek { border-color: rgba(88, 166, 255, 0.4); }
    .model-tag.openai { border-color: rgba(163, 113, 247, 0.4); }
    .model-tag.anthropic { border-color: rgba(212, 165, 116, 0.4); }
    .model-tag.xai { border-color: rgba(248, 81, 73, 0.4); }
    .model-tag.mistral { border-color: rgba(255, 123, 0, 0.4); }
    .model-tag.google { border-color: rgba(66, 133, 244, 0.4); }
    .model-tag.together { border-color: rgba(232, 121, 249, 0.4); }
    .model-tag.cohere { border-color: rgba(57, 211, 83, 0.4); }

    .add-model-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      background: #21262d;
      border: 1px dashed #30363d;
      border-radius: 4px;
      color: #6e7681;
      cursor: pointer;
      font-size: 14px;
      transition: all 150ms;
    }

    .add-model-btn:hover {
      background: #30363d;
      border-color: #58a6ff;
      color: #58a6ff;
    }

    .model-dropdown {
      position: relative;
    }

    .model-dropdown-menu {
      position: absolute;
      top: 100%;
      left: 0;
      margin-top: 4px;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      z-index: 100;
      min-width: 180px;
    }

    .model-option {
      padding: 8px 12px;
      font-size: 12px;
      color: #c9d1d9;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .model-option:hover {
      background: #21262d;
    }

    .model-option:first-child {
      border-radius: 6px 6px 0 0;
    }

    .model-option:last-child {
      border-radius: 0 0 6px 6px;
    }

    .model-option .provider {
      font-size: 10px;
      color: #6e7681;
    }

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
      align-items: flex-start;
      gap: 10px;
      padding: 10px 14px;
      background: linear-gradient(90deg, rgba(88, 166, 255, 0.08) 0%, rgba(163, 113, 247, 0.08) 100%);
      border: 1px solid rgba(88, 166, 255, 0.2);
      border-radius: 6px;
      margin-bottom: 16px;
      font-size: 12px;
      line-height: 1.5;
    }

    .running-banner:not(.idle) {
      animation: glow-pulse 2s ease-in-out infinite;
    }

    .running-banner.idle {
      background: rgba(110, 118, 129, 0.1);
      border-color: rgba(110, 118, 129, 0.2);
    }

    .narrative-box {
      background: rgba(63, 185, 80, 0.08);
      border: 1px solid rgba(63, 185, 80, 0.25);
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 16px;
    }

    .narrative-header {
      font-size: 11px;
      font-weight: 700;
      color: #3fb950;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }

    .narrative-content {
      font-size: 13px;
      color: #c9d1d9;
      line-height: 1.6;
      white-space: pre-wrap;
      max-height: 150px;
      overflow-y: auto;
      cursor: pointer;
    }

    .narrative-content:hover {
      background: rgba(63, 185, 80, 0.05);
      border-radius: 4px;
      margin: -4px;
      padding: 4px;
    }

    .narrative-edit {
      width: 100%;
      min-height: 250px;
      max-height: 60vh;
      padding: 12px;
      background: #0d1117;
      border: 1px solid rgba(63, 185, 80, 0.4);
      border-radius: 6px;
      font-size: 13px;
      font-family: inherit;
      color: #c9d1d9;
      line-height: 1.6;
      resize: vertical;
    }

    .narrative-edit:focus {
      outline: none;
      border-color: #3fb950;
      box-shadow: 0 0 0 2px rgba(63, 185, 80, 0.2);
    }

    .narrative-header {
      cursor: pointer;
    }

    .narrative-header:hover {
      color: #56d364;
    }

    .running-indicator {
      margin-left: 8px;
      color: #f85149;
      animation: blink 1s ease-in-out infinite;
    }

    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
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
      font-size: 12px;
      font-weight: 500;
      color: #8b949e;
      white-space: pre-wrap;
      flex: 1;
    }

    .running-banner:not(.idle) .running-text {
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
    .model-indicator.anthropic { background: #d4a574; box-shadow: 0 0 8px rgba(212, 165, 116, 0.5); }
    .model-indicator.xai { background: #f85149; box-shadow: 0 0 8px rgba(248, 81, 73, 0.5); }
    .model-indicator.mistral { background: #ff7b00; box-shadow: 0 0 8px rgba(255, 123, 0, 0.5); }
    .model-indicator.google { background: #4285f4; box-shadow: 0 0 8px rgba(66, 133, 244, 0.5); }
    .model-indicator.together { background: #e879f9; box-shadow: 0 0 8px rgba(232, 121, 249, 0.5); }
    .model-indicator.cohere { background: #39d353; box-shadow: 0 0 8px rgba(57, 211, 83, 0.5); }

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

  @state()
  private showModelDropdown = false;

  @state()
  private editingNarrative = false;

  private _unsubscribe?: () => void;

  private handleClickOutside = (e: MouseEvent) => {
    if (this.showModelDropdown) {
      const dropdown = this.shadowRoot?.querySelector('.model-dropdown');
      if (dropdown && !dropdown.contains(e.target as Node)) {
        this.showModelDropdown = false;
      }
    }
  };

  connectedCallback() {
    super.connectedCallback();
    this._unsubscribe = probeState.subscribe((s) => {
      this._probeState = s;
    });
    document.addEventListener('click', this.handleClickOutside);
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
    document.removeEventListener('click', this.handleClickOutside);
  }

  private async removeModel(model: string) {
    const current = probeState.get();
    const remaining = current.selectedModels.filter(m => m !== model);

    if (remaining.length === 0 && current.isRunning) {
      // Stop the probe if we remove the last model
      probeState.update(s => ({
        ...s,
        selectedModels: remaining,
        isRunning: false
      }));
      // Dispatch stop event for parent to handle abort
      this.dispatchEvent(new CustomEvent('stop-probe', { bubbles: true, composed: true }));
    } else {
      probeState.update(s => ({
        ...s,
        selectedModels: remaining
      }));
    }
    // Save to project
    if (this.projectName) {
      updateProject(this.projectName, { selected_models: remaining });
    }
  }

  private async addModel(model: string) {
    const newModels = probeState.get().selectedModels.includes(model)
      ? probeState.get().selectedModels
      : [...probeState.get().selectedModels, model];

    probeState.update(s => ({
      ...s,
      selectedModels: newModels
    }));
    this.showModelDropdown = false;

    // Save to project
    if (this.projectName) {
      updateProject(this.projectName, { selected_models: newModels });
    }
  }

  private async saveNarrative(e: Event) {
    const textarea = e.target as HTMLTextAreaElement;
    const newNarrative = textarea.value;

    probeState.update(s => ({
      ...s,
      narrative: newNarrative
    }));

    this.editingNarrative = false;

    // Save to project
    if (this.projectName) {
      updateProject(this.projectName, { narrative: newNarrative });
    }
  }

  private getModelProvider(model: string): string {
    if (model.includes('groq')) return 'groq';
    if (model.includes('deepseek')) return 'deepseek';
    if (model.includes('openai')) return 'openai';
    if (model.includes('anthropic') || model.includes('claude')) return 'anthropic';
    if (model.includes('xai') || model.includes('grok')) return 'xai';
    if (model.includes('mistral')) return 'mistral';
    if (model.includes('google') || model.includes('gemini')) return 'google';
    if (model.includes('together') || model.includes('meta-llama')) return 'together';
    if (model.includes('cohere') || model.includes('command')) return 'cohere';
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
              <div class="model-dropdown">
                <button
                  class="add-model-btn"
                  @click=${(e: Event) => { e.stopPropagation(); this.showModelDropdown = !this.showModelDropdown; }}
                  title="Add model"
                >+</button>
                ${this.showModelDropdown ? html`
                  <div class="model-dropdown-menu">
                    ${AVAILABLE_MODELS
                      .filter(m => !this._probeState.selectedModels.includes(m.id))
                      .map(m => html`
                        <div class="model-option" @click=${(e: Event) => { e.stopPropagation(); this.addModel(m.id); }}>
                          <span>${m.name}</span>
                          <span class="provider">${m.provider}</span>
                        </div>
                      `)}
                    ${AVAILABLE_MODELS.filter(m => !this._probeState.selectedModels.includes(m.id)).length === 0 ? html`
                      <div class="model-option" style="color: #6e7681; cursor: default;">All models selected</div>
                    ` : null}
                  </div>
                ` : null}
              </div>
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
          <!-- Always show narrative box so user can add corrections/notes -->
          <div class="narrative-box">
            <div class="narrative-header" @click=${() => this.editingNarrative = !this.editingNarrative}>
              📓 Working Theory ${this.editingNarrative ? '(editing)' : '(click to edit)'}
              ${isRunning ? html`<span class="running-indicator">● Live</span>` : ''}
            </div>
            ${this.editingNarrative ? html`
              <textarea
                class="narrative-edit"
                .value=${this._probeState.narrative || ''}
                @blur=${this.saveNarrative}
                @keydown=${(e: KeyboardEvent) => { if (e.key === 'Escape') this.editingNarrative = false; }}
                placeholder="Add your notes, corrections, or working theory here. Mark things as WRONG or FALSE if the AI hallucinates. This helps guide the interrogation."
              ></textarea>
            ` : html`
              <div class="narrative-content">${this._probeState.narrative || 'Click to add notes, corrections, or your own working theory...'}</div>
            `}
          </div>

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
          ${responses.slice(-10).reverse().map((r: any, i: number) => {
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
