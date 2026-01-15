import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { probeState, type ProbeState } from '../state';
import { TECHNIQUE_INFO, type GeneratedQuestion } from '../types';

@customElement('question-queue')
export class QuestionQueue extends LitElement {
  static styles = css`
    :host {
      display: block;
      height: 100%;
      overflow: hidden;
    }

    .card {
      background: var(--bg-secondary, #161b22);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      height: 100%;
      max-height: 100%;
      overflow: hidden;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
      flex-shrink: 0;
    }

    h3 {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary, #c9d1d9);
      margin: 0;
    }

    .count {
      font-size: 12px;
      color: var(--text-muted, #6e7681);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .clear-btn {
      padding: 4px 10px;
      font-size: 11px;
      background: var(--bg-tertiary, #21262d);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 6px;
      color: var(--text-secondary, #8b949e);
      cursor: pointer;
      transition: all 0.15s;
    }

    .clear-btn:hover:not(:disabled) {
      background: var(--bg-secondary, #161b22);
      border-color: var(--accent-blue, #58a6ff);
      color: var(--text-primary, #c9d1d9);
    }

    .clear-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .empty {
      padding: 24px;
      text-align: center;
      color: var(--text-secondary, #8b949e);
      font-size: 13px;
    }

    .questions {
      display: flex;
      flex-direction: column;
      gap: 8px;
      overflow-y: auto;
      flex: 1;
      min-height: 0;
    }

    .question-item {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 12px;
      background: var(--bg-primary, #0d1117);
      border: 1px solid var(--border-muted, #21262d);
      border-radius: 8px;
      transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
    }

    .question-item:hover {
      border-color: var(--border-default, #30363d);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .question-header {
      display: flex;
      align-items: center;
      gap: 8px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border-muted, #21262d);
      margin-bottom: 2px;
    }

    .question-item.active {
      box-shadow: inset 0 0 0 2px rgba(88, 166, 255, 0.4);
    }

    .question-item.completed {
      opacity: 0.6;
    }

    .active-indicator {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--text-muted, #6e7681);
      margin-top: 6px;
    }

    .pulse {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--text-muted, #6e7681);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 0.3; }
      50% { opacity: 0.7; }
    }

    .question-index {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 20px;
      height: 20px;
      background: var(--bg-tertiary, #21262d);
      border-radius: 50%;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-secondary, #8b949e);
      flex-shrink: 0;
    }

    .question-text {
      font-size: 13px;
      color: var(--text-primary, #c9d1d9);
      line-height: 1.45;
    }

    .question-meta {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 6px;
      padding-top: 4px;
      opacity: 0.8;
    }

    .template-badge {
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .technique-tag {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      background: var(--bg-tertiary, #21262d);
      border-radius: 9999px;
      font-size: 10px;
      color: var(--text-secondary, #8b949e);
    }

    /* Technique colors now come from template YAML via inline style */

    .target-entity {
      font-size: 10px;
      color: var(--text-muted, #6e7681);
    }

    .question-actions {
      display: flex;
      gap: 2px;
      margin-left: auto;
    }

    .action-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      background: var(--bg-tertiary, #21262d);
      border: none;
      border-radius: 4px;
      color: var(--text-muted, #6e7681);
      cursor: pointer;
      font-size: 12px;
      transition: all 0.15s;
    }

    .action-btn:hover {
      background: var(--border-default, #30363d);
      color: var(--text-primary, #c9d1d9);
    }

    .action-btn.delete:hover {
      color: var(--accent-red, #f85149);
    }

    .action-btn.skip {
      color: var(--accent-yellow, #d29922);
    }

    .action-btn.skip:hover {
      background: rgba(210, 153, 34, 0.15);
    }

    .add-question {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-muted, #21262d);
      flex-shrink: 0;
    }

    .add-question input {
      flex: 1;
      padding: 8px 12px;
      background: var(--bg-primary, #0d1117);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 6px;
      color: var(--text-primary, #c9d1d9);
      font-size: 13px;
    }

    .add-question input:focus {
      outline: none;
      border-color: var(--accent-blue, #58a6ff);
    }

    .add-question button {
      padding: 8px 12px;
      background: var(--bg-tertiary, #21262d);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 6px;
      color: var(--text-primary, #c9d1d9);
      font-size: 13px;
      cursor: pointer;
    }

    .add-question button:hover {
      border-color: var(--accent-blue, #58a6ff);
    }
  `;

  @state()
  private _probeState: ProbeState = probeState.get();

  @state()
  private newQuestion = '';

  private _unsubscribe?: () => void;

  connectedCallback() {
    super.connectedCallback();
    this.loadTemplates();  // Load template colors from API
    this._unsubscribe = probeState.subscribe((s) => {
      this._probeState = s;
      this.requestUpdate();
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribe?.();
  }

  private getTechniqueClass(technique?: string): string {
    // Return empty - styling now comes from template color, not hardcoded classes
    return '';
  }

  private getTechniqueName(technique?: string): string {
    if (!technique) return 'unknown';
    return TECHNIQUE_INFO[technique as keyof typeof TECHNIQUE_INFO]?.name || technique;
  }

  private getModelShortName(model: string): string {
    // "groq/llama-3.1-8b-instant" -> "llama-3.1-8b"
    const parts = model.split('/');
    const name = parts[parts.length - 1];
    return name.replace('-instant', '').split('-').slice(0, 3).join('-');
  }

  // Loaded from /api/techniques - maps template name to color
  private static templateCache: Record<string, string> = {};
  private static templateCacheLoaded = false;

  private async loadTemplates() {
    if (QuestionQueue.templateCacheLoaded) return;
    try {
      const resp = await fetch('/api/techniques');
      const templates = await resp.json();
      for (const t of templates) {
        QuestionQueue.templateCache[t.name] = t.color;
      }
      QuestionQueue.templateCacheLoaded = true;
      this.requestUpdate();
    } catch (e) {
      console.warn('Failed to load templates:', e);
    }
  }

  private parseQuestion(q: GeneratedQuestion): { template: string; technique: string; color: string } {
    let template = q.template || '';
    let technique = q.technique || '';

    // If technique contains "/", split it
    if (technique.includes('/')) {
      const parts = technique.split('/');
      if (!template) template = parts[0];
      technique = parts[1] || parts[0];
    }

    // Strip prefixes and clean up technique for display
    const cleanTechnique = technique
      .replace(/^[a-z]+_/, '')  // Strip any prefix like fbi_, scharff_, etc
      .replace(/_/g, ' ');

    // Color from: explicit q.color > cached template color > gray fallback
    const color = q.color || QuestionQueue.templateCache[template] || '#8b949e';

    return { template, technique: cleanTechnique || 'custom', color };
  }

  private getTemplateBadgeStyle(color: string): string {
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    return `background: rgba(${r}, ${g}, ${b}, 0.15); border: 1px solid rgba(${r}, ${g}, ${b}, 0.3); color: ${color};`;
  }

  private removeQuestion(index: number) {
    probeState.update((s) => ({
      ...s,
      questions: s.questions.filter((_, i) => i !== index),
    }));
  }

  private moveQuestion(from: number, to: number) {
    probeState.update((s) => {
      const questions = [...s.questions];
      const [moved] = questions.splice(from, 1);
      questions.splice(to, 0, moved);
      return { ...s, questions };
    });
  }

  private skipCurrent() {
    // Move current question to end of queue
    const { currentQuestionIndex, questions } = this._probeState;
    if (currentQuestionIndex >= 0 && currentQuestionIndex < questions.length) {
      this.moveQuestion(currentQuestionIndex, questions.length - 1);
    }
  }

  private addQuestion() {
    if (!this.newQuestion.trim()) return;

    const question: GeneratedQuestion = {
      question: this.newQuestion.trim(),
      technique: 'custom', // Manual question - no template technique
    };

    probeState.update((s) => ({
      ...s,
      questions: [question, ...s.questions],
    }));

    this.newQuestion = '';
  }

  private handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      this.addQuestion();
    }
  }

  private clearAndRegenerate() {
    // Clear queue and trigger regeneration with current template settings
    probeState.update((s) => ({
      ...s,
      questions: [],
      currentQuestionIndex: -1,
    }));

    // Dispatch event to trigger question generation in probe-controls
    this.dispatchEvent(new CustomEvent('regenerate-questions', {
      bubbles: true,
      composed: true,
    }));
  }

  render() {
    const { questions, isRunning, currentQuestionIndex, activeModel } = this._probeState;

    if (questions.length === 0 && !isRunning) {
      return html`
        <div class="card">
          <div class="header">
            <h3>Question Queue</h3>
          </div>
          <div class="empty">
            Click "Preview Questions" to generate questions using interrogation techniques,
            or add your own below.
          </div>
          <div class="add-question">
            <input
              type="text"
              placeholder="Add a custom question..."
              .value=${this.newQuestion}
              @input=${(e: Event) =>
                (this.newQuestion = (e.target as HTMLInputElement).value)}
              @keydown=${this.handleKeydown}
            />
            <button @click=${this.addQuestion}>Add</button>
          </div>
        </div>
      `;
    }

    // Calculate pending count for header
    const pendingCount = questions.length - Math.max(0, currentQuestionIndex);
    const doneCount = Math.max(0, currentQuestionIndex);

    return html`
      <div class="card">
        <div class="header">
          <h3>Question Queue</h3>
          <div class="header-actions">
            <span class="count">${pendingCount} pending${doneCount > 0 ? ` · ${doneCount} done` : ''}</span>
            <button
              class="clear-btn"
              @click=${this.clearAndRegenerate}
              title="Clear queue and regenerate with current template"
              ?disabled=${isRunning}
            >🔄 Refresh</button>
          </div>
        </div>

        <div class="add-question">
          <input
            type="text"
            placeholder="Add a custom question..."
            .value=${this.newQuestion}
            @input=${(e: Event) =>
              (this.newQuestion = (e.target as HTMLInputElement).value)}
            @keydown=${this.handleKeydown}
          />
          <button @click=${this.addQuestion}>Add</button>
        </div>

        <div class="questions">
          ${(() => {
            // QUEUE BEHAVIOR: Only show current + pending questions
            // Completed questions disappear from view
            const startIdx = Math.max(0, currentQuestionIndex);
            const pendingQs = questions.slice(startIdx);
            const doneCount = startIdx;

            if (pendingQs.length === 0 && doneCount > 0) {
              return html`<div class="empty">All ${doneCount} questions completed</div>`;
            }

            return pendingQs.map((q, i) => {
              const realIdx = startIdx + i;
              const isActive = realIdx === currentQuestionIndex;
              return html`
                <div class="question-item ${isActive ? 'active' : ''}">
                  <div class="question-header">
                    <span class="question-index">${i + 1}</span>
                    <div class="question-actions">
                      ${!isRunning ? html`
                        <button
                          class="action-btn"
                          @click=${() => this.moveQuestion(realIdx, startIdx)}
                          title="Run next"
                        >↑</button>
                        <button
                          class="action-btn delete"
                          @click=${() => this.removeQuestion(realIdx)}
                          title="Remove"
                        >&times;</button>
                      ` : isActive ? html`
                        <button
                          class="action-btn skip"
                          @click=${() => this.skipCurrent()}
                          title="Skip"
                        >⏭</button>
                      ` : null}
                    </div>
                  </div>
                  <div class="question-text">${q.question}</div>
                  <div class="question-meta">
                    ${(() => {
                      const { template, technique, color } = this.parseQuestion(q);
                      return html`
                        ${template ? html`
                          <span class="template-badge" style=${this.getTemplateBadgeStyle(color)}>
                            ${template}
                          </span>
                        ` : null}
                        <span class="technique-tag">${technique}</span>
                      `;
                    })()}
                    ${q.target_entity ? html`<span class="target-entity">→ ${q.target_entity}</span>` : null}
                  </div>
                  ${isActive && activeModel ? html`
                    <div class="active-indicator">
                      <span class="pulse"></span>
                      Asking ${this.getModelShortName(activeModel)}...
                    </div>
                  ` : null}
                </div>
              `;
            });
          })()}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'question-queue': QuestionQueue;
  }
}
