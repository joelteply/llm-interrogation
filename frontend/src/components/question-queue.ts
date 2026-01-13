import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { probeState, type ProbeState } from '../state';
import { TECHNIQUE_INFO, type GeneratedQuestion } from '../types';

@customElement('question-queue')
export class QuestionQueue extends LitElement {
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

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
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
    }

    .question-item {
      display: flex;
      gap: 12px;
      padding: 12px;
      background: var(--bg-primary, #0d1117);
      border: 1px solid var(--border-muted, #21262d);
      border-radius: 6px;
    }

    .question-index {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      background: var(--bg-tertiary, #21262d);
      border-radius: 50%;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-secondary, #8b949e);
      flex-shrink: 0;
    }

    .question-content {
      flex: 1;
      min-width: 0;
    }

    .question-text {
      font-size: 13px;
      color: var(--text-primary, #c9d1d9);
      margin-bottom: 6px;
      line-height: 1.4;
    }

    .question-meta {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .technique-tag {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      background: var(--bg-tertiary, #21262d);
      border-radius: 9999px;
      font-size: 10px;
      color: var(--text-muted, #6e7681);
    }

    .technique-tag.scharff {
      background: rgba(163, 113, 247, 0.15);
      color: var(--accent-purple, #a371f7);
    }

    .technique-tag.fbi {
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent-blue, #58a6ff);
    }

    .technique-tag.cognitive {
      background: rgba(63, 185, 80, 0.15);
      color: var(--accent-green, #3fb950);
    }

    .target-entity {
      font-size: 10px;
      color: var(--text-muted, #6e7681);
    }

    .question-actions {
      display: flex;
      gap: 4px;
      flex-shrink: 0;
    }

    .action-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      background: none;
      border: none;
      border-radius: 4px;
      color: var(--text-muted, #6e7681);
      cursor: pointer;
      font-size: 14px;
    }

    .action-btn:hover {
      background: var(--bg-tertiary, #21262d);
      color: var(--text-secondary, #8b949e);
    }

    .action-btn.delete:hover {
      color: var(--accent-red, #f85149);
    }

    .add-question {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-muted, #21262d);
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
    this._unsubscribe = probeState.subscribe((s) => {
      this._probeState = s;
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribe?.();
  }

  private getTechniqueClass(technique: string): string {
    if (technique.startsWith('scharff')) return 'scharff';
    if (technique.startsWith('fbi')) return 'fbi';
    if (technique.startsWith('cognitive')) return 'cognitive';
    return '';
  }

  private getTechniqueName(technique: string): string {
    return TECHNIQUE_INFO[technique as keyof typeof TECHNIQUE_INFO]?.name || technique;
  }

  private removeQuestion(index: number) {
    probeState.update((s) => ({
      ...s,
      questions: s.questions.filter((_, i) => i !== index),
    }));
  }

  private addQuestion() {
    if (!this.newQuestion.trim()) return;

    const question: GeneratedQuestion = {
      question: this.newQuestion.trim(),
      technique: 'fbi_macro_to_micro', // Default for manual questions
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

  render() {
    const { questions, isRunning } = this._probeState;

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

    return html`
      <div class="card">
        <div class="header">
          <h3>Question Queue</h3>
          <span class="count">${questions.length} questions</span>
        </div>

        ${!isRunning
          ? html`
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
            `
          : null}

        <div class="questions">
          ${questions.map(
            (q, i) => html`
              <div class="question-item">
                <span class="question-index">${i + 1}</span>
                <div class="question-content">
                  <div class="question-text">${q.question}</div>
                  <div class="question-meta">
                    <span class="technique-tag ${this.getTechniqueClass(q.technique)}">
                      ${this.getTechniqueName(q.technique)}
                    </span>
                    ${q.target_entity
                      ? html`<span class="target-entity">→ ${q.target_entity}</span>`
                      : null}
                  </div>
                </div>
                ${!isRunning
                  ? html`
                      <div class="question-actions">
                        <button
                          class="action-btn delete"
                          @click=${() => this.removeQuestion(i)}
                          title="Remove question"
                        >
                          &times;
                        </button>
                      </div>
                    `
                  : null}
              </div>
            `
          )}
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
