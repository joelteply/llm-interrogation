import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

@customElement('tag-input')
export class TagInput extends LitElement {
  static styles = css`
    :host {
      display: block;
    }

    .container {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 8px;
      background: var(--bg-secondary, #161b22);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 6px;
      min-height: 40px;
    }

    .container:focus-within {
      border-color: var(--accent-blue, #58a6ff);
    }

    .tag {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
      background: var(--bg-tertiary, #21262d);
      border-radius: 9999px;
      font-size: 12px;
      color: var(--text-secondary, #8b949e);
    }

    .tag-remove {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 14px;
      height: 14px;
      background: none;
      border: none;
      border-radius: 50%;
      color: var(--text-muted, #6e7681);
      cursor: pointer;
      font-size: 14px;
      line-height: 1;
      padding: 0;
    }

    .tag-remove:hover {
      background: rgba(248, 81, 73, 0.2);
      color: var(--accent-red, #f85149);
    }

    input {
      flex: 1;
      min-width: 100px;
      background: none;
      border: none;
      color: var(--text-primary, #c9d1d9);
      font-size: 14px;
      outline: none;
    }

    input::placeholder {
      color: var(--text-muted, #6e7681);
    }
  `;

  @property({ type: Array })
  tags: string[] = [];

  @property({ type: String })
  placeholder = 'Add tag...';

  @state()
  private inputValue = '';

  private handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      this.addTag();
    } else if (e.key === 'Backspace' && !this.inputValue && this.tags.length) {
      this.removeTag(this.tags.length - 1);
    }
  }

  private handleInput(e: Event) {
    const value = (e.target as HTMLInputElement).value;
    if (value.includes(',')) {
      this.inputValue = value.replace(',', '');
      this.addTag();
    } else {
      this.inputValue = value;
    }
  }

  private addTag() {
    const tag = this.inputValue.trim();
    if (tag && !this.tags.includes(tag)) {
      const newTags = [...this.tags, tag];
      this.tags = newTags;
      this.dispatchEvent(
        new CustomEvent('tags-changed', {
          detail: { tags: newTags },
          bubbles: true,
          composed: true,
        })
      );
    }
    this.inputValue = '';
  }

  private removeTag(index: number) {
    const newTags = this.tags.filter((_, i) => i !== index);
    this.tags = newTags;
    this.dispatchEvent(
      new CustomEvent('tags-changed', {
        detail: { tags: newTags },
        bubbles: true,
        composed: true,
      })
    );
  }

  render() {
    return html`
      <div class="container">
        ${this.tags.map(
          (tag, i) => html`
            <span class="tag">
              ${tag}
              <button
                class="tag-remove"
                @click=${() => this.removeTag(i)}
                aria-label="Remove ${tag}"
              >
                &times;
              </button>
            </span>
          `
        )}
        <input
          type="text"
          .value=${this.inputValue}
          .placeholder=${this.tags.length ? '' : this.placeholder}
          @input=${this.handleInput}
          @keydown=${this.handleKeydown}
          @blur=${() => this.addTag()}
        />
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'tag-input': TagInput;
  }
}
