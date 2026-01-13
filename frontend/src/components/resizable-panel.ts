import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

@customElement('resizable-panel')
export class ResizablePanel extends LitElement {
  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }

    .top-panel {
      overflow: hidden;
      position: relative;
      min-height: 100px;
    }

    .top-panel ::slotted(*) {
      height: 100%;
      box-sizing: border-box;
    }

    .top-panel.maximized {
      position: fixed;
      top: 65px;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 100;
      background: #0d1117;
    }

    .panel-controls {
      position: absolute;
      top: 8px;
      right: 8px;
      display: flex;
      gap: 4px;
      z-index: 10;
    }

    .panel-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      background: rgba(33, 38, 45, 0.9);
      border: 1px solid #30363d;
      border-radius: 4px;
      color: #8b949e;
      cursor: pointer;
      font-size: 14px;
      transition: all 150ms;
    }

    .panel-btn:hover {
      background: #30363d;
      color: #f0f6fc;
      border-color: #58a6ff;
    }

    .divider {
      height: 6px;
      background: #21262d;
      cursor: ns-resize;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 150ms;
      flex-shrink: 0;
    }

    .divider:hover,
    .divider.dragging {
      background: #58a6ff;
    }

    .divider-handle {
      width: 40px;
      height: 3px;
      background: #484f58;
      border-radius: 2px;
    }

    .divider:hover .divider-handle,
    .divider.dragging .divider-handle {
      background: #f0f6fc;
    }

    .divider.hidden {
      display: none;
    }

    .bottom-panel {
      flex: 1;
      overflow: auto;
      min-height: 200px;
    }

    .bottom-panel.hidden {
      display: none;
    }
  `;

  @property({ type: Number })
  initialHeight = 300;

  @property({ type: Number })
  minHeight = 100;

  @property({ type: Number })
  maxHeight = 800;

  @state()
  private topHeight = 300;

  @state()
  private isMaximized = false;

  @state()
  private isDragging = false;

  @state()
  private preMaximizeHeight = 300;

  connectedCallback() {
    super.connectedCallback();
    this.topHeight = this.initialHeight;
    this.preMaximizeHeight = this.initialHeight;
  }

  private handleDragStart = (e: MouseEvent) => {
    e.preventDefault();
    this.isDragging = true;
    document.addEventListener('mousemove', this.handleDrag);
    document.addEventListener('mouseup', this.handleDragEnd);
  };

  private handleDrag = (e: MouseEvent) => {
    if (!this.isDragging) return;

    const container = this.shadowRoot?.querySelector('.top-panel');
    if (!container) return;

    const rect = this.getBoundingClientRect();
    const newHeight = e.clientY - rect.top;

    this.topHeight = Math.max(this.minHeight, Math.min(this.maxHeight, newHeight));
  };

  private handleDragEnd = () => {
    this.isDragging = false;
    document.removeEventListener('mousemove', this.handleDrag);
    document.removeEventListener('mouseup', this.handleDragEnd);
  };

  private toggleMaximize() {
    if (this.isMaximized) {
      this.topHeight = this.preMaximizeHeight;
    } else {
      this.preMaximizeHeight = this.topHeight;
    }
    this.isMaximized = !this.isMaximized;

    this.dispatchEvent(new CustomEvent('maximize-change', {
      detail: { maximized: this.isMaximized },
      bubbles: true,
      composed: true
    }));
  }

  render() {
    return html`
      <div
        class="top-panel ${this.isMaximized ? 'maximized' : ''}"
        style="${this.isMaximized ? '' : `height: ${this.topHeight}px;`}"
      >
        <div class="panel-controls">
          <button
            class="panel-btn"
            @click=${this.toggleMaximize}
            title="${this.isMaximized ? 'Restore' : 'Maximize'}"
          >
            ${this.isMaximized ? '⊙' : '⤢'}
          </button>
        </div>
        <slot name="top"></slot>
      </div>

      <div
        class="divider ${this.isDragging ? 'dragging' : ''} ${this.isMaximized ? 'hidden' : ''}"
        @mousedown=${this.handleDragStart}
      >
        <div class="divider-handle"></div>
      </div>

      <div class="bottom-panel ${this.isMaximized ? 'hidden' : ''}">
        <slot name="bottom"></slot>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'resizable-panel': ResizablePanel;
  }
}
