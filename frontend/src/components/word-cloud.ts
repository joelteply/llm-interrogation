import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface WordItem {
  text: string;
  count: number;
  size: number;
  x: number;
  y: number;
  color: string;
}

@customElement('word-cloud')
export class WordCloud extends LitElement {
  @property({ type: Array })
  promotedEntities: string[] = [];

  static styles = css`
    :host {
      display: block;
    }

    .cloud-container {
      position: relative;
      width: 100%;
      height: var(--height, 350px);
      background: linear-gradient(135deg, rgba(13, 17, 23, 0.9) 0%, rgba(22, 27, 34, 0.9) 100%);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 12px;
      overflow: hidden;
    }

    .word {
      position: absolute;
      cursor: pointer;
      transition: all 0.15s ease;
      text-shadow: 0 0 15px currentColor;
      font-weight: 600;
      white-space: nowrap;
      user-select: none;
    }

    .word:hover {
      transform: scale(1.2);
      z-index: 100;
      text-shadow: 0 0 25px currentColor, 0 0 40px currentColor;
      filter: brightness(1.2);
    }

    .word:active {
      transform: scale(1.05);
    }

    .word.selected {
      outline: 2px solid white;
      outline-offset: 2px;
    }

    .word.signal {
      animation: pulse 2s ease-in-out infinite;
    }

    .word.promoted {
      outline: 2px solid #3fb950;
      outline-offset: 2px;
      animation: none;
      background: rgba(63, 185, 80, 0.15);
      padding: 2px 4px;
      border-radius: 3px;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }

    .glow-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: radial-gradient(ellipse at center, rgba(88, 166, 255, 0.1) 0%, transparent 70%);
      pointer-events: none;
    }

    .stats-bar {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      padding: 6px 12px;
      background: rgba(0, 0, 0, 0.6);
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: var(--text-muted, #6e7681);
    }

    .legend {
      display: flex;
      gap: 12px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .legend-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
    }

    .empty {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: var(--text-secondary, #8b949e);
      font-size: 14px;
    }
  `;

  @property({ type: Object })
  entities: Record<string, number> = {};

  @property({ type: Number })
  signalThreshold = 3;

  @state()
  private words: WordItem[] = [];

  @state()
  private containerWidth = 0;

  @state()
  private containerHeight = 0;

  @state()
  private selectedWord: string | null = null;

  @state()
  private pressStartTime = 0;

  private resizeObserver: ResizeObserver | null = null;

  private colors = {
    hot: ['#ff6b6b', '#ee5a5a', '#ff8787', '#fa5252'],      // High frequency - red/orange
    warm: ['#ffd43b', '#fab005', '#f59f00', '#f08c00'],     // Medium - yellow
    cool: ['#69db7c', '#51cf66', '#40c057', '#37b24d'],     // Good signal - green
    cold: ['#748ffc', '#5c7cfa', '#4c6ef5', '#4263eb'],     // Low frequency - blue
    muted: ['#868e96', '#adb5bd', '#ced4da', '#dee2e6'],    // Noise - gray
  };

  updated(changedProperties: Map<string, unknown>) {
    if (changedProperties.has('entities') || changedProperties.has('signalThreshold')) {
      this.calculateLayout();
    }
  }

  firstUpdated() {
    const container = this.shadowRoot?.querySelector('.cloud-container');
    if (container) {
      this.resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 50 && height > 50) {
            this.containerWidth = width;
            this.containerHeight = height - 35; // Stats bar height
            this.calculateLayout();
          }
        }
      });
      this.resizeObserver.observe(container);
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.resizeObserver?.disconnect();
  }

  private calculateLayout() {
    if (this.containerWidth === 0 || this.containerHeight === 0) return;

    const entries = Object.entries(this.entities);
    if (entries.length === 0) {
      this.words = [];
      return;
    }

    entries.sort((a, b) => b[1] - a[1]);

    const maxCount = entries[0][1];
    const minCount = Math.min(...entries.map(e => e[1]));

    const topEntries = entries
      .filter(([text]) => !this.isBoringWord(text))
      .slice(0, 40);

    const words: WordItem[] = [];
    const placed: Array<{ x: number; y: number; w: number; h: number }> = [];

    // Center of container
    const centerX = this.containerWidth / 2;
    const centerY = this.containerHeight / 2;

    for (const [text, count] of topEntries) {
      const normalizedCount = (count - minCount) / (maxCount - minCount || 1);
      const size = Math.floor(14 + normalizedCount * 32);
      const color = this.getColor(count, normalizedCount, text);

      const wordWidth = text.length * size * 0.55;
      const wordHeight = size * 1.1;

      const pos = this.findPosition(placed, wordWidth, wordHeight, centerX, centerY);

      if (pos) {
        placed.push({ x: pos.x, y: pos.y, w: wordWidth, h: wordHeight });
        words.push({
          text,
          count,
          size,
          x: pos.x,
          y: pos.y,
          color,
        });
      }
    }

    this.words = words;
  }

  private findPosition(
    placed: Array<{ x: number; y: number; w: number; h: number }>,
    width: number,
    height: number,
    centerX: number,
    centerY: number
  ): { x: number; y: number } | null {
    let angle = placed.length * 0.5;
    let radius = 0;

    // Stretch factor to fill width (container is typically wider than tall)
    const stretchX = (this.containerWidth / this.containerHeight) * 1.3;

    for (let i = 0; i < 2000; i++) {
      // Elliptical spiral - stretch to match container aspect ratio
      const x = centerX + Math.cos(angle) * radius * stretchX - width / 2;
      const y = centerY + Math.sin(angle) * radius - height / 2;

      // Check bounds
      if (x >= 5 && x + width <= this.containerWidth - 5 &&
          y >= 5 && y + height <= this.containerHeight - 5) {

        // Check collisions
        let collision = false;
        for (const p of placed) {
          if (!(x + width + 3 < p.x || x > p.x + p.w + 3 ||
                y + height + 3 < p.y || y > p.y + p.h + 3)) {
            collision = true;
            break;
          }
        }

        if (!collision) {
          return { x, y };
        }
      }

      angle += 0.15;
      radius += 2;
    }

    return null;
  }

  private isBoringWord(text: string): boolean {
    const boring = ['Based', 'Here', 'For', 'The', 'This', 'That', 'With', 'From', 'However',
                    'Therefore', 'While', 'After', 'Before', 'Without', 'About', 'Some',
                    'Public', 'His', 'Her', 'Their', 'Would', 'Could', 'Should', 'Please',
                    'Keep', 'Even', 'Also', 'Just', 'Still', 'Yet', 'Already', 'Always'];
    return boring.includes(text);
  }

  private getColor(count: number, _normalized: number, text: string): string {
    const hash = text.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    if (count >= this.signalThreshold * 3) {
      return this.colors.hot[hash % this.colors.hot.length];
    } else if (count >= this.signalThreshold * 2) {
      return this.colors.warm[hash % this.colors.warm.length];
    } else if (count >= this.signalThreshold) {
      return this.colors.cool[hash % this.colors.cool.length];
    } else if (count >= 2) {
      return this.colors.cold[hash % this.colors.cold.length];
    } else {
      return this.colors.muted[hash % this.colors.muted.length];
    }
  }

  private handleWordMouseDown(word: string, e: MouseEvent) {
    e.preventDefault();
    this.pressStartTime = Date.now();
    this.selectedWord = word;
  }

  private handleWordMouseUp(word: string, e: MouseEvent) {
    e.preventDefault();
    const pressDuration = Date.now() - this.pressStartTime;

    if (pressDuration < 300) {
      this.dispatchEvent(new CustomEvent('entity-select', {
        detail: { entity: word, action: 'promote', count: this.entities[word] },
        bubbles: true, composed: true
      }));
    } else if (pressDuration < 800) {
      this.dispatchEvent(new CustomEvent('entity-select', {
        detail: { entity: word, action: 'demote', count: this.entities[word] },
        bubbles: true, composed: true
      }));
    } else {
      this.dispatchEvent(new CustomEvent('entity-select', {
        detail: { entity: word, action: 'delete', count: this.entities[word] },
        bubbles: true, composed: true
      }));
    }

    this.selectedWord = null;
    this.pressStartTime = 0;
  }

  render() {
    const signalCount = this.words.filter(w => w.count >= this.signalThreshold).length;
    const noiseCount = this.words.filter(w => w.count < this.signalThreshold).length;

    return html`
      <div class="cloud-container">
        ${this.words.length === 0
          ? html`<div class="empty">Run probes to see extracted entities</div>`
          : html`
              <div class="glow-overlay"></div>
              ${this.words.map(word => {
                const isPromoted = this.promotedEntities.includes(word.text);
                return html`
                  <span
                    class="word ${word.count >= this.signalThreshold ? 'signal' : ''} ${this.selectedWord === word.text ? 'selected' : ''} ${isPromoted ? 'promoted' : ''}"
                    style="font-size: ${word.size}px; color: ${word.color}; left: ${word.x}px; top: ${word.y}px;"
                    title="${word.text}: ${word.count}x${isPromoted ? ' [PROMOTED]' : ''}"
                    @mousedown=${(e: MouseEvent) => this.handleWordMouseDown(word.text, e)}
                    @mouseup=${(e: MouseEvent) => this.handleWordMouseUp(word.text, e)}
                    @mouseleave=${() => { this.selectedWord = null; this.pressStartTime = 0; }}
                  >
                    ${word.text}
                  </span>
                `;
              })}
            `}

        <div class="stats-bar">
          <div class="legend">
            <div class="legend-item">
              <span class="legend-dot" style="background: #ff6b6b"></span>
              Hot (${this.signalThreshold * 3}+)
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: #ffd43b"></span>
              Warm
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: #69db7c"></span>
              Signal (${this.signalThreshold}+)
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: #748ffc"></span>
              Emerging
            </div>
          </div>
          <div>
            ${signalCount} signals / ${noiseCount} noise
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'word-cloud': WordCloud;
  }
}
