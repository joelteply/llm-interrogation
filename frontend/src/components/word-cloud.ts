import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface WordItem {
  text: string;
  count: number;
  size: number;
  x: number;
  y: number;
  color: string;
  heat: number;  // 0 = cold/dead, 1 = hot/new
}

@customElement('word-cloud')
export class WordCloud extends LitElement {
  @property({ type: Array })
  promotedEntities: string[] = [];

  @property({ type: Array })
  deadEnds: string[] = [];  // Entities that only lead to noise/public info

  @property({ type: Array })
  liveThreads: string[] = [];  // Entities with high-quality connections

  @property({ type: Object })
  entityHeat: Record<string, number> = {};  // 0-1 heat score per entity

  static styles = css`
    :host {
      display: block;
      height: 100%;
    }

    .cloud-container {
      position: relative;
      width: 100%;
      height: 100%;
      min-height: 150px;
      background: linear-gradient(135deg, rgba(13, 17, 23, 0.9) 0%, rgba(22, 27, 34, 0.9) 100%);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 12px;
      overflow: hidden;
      box-sizing: border-box;
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

    .word.dead-end {
      opacity: 0.35;
      text-decoration: line-through;
      text-decoration-color: rgba(248, 81, 73, 0.6);
    }

    .word.dead-end:hover {
      opacity: 0.6;
    }

    .word.live-thread {
      text-shadow: 0 0 20px #3fb950, 0 0 30px #3fb950;
      animation: liveGlow 1.5s ease-in-out infinite;
    }

    @keyframes liveGlow {
      0%, 100% { text-shadow: 0 0 20px currentColor, 0 0 30px #3fb950; }
      50% { text-shadow: 0 0 25px currentColor, 0 0 40px #3fb950, 0 0 50px #3fb950; }
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
            this.containerHeight = height - 30; // Stats bar height
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

    // Scale font size based on container area (reference: 800x400)
    const baseArea = 800 * 400;
    const actualArea = this.containerWidth * this.containerHeight;
    const scaleFactor = Math.sqrt(actualArea / baseArea);

    for (const [text, count] of topEntries) {
      const normalizedCount = (count - minCount) / (maxCount - minCount || 1);
      const baseSize = 14 + normalizedCount * 32;
      const size = Math.floor(baseSize * scaleFactor);
      const color = this.getColor(count, normalizedCount, text);

      const wordWidth = text.length * size * 0.55;
      const wordHeight = size * 1.1;

      const pos = this.findPosition(placed, wordWidth, wordHeight, centerX, centerY);

      if (pos) {
        placed.push({ x: pos.x, y: pos.y, w: wordWidth, h: wordHeight });
        const heat = this.getHeat(text, count);
        words.push({
          text,
          count,
          size,
          x: pos.x,
          y: pos.y,
          color,
          heat,
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
    // Inscribed ellipse: a = W/2, b = H/2 (with padding)
    const padding = 20;
    const a = this.containerWidth / 2 - padding;
    const b = this.containerHeight / 2 - padding;

    // Force words to spread: each word assigned to a ring, searches within that ring
    const totalWords = Math.max(Object.keys(this.entities).length, 1);
    const startT = Math.sqrt(placed.length / totalWords); // sqrt gives better distribution
    let t = startT;
    let angle = placed.length * 2.8;

    for (let i = 0; i < 2000; i++) {
      // Parametric ellipse scaled by t
      const x = centerX + t * a * Math.cos(angle) - width / 2;
      const y = centerY + t * b * Math.sin(angle) - height / 2;

      // Check bounds
      if (x >= 5 && x + width <= this.containerWidth - 5 &&
          y >= 5 && y + height <= this.containerHeight - 5) {

        // Check collisions
        let collision = false;
        for (const p of placed) {
          if (!(x + width + 6 < p.x || x > p.x + p.w + 6 ||
                y + height + 6 < p.y || y > p.y + p.h + 6)) {
            collision = true;
            break;
          }
        }

        if (!collision) {
          return { x, y };
        }
      }

      // Search around ring first (angle), then expand outward
      angle += 0.25;
      if (i % 15 === 0) t += 0.03;
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

  private getHeat(text: string, count: number): number {
    // Check explicit heat score first
    if (this.entityHeat[text] !== undefined) {
      return this.entityHeat[text];
    }

    // Dead ends are cold
    if (this.deadEnds.includes(text)) {
      return 0.1;
    }

    // Live threads are hot
    if (this.liveThreads.includes(text)) {
      return 0.9;
    }

    // Otherwise base on frequency relative to threshold
    const values = Object.values(this.entities);
    const maxCount = values.length > 0 ? Math.max(...values) : 1;

    // High frequency + above threshold = warmer
    if (count >= this.signalThreshold * 3) return 0.85;
    if (count >= this.signalThreshold * 2) return 0.7;
    if (count >= this.signalThreshold) return 0.5;
    if (count >= 2) return 0.3;
    return 0.15;
  }

  private heatToColor(heat: number): string {
    // Red (hot) → Orange → Yellow → Cyan → Blue (cold)
    // heat: 1 = red, 0 = blue

    if (heat > 0.8) {
      // Hot: bright red/orange
      return `hsl(${10 + (1 - heat) * 30}, 90%, 60%)`;
    } else if (heat > 0.6) {
      // Warm: orange/yellow
      return `hsl(${40 + (0.8 - heat) * 40}, 85%, 55%)`;
    } else if (heat > 0.4) {
      // Medium: yellow/green
      return `hsl(${80 + (0.6 - heat) * 60}, 75%, 50%)`;
    } else if (heat > 0.2) {
      // Cool: cyan/light blue
      return `hsl(${180 + (0.4 - heat) * 40}, 70%, 55%)`;
    } else {
      // Cold: blue/gray
      return `hsl(${220 + (0.2 - heat) * 20}, 50%, 50%)`;
    }
  }

  private getColor(count: number, _normalized: number, text: string): string {
    const heat = this.getHeat(text, count);
    return this.heatToColor(heat);
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
                const isDeadEnd = this.deadEnds.includes(word.text);
                const isLiveThread = this.liveThreads.includes(word.text);
                const statusLabel = isPromoted ? ' [PROMOTED]' : isDeadEnd ? ' [DEAD END]' : isLiveThread ? ' [LIVE]' : '';
                return html`
                  <span
                    class="word ${word.count >= this.signalThreshold ? 'signal' : ''} ${this.selectedWord === word.text ? 'selected' : ''} ${isPromoted ? 'promoted' : ''} ${isDeadEnd ? 'dead-end' : ''} ${isLiveThread && !isPromoted ? 'live-thread' : ''}"
                    style="font-size: ${word.size}px; color: ${word.color}; left: ${word.x}px; top: ${word.y}px;"
                    title="${word.text}: ${word.count}x${statusLabel}"
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
              <span class="legend-dot" style="background: hsl(15, 90%, 60%)"></span>
              Hot
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: hsl(50, 85%, 55%)"></span>
              Warm
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: hsl(120, 75%, 50%)"></span>
              Cool
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: hsl(220, 50%, 50%); opacity: 0.6"></span>
              Cold
            </div>
          </div>
          <div>
            ${this.liveThreads.length} live / ${this.deadEnds.length} dead / ${signalCount} signal
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
