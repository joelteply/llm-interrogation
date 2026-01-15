import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { EntityMatch } from '../types';

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
      align-items: center;
      font-size: 10px;
      color: var(--text-muted, #6e7681);
    }

    .count-link {
      cursor: pointer;
      text-decoration: underline;
      text-decoration-style: dotted;
    }

    .count-link:hover {
      text-decoration-style: solid;
    }

    .count-link.promoted {
      color: #3fb950;
    }

    .count-link.banned {
      color: #f85149;
    }

    .count-link.active {
      font-weight: 700;
      text-decoration: none;
      background: rgba(255,255,255,0.1);
      padding: 2px 6px;
      border-radius: 3px;
    }

    .legend {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .temp-scale {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-left: 12px;
    }

    .temp-gradient {
      width: 100px;
      height: 10px;
      border-radius: 5px;
      background: linear-gradient(to right,
        #313695,
        #4575b4,
        #74add1,
        #abd9e9,
        #fee090,
        #fdae61,
        #f46d43,
        #d73027
      );
    }

    .temp-label {
      font-size: 10px;
      color: #8b949e;
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

    .hover-popup {
      position: fixed;
      background: #1c2128;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 0;
      max-width: 400px;
      max-height: 300px;
      z-index: 1000;
      box-shadow: 0 8px 24px rgba(0,0,0,0.6);
      font-size: 12px;
      display: flex;
      flex-direction: column;
    }

    .hover-popup-content {
      padding: 12px;
      overflow-y: auto;
      flex: 1;
      min-height: 0;
    }

    .hover-popup-title {
      font-weight: 600;
      color: #58a6ff;
      padding: 10px 12px;
      border-bottom: 1px solid #30363d;
      background: #1c2128;
      position: sticky;
      top: 0;
      flex-shrink: 0;
    }

    .hover-match {
      margin-bottom: 10px;
      padding-bottom: 8px;
      border-bottom: 1px solid #21262d;
    }

    .hover-match:last-child {
      margin-bottom: 0;
      padding-bottom: 0;
      border-bottom: none;
    }

    .hover-match-header {
      display: flex;
      gap: 8px;
      margin-bottom: 4px;
    }

    .hover-match-num {
      color: #6e7681;
      font-weight: 600;
    }

    .hover-match-model {
      color: #7ee787;
      font-weight: 500;
    }

    .hover-match-question {
      color: #8b949e;
      font-style: italic;
      margin-bottom: 4px;
    }

    .hover-match-context {
      color: #c9d1d9;
      line-height: 1.4;
    }

    .hover-refusal {
      color: #f85149;
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid #30363d;
    }

    .hover-refusal-item {
      margin-bottom: 6px;
    }
  `;

  @property({ type: Object })
  entities: Record<string, number> = {};

  @property({ type: Object })
  entityMatches: Record<string, EntityMatch[]> = {};

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

  @state()
  private longPressTimer: number | null = null;

  @state()
  private hoveredWord: string | null = null;

  @state()
  private hoverPos: { x: number; y: number } = { x: 0, y: 0 };

  private hoverHideTimer: number | null = null;

  @property({ type: Array })
  hiddenEntities: string[] = [];  // Banned entities (for count display)

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
      return 0.15;
    }

    // Live threads are hot
    if (this.liveThreads.includes(text)) {
      return 0.85;
    }

    // Percentile-based heat - most items cluster in middle colors
    const values = Object.values(this.entities).sort((a, b) => a - b);
    if (values.length === 0) return 0.5;

    // Find this value's percentile rank
    const rank = values.filter(v => v < count).length;
    const percentile = rank / values.length;

    // Apply sigmoid-like curve to cluster mode in middle
    // This pushes most values toward 0.5 while extremes go to edges
    const curved = 0.5 + 0.4 * Math.tanh((percentile - 0.5) * 3);

    // Clamp to leave room for dead ends (0.15) and live threads (0.85)
    return Math.max(0.2, Math.min(0.75, curved));
  }

  private heatToColor(heat: number): string {
    // RdYlBu diverging palette: blue (cold) → yellow → red (hot)
    // Matches the gradient bar in the legend
    const palette = [
      '#313695',  // 0.0 - deep blue (cold)
      '#4575b4',  // 0.15
      '#74add1',  // 0.3
      '#abd9e9',  // 0.45
      '#fee090',  // 0.55 - yellow (neutral)
      '#fdae61',  // 0.7
      '#f46d43',  // 0.85
      '#d73027',  // 1.0 - red (hot)
    ];

    // Map heat (0-1) to palette index
    const idx = Math.min(Math.floor(heat * (palette.length - 1)), palette.length - 2);
    const t = (heat * (palette.length - 1)) - idx;

    // Interpolate between colors
    const c1 = palette[idx];
    const c2 = palette[idx + 1];

    // Simple hex interpolation
    const r1 = parseInt(c1.slice(1, 3), 16);
    const g1 = parseInt(c1.slice(3, 5), 16);
    const b1 = parseInt(c1.slice(5, 7), 16);
    const r2 = parseInt(c2.slice(1, 3), 16);
    const g2 = parseInt(c2.slice(3, 5), 16);
    const b2 = parseInt(c2.slice(5, 7), 16);

    const r = Math.round(r1 + t * (r2 - r1));
    const g = Math.round(g1 + t * (g2 - g1));
    const b = Math.round(b1 + t * (b2 - b1));

    return `rgb(${r}, ${g}, ${b})`;
  }

  private getColor(count: number, _normalized: number, text: string): string {
    const heat = this.getHeat(text, count);
    return this.heatToColor(heat);
  }

  private handleWordMouseDown(word: string, e: MouseEvent) {
    e.preventDefault();
    this.pressStartTime = Date.now();
    this.selectedWord = word;

    // Start long press timer for ban (500ms)
    this.longPressTimer = window.setTimeout(() => {
      // Long press = BAN (same as right click)
      this.dispatchEvent(new CustomEvent('entity-select', {
        detail: { entity: word, action: 'delete', count: this.entities[word] },
        bubbles: true, composed: true
      }));
      this.selectedWord = null;
      this.pressStartTime = 0;
      this.longPressTimer = null;
    }, 500);
  }

  private handleWordMouseUp(word: string, e: MouseEvent) {
    e.preventDefault();

    // Cancel long press timer
    if (this.longPressTimer) {
      clearTimeout(this.longPressTimer);
      this.longPressTimer = null;
    }

    // If we already handled via long press, skip
    if (this.pressStartTime === 0) return;

    // Determine action based on current state
    const isBanned = this.hiddenEntities.includes(word);
    const isPromoted = this.promotedEntities.includes(word);

    let action: string;
    if (isBanned) {
      // Clicking banned entity unbans it
      action = 'unban';
    } else if (isPromoted) {
      action = 'unpromote';
    } else {
      action = 'promote';
    }

    this.dispatchEvent(new CustomEvent('entity-select', {
      detail: { entity: word, action, count: this.entities[word] },
      bubbles: true, composed: true
    }));

    this.selectedWord = null;
    this.pressStartTime = 0;
  }

  private handleWordRightClick(word: string, e: MouseEvent) {
    e.preventDefault();

    // Right click = BAN
    this.dispatchEvent(new CustomEvent('entity-select', {
      detail: { entity: word, action: 'delete', count: this.entities[word] },
      bubbles: true, composed: true
    }));

    this.selectedWord = null;
  }

  private handleWordHover(word: string, e: MouseEvent) {
    // Cancel any pending hide
    if (this.hoverHideTimer) {
      clearTimeout(this.hoverHideTimer);
      this.hoverHideTimer = null;
    }
    this.hoveredWord = word;
    // Position popup near cursor but ensure it stays on screen
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    this.hoverPos = {
      x: Math.min(rect.left, window.innerWidth - 420),
      y: Math.min(rect.bottom + 8, window.innerHeight - 320)  // Below the word
    };
  }

  private startHoverHideTimer() {
    // Delay before hiding to allow moving mouse to popup
    if (this.hoverHideTimer) clearTimeout(this.hoverHideTimer);
    this.hoverHideTimer = window.setTimeout(() => {
      this.hoveredWord = null;
      this.hoverHideTimer = null;
    }, 400);  // 400ms gives time to move mouse to popup
  }

  private cancelHoverHideTimer() {
    if (this.hoverHideTimer) {
      clearTimeout(this.hoverHideTimer);
      this.hoverHideTimer = null;
    }
  }

  @state()
  private showPromotedList = false;

  @state()
  private showBannedList = false;

  private handlePromotedClick() {
    this.showPromotedList = !this.showPromotedList;
    this.showBannedList = false;
  }

  private handleBannedClick() {
    this.showBannedList = !this.showBannedList;
    this.showPromotedList = false;
  }

  private handleUnpromote(entity: string) {
    this.dispatchEvent(new CustomEvent('entity-select', {
      detail: { entity, action: 'unpromote' },
      bubbles: true, composed: true
    }));
  }

  private handleUnban(entity: string) {
    this.dispatchEvent(new CustomEvent('entity-select', {
      detail: { entity, action: 'unban' },
      bubbles: true, composed: true
    }));
  }

  private renderHoverPopup() {
    if (!this.hoveredWord) return null;

    const matches = this.entityMatches[this.hoveredWord] || [];
    if (matches.length === 0) {
      return html`
        <div
          class="hover-popup"
          style="left: ${this.hoverPos.x}px; top: ${this.hoverPos.y}px;"
          @mouseenter=${this.cancelHoverHideTimer}
          @mouseleave=${this.startHoverHideTimer}
        >
          <div class="hover-popup-title">${this.hoveredWord}: ${this.entities[this.hoveredWord] || 0}x</div>
          <div class="hover-popup-content">
            <div style="color: #6e7681;">No detailed match data available</div>
          </div>
        </div>
      `;
    }

    // Separate refusals from regular matches
    const regularMatches = matches.filter(m => !m.is_refusal);
    const refusals = matches.filter(m => m.is_refusal);

    return html`
      <div
        class="hover-popup"
        style="left: ${this.hoverPos.x}px; top: ${this.hoverPos.y}px;"
        @mouseenter=${this.cancelHoverHideTimer}
        @mouseleave=${this.startHoverHideTimer}
      >
        <div class="hover-popup-title">${this.hoveredWord}: ${matches.length} match${matches.length !== 1 ? 'es' : ''}</div>
        <div class="hover-popup-content">
          ${regularMatches.map((m, i) => html`
            <div class="hover-match">
              <div class="hover-match-header">
                <span class="hover-match-num">${i + 1}.</span>
                <span class="hover-match-model">${m.model.split('/').pop()}</span>
              </div>
              <div class="hover-match-question">"${m.question}"</div>
              <div class="hover-match-context">${m.context || '(no context)'}</div>
            </div>
          `)}
          ${refusals.length > 0 ? html`
            <div class="hover-refusal">
              ${refusals.map(r => html`
                <div class="hover-refusal-item">
                  <strong>${r.model.split('/').pop()}</strong> refused: "${r.context || 'No response'}"
                </div>
              `)}
            </div>
          ` : null}
        </div>
      </div>
    `;
  }

  render() {
    const signalCount = this.words.filter(w => w.count >= this.signalThreshold).length;

    // Filter words based on active filter
    let displayWords = this.words;
    let filterLabel = '';
    if (this.showPromotedList) {
      displayWords = this.words.filter(w => this.promotedEntities.includes(w.text));
      filterLabel = 'PROMOTED';
    } else if (this.showBannedList) {
      // For banned, we need to show them even though they're hidden from main view
      // Create temporary word items for banned entities
      displayWords = this.hiddenEntities.map((text, i) => ({
        text,
        count: 0,
        size: 18,
        x: 50 + (i % 5) * 120,
        y: 30 + Math.floor(i / 5) * 35,
        color: '#f85149',
        heat: 0
      }));
      filterLabel = 'BANNED';
    }

    return html`
      <div class="cloud-container">
        ${filterLabel ? html`
          <div style="position: absolute; top: 8px; left: 12px; font-size: 11px; color: ${filterLabel === 'PROMOTED' ? '#3fb950' : '#f85149'}; font-weight: 600; z-index: 50;">
            ${filterLabel} — click to ${filterLabel === 'PROMOTED' ? 'unfocus' : 'unban'}
          </div>
        ` : ''}
        ${displayWords.length === 0 && !filterLabel
          ? html`<div class="empty">Run probes to see extracted entities</div>`
          : displayWords.length === 0 && filterLabel
          ? html`<div class="empty">No ${filterLabel.toLowerCase()} entities</div>`
          : html`
              <div class="glow-overlay"></div>
              ${displayWords.map(word => {
                const isPromoted = this.promotedEntities.includes(word.text);
                const isBanned = this.hiddenEntities.includes(word.text);
                const isDeadEnd = this.deadEnds.includes(word.text);
                const isLiveThread = this.liveThreads.includes(word.text);
                return html`
                  <span
                    class="word ${word.count >= this.signalThreshold ? 'signal' : ''} ${this.selectedWord === word.text ? 'selected' : ''} ${isPromoted ? 'promoted' : ''} ${isBanned ? 'dead-end' : ''} ${isDeadEnd && !isBanned ? 'dead-end' : ''} ${isLiveThread && !isPromoted ? 'live-thread' : ''}"
                    style="font-size: ${word.size}px; color: ${word.color}; left: ${word.x}px; top: ${word.y}px;"
                    @mouseenter=${(e: MouseEvent) => this.handleWordHover(word.text, e)}
                    @mousedown=${(e: MouseEvent) => this.handleWordMouseDown(word.text, e)}
                    @mouseup=${(e: MouseEvent) => this.handleWordMouseUp(word.text, e)}
                    @contextmenu=${(e: MouseEvent) => this.handleWordRightClick(word.text, e)}
                    @mouseleave=${() => { this.startHoverHideTimer(); this.selectedWord = null; this.pressStartTime = 0; if (this.longPressTimer) { clearTimeout(this.longPressTimer); this.longPressTimer = null; } }}
                  >
                    ${word.text}
                  </span>
                `;
              })}
              ${this.renderHoverPopup()}
            `}

        <div class="stats-bar">
          <div class="legend">
            <span style="color: #6e7681;">Click: focus</span>
            <span style="color: #3fb950;">Again: unfocus</span>
            <span style="color: #f85149;">Hold/Right: BAN</span>
            <div class="temp-scale">
              <span class="temp-label">cold</span>
              <div class="temp-gradient"></div>
              <span class="temp-label">hot</span>
            </div>
          </div>
          <div class="counts">
            ${this.liveThreads.length} live / ${this.deadEnds.length} dead / ${signalCount} signal
            ${this.promotedEntities.length > 0 ? html` / <span class="count-link promoted ${this.showPromotedList ? 'active' : ''}" @click=${this.handlePromotedClick}>${this.promotedEntities.length} promoted</span>` : ''}
            ${this.hiddenEntities.length > 0 ? html` / <span class="count-link banned ${this.showBannedList ? 'active' : ''}" @click=${this.handleBannedClick}>${this.hiddenEntities.length} banned</span>` : ''}
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
