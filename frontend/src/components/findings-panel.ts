import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { probeState, groundTruthState, type ProbeState } from '../state';
import type { Findings } from '../types';
import './word-cloud';
import './concept-graph';

type ViewMode = 'cloud' | 'graph';

@customElement('findings-panel')
export class FindingsPanel extends LitElement {
  static styles = css`
    :host {
      display: block;
      height: 100%;
    }

    .card {
      background: transparent;
      border: none;
      padding: 0;
      height: 100%;
      box-sizing: border-box;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
    }

    h3 {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary, #c9d1d9);
      margin: 0;
    }

    .corpus-size {
      font-size: 12px;
      color: var(--text-muted, #6e7681);
    }

    .section {
      margin-bottom: 16px;
    }

    .section:last-child {
      margin-bottom: 0;
    }

    .section-title {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary, #8b949e);
      margin-bottom: 8px;
    }

    .entities {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .entity {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      background: var(--bg-tertiary, #21262d);
      border-radius: 9999px;
      font-size: 12px;
    }

    .entity-name {
      color: var(--text-primary, #c9d1d9);
    }

    .entity-count {
      color: var(--text-muted, #6e7681);
      font-size: 10px;
    }

    .entity.consistent {
      background: rgba(63, 185, 80, 0.15);
      border: 1px solid rgba(63, 185, 80, 0.3);
    }

    .entity.consistent .entity-name {
      color: var(--accent-green, #3fb950);
    }

    .entity.warm {
      background: rgba(210, 153, 34, 0.15);
      border: 1px solid rgba(210, 153, 34, 0.3);
    }

    .entity.warm .entity-name {
      color: var(--accent-yellow, #d29922);
    }

    .entity.noise {
      opacity: 0.6;
    }

    .warmth-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-yellow, #d29922);
    }

    .empty {
      padding: 24px;
      text-align: center;
      color: var(--text-secondary, #8b949e);
      font-size: 13px;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      padding: 12px;
      background: var(--bg-primary, #0d1117);
      border-radius: 6px;
      margin-bottom: 16px;
    }

    .stat {
      text-align: center;
    }

    .stat-value {
      font-size: 20px;
      font-weight: 600;
      color: var(--text-primary, #c9d1d9);
    }

    .stat-label {
      font-size: 11px;
      color: var(--text-muted, #6e7681);
      margin-top: 2px;
    }

    .model-breakdown {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .model-row {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px;
      background: var(--bg-primary, #0d1117);
      border-radius: 6px;
    }

    .model-name {
      font-size: 12px;
      color: var(--text-secondary, #8b949e);
      min-width: 100px;
    }

    .model-entities {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      flex: 1;
    }

    .model-entity {
      padding: 2px 6px;
      background: var(--bg-tertiary, #21262d);
      border-radius: 4px;
      font-size: 10px;
      color: var(--text-muted, #6e7681);
    }

    .threshold-control {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
    }

    .threshold-control label {
      font-size: 12px;
      color: var(--text-secondary, #8b949e);
    }

    .threshold-control input {
      width: 60px;
      padding: 4px 8px;
      background: var(--bg-primary, #0d1117);
      border: 1px solid var(--border-default, #30363d);
      border-radius: 4px;
      color: var(--text-primary, #c9d1d9);
      font-size: 12px;
      text-align: center;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
      to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    }

    .hidden-count {
      font-size: 11px;
      color: var(--text-muted, #6e7681);
      margin-top: 8px;
      cursor: pointer;
    }

    .hidden-count:hover {
      color: var(--accent-blue, #58a6ff);
      text-decoration: underline;
    }

    .view-toggle {
      position: absolute;
      top: 8px;
      right: 45px;
      display: flex;
      gap: 2px;
      background: rgba(33, 38, 45, 0.9);
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 2px;
      z-index: 10;
    }

    .view-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 24px;
      background: transparent;
      border: none;
      border-radius: 4px;
      color: #6e7681;
      cursor: pointer;
      font-size: 12px;
      transition: all 150ms;
    }

    .view-btn:hover {
      color: #c9d1d9;
      background: #30363d;
    }

    .view-btn.active {
      background: #58a6ff;
      color: white;
    }

    .view-container {
      height: 100%;
      position: relative;
      box-sizing: border-box;
    }

    concept-graph,
    word-cloud {
      height: 100%;
      display: block;
    }
  `;

  @property({ type: Object })
  findings: Findings | null = null;

  @state()
  private _probeState: ProbeState = probeState.get();

  @state()
  private _groundTruth: string[] = groundTruthState.get().facts;

  @state()
  private consistentThreshold = 3;

  @state()
  private lastAction: { entity: string; action: string } | null = null;

  @state()
  private viewMode: ViewMode = 'cloud';

  private _unsubscribes: Array<() => void> = [];

  connectedCallback() {
    super.connectedCallback();
    this._unsubscribes.push(
      probeState.subscribe((s) => {
        this._probeState = s;
        if (s.findings) {
          this.findings = s.findings;
        }
      }),
      groundTruthState.subscribe((s) => {
        this._groundTruth = s.facts;
      })
    );
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribes.forEach((unsub) => unsub());
  }

  private getSortedEntities(): Array<[string, number]> {
    if (!this.findings?.entities) return [];
    const hidden = new Set(this._probeState.hiddenEntities);
    return Object.entries(this.findings.entities)
      .filter(([entity]) => !hidden.has(entity))
      .sort((a, b) => b[1] - a[1]);
  }

  private getFilteredEntities(): Record<string, number> {
    if (!this.findings?.entities) return {};
    const hidden = new Set(this._probeState.hiddenEntities);
    const filtered: Record<string, number> = {};
    for (const [entity, count] of Object.entries(this.findings.entities)) {
      if (!hidden.has(entity)) {
        filtered[entity] = count;
      }
    }
    return filtered;
  }

  private handleEntitySelect(e: CustomEvent) {
    const { entity, action } = e.detail;
    this.lastAction = { entity, action };

    if (action === 'delete') {
      probeState.update(s => ({
        ...s,
        hiddenEntities: [...s.hiddenEntities, entity],
        promotedEntities: s.promotedEntities.filter(e => e !== entity)  // Remove from positives if was there
      }));
    } else if (action === 'demote') {
      // Remove from promoted if it was there
      probeState.update(s => ({
        ...s,
        promotedEntities: s.promotedEntities.filter(e => e !== entity)
      }));
    } else if (action === 'promote') {
      // Add to promoted entities (positives to focus on)
      probeState.update(s => ({
        ...s,
        promotedEntities: s.promotedEntities.includes(entity)
          ? s.promotedEntities
          : [...s.promotedEntities, entity],
        hiddenEntities: s.hiddenEntities.filter(e => e !== entity)  // Remove from negatives if was there
      }));
      // Also emit drill event for any UI that wants it
      this.dispatchEvent(new CustomEvent('drill-entity', {
        detail: { entity },
        bubbles: true, composed: true
      }));
    }

    // Clear action indicator after a moment
    setTimeout(() => { this.lastAction = null; }, 1500);
  }

  private isWarm(entity: string): boolean {
    if (!this._groundTruth.length) return false;
    const lowerEntity = entity.toLowerCase();
    return this._groundTruth.some(
      (gt) =>
        lowerEntity.includes(gt.toLowerCase()) ||
        gt.toLowerCase().includes(lowerEntity)
    );
  }

  private getWarmthScore(entity: string): number {
    return this.findings?.warmth_scores?.[entity] || 0;
  }

  private getCooccurrences(): Array<{ entities: string[]; count: number }> {
    // Compute co-occurrences from responses - entities that appear in the same response
    const responses = this._probeState.responses;
    const cooccurrenceMap = new Map<string, number>();
    const entitySet = new Set(Object.keys(this.getFilteredEntities()));

    for (const response of responses) {
      // Get entities mentioned in this response (simplified: check if entity appears in response text)
      const mentionedEntities = Array.from(entitySet).filter(entity =>
        response.response?.toLowerCase().includes(entity.toLowerCase())
      );

      // Create co-occurrence pairs
      if (mentionedEntities.length >= 2) {
        // Store all entities that co-occurred
        const sorted = [...mentionedEntities].sort();
        const key = sorted.join('|||');
        cooccurrenceMap.set(key, (cooccurrenceMap.get(key) || 0) + 1);
      }
    }

    return Array.from(cooccurrenceMap.entries())
      .map(([key, count]) => ({
        entities: key.split('|||'),
        count
      }))
      .sort((a, b) => b.count - a.count);
  }

  render() {
    const entities = this.getSortedEntities();
    const consistent = entities.filter(([, count]) => count >= this.consistentThreshold);
    const noise = entities.filter(([, count]) => count < this.consistentThreshold);

    if (!this.findings && entities.length === 0) {
      return html`
        <div class="card">
          <div class="header">
            <h3>Findings</h3>
          </div>
          <div class="view-container">
            <div class="view-toggle">
              <button
                class="view-btn ${this.viewMode === 'cloud' ? 'active' : ''}"
                @click=${() => this.viewMode = 'cloud'}
                title="Word Cloud"
              >☁</button>
              <button
                class="view-btn ${this.viewMode === 'graph' ? 'active' : ''}"
                @click=${() => this.viewMode = 'graph'}
                title="Concept Graph"
              >◉</button>
            </div>
            ${this.viewMode === 'cloud' ? html`
              <word-cloud .entities=${{}} .signalThreshold=${this.consistentThreshold} .promotedEntities=${this._probeState.promotedEntities}></word-cloud>
            ` : html`
              <concept-graph .entities=${{}} .cooccurrences=${[]}></concept-graph>
            `}
          </div>
          <div class="empty" style="padding: 16px;">
            Extracted entities will appear here as probes complete.
            Consistent entities (appearing multiple times) suggest real signal.
          </div>
        </div>
      `;
    }

    const uniqueEntities = entities.length;
    const totalMentions = entities.reduce((sum, [, count]) => sum + count, 0);
    const refusalRate = this.findings?.refusal_rate
      ? Math.round(this.findings.refusal_rate * 100)
      : 0;

    return html`
      <div class="card" style="position: relative; height: 100%;">
        <div class="view-container">
          <div class="view-toggle">
            <button
              class="view-btn ${this.viewMode === 'cloud' ? 'active' : ''}"
              @click=${() => this.viewMode = 'cloud'}
              title="Word Cloud"
            >☁</button>
            <button
              class="view-btn ${this.viewMode === 'graph' ? 'active' : ''}"
              @click=${() => this.viewMode = 'graph'}
              title="Concept Graph"
            >◉</button>
          </div>

          ${this.viewMode === 'cloud' ? html`
            <word-cloud
              .entities=${this.getFilteredEntities()}
              .signalThreshold=${this.consistentThreshold}
              .promotedEntities=${this._probeState.promotedEntities}
              @entity-select=${this.handleEntitySelect}
            ></word-cloud>
          ` : html`
            <concept-graph
              .entities=${this.getFilteredEntities()}
              .cooccurrences=${this.getCooccurrences()}
              @node-select=${(e: CustomEvent) => this.handleEntitySelect(new CustomEvent('entity-select', {
                detail: { entity: e.detail.entity, action: 'promote' }
              }))}
            ></concept-graph>
          `}
        </div>

        ${this.lastAction ? html`
          <div style="
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            padding: 10px 20px;
            background: ${this.lastAction.action === 'delete' ? 'rgba(248, 81, 73, 0.95)' : this.lastAction.action === 'demote' ? 'rgba(210, 153, 34, 0.95)' : 'rgba(63, 185, 80, 0.95)'};
            border-radius: 8px;
            color: white;
            font-weight: 600;
            font-size: 13px;
            z-index: 100;
          ">
            ${this.lastAction.action === 'delete' ? '🗑️' : this.lastAction.action === 'demote' ? '👇' : '🎯'} ${this.lastAction.entity}
          </div>
        ` : null}

        <!-- Entity list collapsed by default -->
        ${false && consistent.length
          ? html`
              <div class="section">
                <div class="section-title">
                  Consistent (${consistent.length} entities with ${this.consistentThreshold}+ occurrences)
                </div>
                <div class="entities">
                  ${consistent.map(
                    ([entity, count]) => html`
                      <span class="entity consistent ${this.isWarm(entity) ? 'warm' : ''}">
                        ${this.isWarm(entity)
                          ? html`<span class="warmth-indicator" title="Matches ground truth"></span>`
                          : null}
                        <span class="entity-name">${entity}</span>
                        <span class="entity-count">${count}x</span>
                      </span>
                    `
                  )}
                </div>
              </div>
            `
          : null}

        ${false && noise.length
          ? html`
              <div class="section">
                <div class="section-title">
                  Noise (${noise.length} entities with &lt;${this.consistentThreshold} occurrences)
                </div>
                <div class="entities">
                  ${noise.slice(0, 30).map(
                    ([entity, count]) => html`
                      <span class="entity noise">
                        <span class="entity-name">${entity}</span>
                        <span class="entity-count">${count}x</span>
                      </span>
                    `
                  )}
                  ${noise.length > 30
                    ? html`<span class="entity noise">+${noise.length - 30} more</span>`
                    : null}
                </div>
              </div>
            `
          : null}

        ${false && this.findings?.by_model && Object.keys(this.findings!.by_model!).length > 1
          ? html`
              <div class="section">
                <div class="section-title">By Model</div>
                <div class="model-breakdown">
                  ${Object.entries(this.findings!.by_model!).map(
                    ([model, modelEntities]) => {
                      const sorted = Object.entries(modelEntities)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10);
                      const displayName = model.split('/').pop() || model;
                      return html`
                        <div class="model-row">
                          <span class="model-name">${displayName}</span>
                          <div class="model-entities">
                            ${sorted.map(
                              ([e, c]) =>
                                html`<span class="model-entity">${e} (${c})</span>`
                            )}
                          </div>
                        </div>
                      `;
                    }
                  )}
                </div>
              </div>
            `
          : null}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'findings-panel': FindingsPanel;
  }
}
