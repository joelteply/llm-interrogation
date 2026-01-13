import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface GraphNode {
  id: string;
  count: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  category?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

@customElement('concept-graph')
export class ConceptGraph extends LitElement {
  static styles = css`
    :host {
      display: block;
    }

    .graph-container {
      position: relative;
      width: 100%;
      height: 100%;
      background: linear-gradient(135deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.95) 100%);
      border-radius: 8px;
      overflow: hidden;
    }

    canvas {
      width: 100%;
      height: 100%;
    }

    .legend {
      position: absolute;
      bottom: 8px;
      left: 8px;
      display: flex;
      gap: 12px;
      font-size: 10px;
      color: #6e7681;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .legend-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .empty {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: #6e7681;
      font-size: 14px;
    }

    .stats {
      position: absolute;
      top: 8px;
      right: 8px;
      font-size: 10px;
      color: #6e7681;
      background: rgba(0,0,0,0.5);
      padding: 4px 8px;
      border-radius: 4px;
    }
  `;

  @property({ type: Object })
  entities: Record<string, number> = {};

  @property({ type: Array })
  cooccurrences: Array<{ entities: string[]; count: number }> = [];

  @state()
  private nodes: GraphNode[] = [];

  @state()
  private edges: GraphEdge[] = [];

  @state()
  private width = 800;

  @state()
  private height = 400;

  @state()
  private hoveredNode: string | null = null;

  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private animationFrame: number | null = null;
  private resizeObserver: ResizeObserver | null = null;

  private categoryColors: Record<string, string> = {
    person: '#58a6ff',
    company: '#3fb950',
    date: '#f0883e',
    tech: '#a371f7',
    place: '#f85149',
    default: '#8b949e'
  };

  updated(changed: Map<string, unknown>) {
    if (changed.has('entities') || changed.has('cooccurrences')) {
      this.buildGraph();
    }
  }

  firstUpdated() {
    this.canvas = this.shadowRoot?.querySelector('canvas') || null;
    if (this.canvas) {
      this.ctx = this.canvas.getContext('2d');

      this.resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) {
            this.width = width;
            this.height = height;
            this.canvas!.width = width * window.devicePixelRatio;
            this.canvas!.height = height * window.devicePixelRatio;
            this.ctx!.scale(window.devicePixelRatio, window.devicePixelRatio);
            this.buildGraph();
          }
        }
      });
      this.resizeObserver.observe(this.canvas.parentElement!);

      this.canvas.addEventListener('mousemove', this.handleMouseMove);
      this.canvas.addEventListener('click', this.handleClick);
    }
    this.startSimulation();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    this.resizeObserver?.disconnect();
  }

  private handleMouseMove = (e: MouseEvent) => {
    const rect = this.canvas!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    let found: string | null = null;
    for (const node of this.nodes) {
      const size = this.getNodeSize(node.count);
      const dx = node.x - x;
      const dy = node.y - y;
      if (Math.sqrt(dx * dx + dy * dy) < size + 5) {
        found = node.id;
        break;
      }
    }
    this.hoveredNode = found;
    this.canvas!.style.cursor = found ? 'pointer' : 'default';
  };

  private handleClick = (e: MouseEvent) => {
    if (this.hoveredNode) {
      this.dispatchEvent(new CustomEvent('node-select', {
        detail: { entity: this.hoveredNode },
        bubbles: true,
        composed: true
      }));
    }
  };

  private buildGraph() {
    const entries = Object.entries(this.entities);
    if (entries.length === 0) {
      this.nodes = [];
      this.edges = [];
      return;
    }

    // Build nodes
    this.nodes = entries
      .sort((a, b) => b[1] - a[1])
      .slice(0, 40)
      .map(([id, count], i) => ({
        id,
        count,
        x: this.width / 2 + (Math.random() - 0.5) * 200,
        y: this.height / 2 + (Math.random() - 0.5) * 200,
        vx: 0,
        vy: 0,
        category: this.categorizeEntity(id)
      }));

    // Build edges from co-occurrences
    const nodeIds = new Set(this.nodes.map(n => n.id));
    const edgeMap = new Map<string, number>();

    for (const { entities, count } of this.cooccurrences) {
      for (let i = 0; i < entities.length; i++) {
        for (let j = i + 1; j < entities.length; j++) {
          const a = entities[i];
          const b = entities[j];
          if (nodeIds.has(a) && nodeIds.has(b)) {
            const key = [a, b].sort().join('|||');
            edgeMap.set(key, (edgeMap.get(key) || 0) + count);
          }
        }
      }
    }

    this.edges = Array.from(edgeMap.entries())
      .map(([key, weight]) => {
        const [source, target] = key.split('|||');
        return { source, target, weight };
      })
      .filter(e => e.weight >= 2) // Only show edges with 2+ co-occurrences
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 100);
  }

  private categorizeEntity(entity: string): string {
    // Simple heuristics for categorization
    if (/^\d{4}$/.test(entity)) return 'date';
    if (/^(19|20)\d{2}s?$/.test(entity)) return 'date';
    if (/software|tech|code|programming|agile|scrum/i.test(entity)) return 'tech';
    if (/inc|corp|llc|company|software$/i.test(entity)) return 'company';
    if (/city|state|country/i.test(entity)) return 'place';
    // Could add ML-based NER here
    return 'default';
  }

  private getNodeSize(count: number): number {
    const maxCount = Math.max(...this.nodes.map(n => n.count), 1);
    const normalized = count / maxCount;
    return 8 + normalized * 25;
  }

  private startSimulation() {
    const simulate = () => {
      this.simulationStep();
      this.draw();
      this.animationFrame = requestAnimationFrame(simulate);
    };
    simulate();
  }

  private simulationStep() {
    const centerX = this.width / 2;
    const centerY = this.height / 2;

    // Forces
    for (const node of this.nodes) {
      // Center gravity
      node.vx += (centerX - node.x) * 0.001;
      node.vy += (centerY - node.y) * 0.001;

      // Node repulsion
      for (const other of this.nodes) {
        if (node.id === other.id) continue;
        const dx = node.x - other.x;
        const dy = node.y - other.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = 500 / (dist * dist);
        node.vx += (dx / dist) * force;
        node.vy += (dy / dist) * force;
      }
    }

    // Edge attraction
    for (const edge of this.edges) {
      const source = this.nodes.find(n => n.id === edge.source);
      const target = this.nodes.find(n => n.id === edge.target);
      if (!source || !target) continue;

      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = (dist - 100) * 0.01 * (edge.weight / 10);

      source.vx += (dx / dist) * force;
      source.vy += (dy / dist) * force;
      target.vx -= (dx / dist) * force;
      target.vy -= (dy / dist) * force;
    }

    // Apply velocities with damping
    for (const node of this.nodes) {
      node.vx *= 0.9;
      node.vy *= 0.9;
      node.x += node.vx;
      node.y += node.vy;

      // Boundary constraints
      const size = this.getNodeSize(node.count);
      node.x = Math.max(size, Math.min(this.width - size, node.x));
      node.y = Math.max(size, Math.min(this.height - size, node.y));
    }
  }

  private draw() {
    if (!this.ctx) return;

    this.ctx.clearRect(0, 0, this.width, this.height);

    // Draw edges
    for (const edge of this.edges) {
      const source = this.nodes.find(n => n.id === edge.source);
      const target = this.nodes.find(n => n.id === edge.target);
      if (!source || !target) continue;

      const isHighlighted = this.hoveredNode === source.id || this.hoveredNode === target.id;

      this.ctx.beginPath();
      this.ctx.moveTo(source.x, source.y);
      this.ctx.lineTo(target.x, target.y);
      this.ctx.strokeStyle = isHighlighted
        ? 'rgba(88, 166, 255, 0.8)'
        : `rgba(88, 166, 255, ${0.1 + (edge.weight / 20)})`;
      this.ctx.lineWidth = isHighlighted ? 2 : Math.min(edge.weight / 3, 4);
      this.ctx.stroke();
    }

    // Draw nodes
    for (const node of this.nodes) {
      const size = this.getNodeSize(node.count);
      const color = this.categoryColors[node.category || 'default'];
      const isHovered = this.hoveredNode === node.id;

      // Node circle
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
      this.ctx.fillStyle = isHovered ? color : this.hexToRgba(color, 0.7);
      this.ctx.fill();

      if (isHovered) {
        this.ctx.strokeStyle = '#fff';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
      }

      // Label
      this.ctx.font = `${isHovered ? 'bold ' : ''}${Math.max(10, size / 2)}px sans-serif`;
      this.ctx.fillStyle = '#f0f6fc';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';

      // Truncate long labels
      const label = node.id.length > 15 ? node.id.slice(0, 12) + '...' : node.id;
      this.ctx.fillText(label, node.x, node.y + size + 12);
    }
  }

  private hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  render() {
    const nodeCount = this.nodes.length;
    const edgeCount = this.edges.length;

    return html`
      <div class="graph-container">
        ${nodeCount === 0 ? html`
          <div class="empty">Run probes to see concept associations</div>
        ` : html`
          <canvas></canvas>
          <div class="stats">${nodeCount} concepts, ${edgeCount} associations</div>
          <div class="legend">
            <div class="legend-item">
              <span class="legend-dot" style="background: ${this.categoryColors.person}"></span>
              Person
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: ${this.categoryColors.company}"></span>
              Company
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: ${this.categoryColors.date}"></span>
              Date
            </div>
            <div class="legend-item">
              <span class="legend-dot" style="background: ${this.categoryColors.tech}"></span>
              Tech
            </div>
          </div>
        `}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'concept-graph': ConceptGraph;
  }
}
