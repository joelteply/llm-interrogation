import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface IntelNode {
  id: string;
  text: string;
  category: 'CORROBORATED' | 'UNCORROBORATED' | 'CONTRADICTED' | 'UNPROCESSED';
  confidence: number;
  sources: number;
  mentions: number;
  isEchoChamber: boolean;
  drillTotal: number | null;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface IntelEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

@customElement('intel-graph')
export class IntelGraph extends LitElement {
  static styles = css`
    :host {
      display: block;
      height: 100%;
    }

    .graph-container {
      position: relative;
      width: 100%;
      height: 100%;
      background: linear-gradient(135deg, rgba(13, 17, 23, 0.98) 0%, rgba(22, 27, 34, 0.98) 100%);
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
      gap: 16px;
      font-size: 11px;
      color: #8b949e;
      background: rgba(0,0,0,0.6);
      padding: 6px 12px;
      border-radius: 6px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }

    .legend-dot.echo {
      border: 2px solid #f85149;
      width: 6px;
      height: 6px;
    }

    .stats {
      position: absolute;
      top: 8px;
      right: 8px;
      font-size: 11px;
      color: #8b949e;
      background: rgba(0,0,0,0.6);
      padding: 6px 12px;
      border-radius: 6px;
    }

    .tooltip {
      position: absolute;
      background: rgba(22, 27, 34, 0.95);
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      color: #f0f6fc;
      pointer-events: none;
      z-index: 100;
      max-width: 300px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }

    .tooltip h4 {
      margin: 0 0 8px 0;
      font-size: 14px;
      color: #58a6ff;
    }

    .tooltip .category {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: bold;
      margin-bottom: 8px;
    }

    .tooltip .category.CORROBORATED { background: #238636; }
    .tooltip .category.UNCORROBORATED { background: #9e6a03; }
    .tooltip .category.CONTRADICTED { background: #da3633; }
    .tooltip .category.UNPROCESSED { background: #484f58; }

    .tooltip .stat {
      display: flex;
      justify-content: space-between;
      margin: 4px 0;
      color: #8b949e;
    }

    .tooltip .stat-value {
      color: #f0f6fc;
      font-weight: bold;
    }

    .tooltip .echo-warning {
      color: #f85149;
      font-weight: bold;
      margin-top: 8px;
    }

    .tooltip .drill-btn {
      margin-top: 8px;
      padding: 6px 12px;
      background: #238636;
      border: none;
      border-radius: 4px;
      color: white;
      cursor: pointer;
      font-size: 11px;
    }

    .controls {
      position: absolute;
      top: 8px;
      left: 8px;
      display: flex;
      gap: 8px;
    }

    .control-btn {
      padding: 6px 12px;
      background: rgba(48, 54, 61, 0.8);
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #8b949e;
      font-size: 11px;
      cursor: pointer;
      transition: all 0.2s;
    }

    .control-btn:hover {
      background: rgba(48, 54, 61, 1);
      color: #f0f6fc;
    }

    .control-btn.active {
      background: #238636;
      border-color: #238636;
      color: white;
    }

    .empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: #6e7681;
      gap: 16px;
    }

    .empty button {
      padding: 8px 16px;
      background: #238636;
      border: none;
      border-radius: 6px;
      color: white;
      cursor: pointer;
    }
  `;

  @property({ type: String })
  projectName = '';

  @state()
  private nodes: IntelNode[] = [];

  @state()
  private edges: IntelEdge[] = [];

  @state()
  private width = 800;

  @state()
  private height = 500;

  @state()
  private hoveredNode: IntelNode | null = null;

  @state()
  private draggedNode: IntelNode | null = null;

  @state()
  private tooltipX = 0;

  @state()
  private tooltipY = 0;

  @state()
  private filter: 'all' | 'high' | 'uncorroborated' | 'echo' = 'all';

  @state()
  private maxNodes = 75;

  // Physics parameters - adjustable (tuned defaults)
  @state()
  private minDistance = 75;

  @state()
  private centerPull = 0.0045;

  @state()
  private repulsion = 0.05;

  @state()
  private damping = 0.60;

  @state()
  private nodeBaseSize = 5;

  @state()
  private nodeSizeScale = 9;

  @state()
  private showLabels = true;

  @state()
  private autoScale = true;  // Auto-interpolate based on minDim

  @state()
  private showControls = false;  // Physics controls collapsed by default

  @state()
  private loading = false;

  @state()
  private categorizing = false;

  @state()
  private categorizeProgress = '';

  @state()
  private summary: any = null;

  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private animationFrame: number | null = null;
  private resizeObserver: ResizeObserver | null = null;

  // Colors by category (for legend/outline)
  private categoryColors: Record<string, string> = {
    CORROBORATED: '#3fb950',
    UNCORROBORATED: '#d29922',
    CONTRADICTED: '#f85149',
    UNPROCESSED: '#484f58'
  };

  // Temperature color scale (cold to hot)
  private getTemperatureColor(value: number): string {
    // value 0-1, returns color from blue (cold) -> cyan -> green -> yellow -> orange -> red (hot)
    const stops = [
      { pos: 0.0, r: 59, g: 130, b: 246 },   // blue
      { pos: 0.2, r: 34, g: 211, b: 238 },   // cyan
      { pos: 0.4, r: 34, g: 197, b: 94 },    // green
      { pos: 0.6, r: 250, g: 204, b: 21 },   // yellow
      { pos: 0.8, r: 249, g: 115, b: 22 },   // orange
      { pos: 1.0, r: 239, g: 68, b: 68 },    // red
    ];

    // Find the two stops we're between
    let lower = stops[0];
    let upper = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (value >= stops[i].pos && value <= stops[i + 1].pos) {
        lower = stops[i];
        upper = stops[i + 1];
        break;
      }
    }

    // Interpolate
    const range = upper.pos - lower.pos;
    const t = range > 0 ? (value - lower.pos) / range : 0;
    const r = Math.round(lower.r + (upper.r - lower.r) * t);
    const g = Math.round(lower.g + (upper.g - lower.g) * t);
    const b = Math.round(lower.b + (upper.b - lower.b) * t);

    return `rgb(${r}, ${g}, ${b})`;
  }

  // Compute graph distance from a source node using BFS
  // Returns map of nodeId -> distance (0 = source, 1 = direct connection, etc.)
  private computeGraphDepth(sourceId: string, nodeIds: Set<string>): Map<string, number> {
    const depths = new Map<string, number>();
    depths.set(sourceId, 0);

    // Build adjacency list from edges
    const adjacency = new Map<string, Array<{ id: string; weight: number }>>();
    for (const edge of this.edges) {
      if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue;

      if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
      if (!adjacency.has(edge.target)) adjacency.set(edge.target, []);

      adjacency.get(edge.source)!.push({ id: edge.target, weight: edge.weight });
      adjacency.get(edge.target)!.push({ id: edge.source, weight: edge.weight });
    }

    // BFS to compute depths
    const queue: string[] = [sourceId];
    const maxDepth = 5;

    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentDepth = depths.get(current)!;

      if (currentDepth >= maxDepth) continue;

      const neighbors = adjacency.get(current) || [];
      for (const { id } of neighbors) {
        if (!depths.has(id)) {
          depths.set(id, currentDepth + 1);
          queue.push(id);
        }
      }
    }

    return depths;
  }

  // Normalize edge weights for visualization
  private normalizeEdgeWeights(): { min: number; max: number; normalize: (w: number) => number } {
    if (this.edges.length === 0) return { min: 0, max: 1, normalize: () => 0.5 };

    let min = Infinity;
    let max = -Infinity;
    for (const e of this.edges) {
      if (e.weight < min) min = e.weight;
      if (e.weight > max) max = e.weight;
    }

    const range = max - min || 1;
    return {
      min,
      max,
      normalize: (w: number) => (w - min) / range
    };
  }

  // Gaussian normalization - use mean and std dev to handle outliers properly
  private getConfidenceNormalizer(nodes: IntelNode[]): (c: number) => number {
    if (nodes.length === 0) return (c) => c;
    if (nodes.length === 1) return () => 0.5;

    const values = nodes.map(n => n.confidence);

    // Compute mean
    const mean = values.reduce((a, b) => a + b, 0) / values.length;

    // Compute standard deviation
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
    const std = Math.sqrt(variance);

    if (std < 0.001) return () => 0.5; // All same confidence

    // Normalize using z-score, clamped to [-2, +2] std devs, mapped to [0, 1]
    // z = (x - mean) / std
    // normalized = (z + 2) / 4  to map [-2, 2] -> [0, 1]
    return (c: number) => {
      const z = (c - mean) / std;
      const clamped = Math.max(-2, Math.min(2, z));
      return (clamped + 2) / 4;
    };
  }

  async connectedCallback() {
    super.connectedCallback();
    if (this.projectName) {
      await this.loadData();
    }
  }

  updated(changedProperties: Map<string, unknown>) {
    if (changedProperties.has('projectName') && this.projectName) {
      this.loadData();
    }
  }

  async loadData() {
    if (!this.projectName) return;

    this.loading = true;
    try {
      // Load graph data
      const graphResp = await fetch(`/api/intel/${this.projectName}/graph`);
      const graphData = await graphResp.json();

      // Load summary
      const summaryResp = await fetch(`/api/intel/${this.projectName}/summary`);
      this.summary = await summaryResp.json();

      // Build nodes (nodes are inside graphData.graph)
      const graph = graphData.graph || graphData;
      // Use actual canvas size or reasonable defaults
      const w = this.width || 800;
      const h = this.height || 500;
      this.nodes = (graph.nodes || []).map((n: any) => {
        // Random spread across entire canvas
        return {
          id: n.id,
          text: n.text,
          category: n.category || 'UNPROCESSED',
          confidence: n.confidence || 0,
          sources: n.models?.length || 1,
          mentions: n.mentions || 1,
          isEchoChamber: n.is_echo_chamber || false,
          drillTotal: n.drill_total,
          x: 50 + Math.random() * (w - 100),
          y: 50 + Math.random() * (h - 100),
          vx: (Math.random() - 0.5) * 5,
          vy: (Math.random() - 0.5) * 5
        };
      });

      // Build edges
      this.edges = (graph.edges || []).map((e: any) => ({
        source: e.source,
        target: e.target,
        type: e.type || 'co_occurred',
        weight: e.weight || 1
      }));

    } catch (err) {
      console.error('Failed to load intel data:', err);
    }
    this.loading = false;
  }

  firstUpdated() {
    this.setupCanvas();
    this.startSimulation();
  }

  private setupCanvas() {
    this.canvas = this.shadowRoot?.querySelector('canvas') || null;
    if (!this.canvas) return;

    this.ctx = this.canvas.getContext('2d');
    this.setupCanvasResize();
  }

  private setupCanvasResize() {
    if (!this.canvas) return;

    // Set up resize observer if not already
    if (!this.resizeObserver) {
      this.resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) {
            this.handleResize(width, height);
          }
        }
      });
    }

    const parent = this.canvas.parentElement;
    if (parent) {
      this.resizeObserver.observe(parent);

      // Initial size
      const rect = parent.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        this.handleResize(rect.width, rect.height);
      }
    }

    this.canvas.addEventListener('mousemove', this.handleMouseMove);
    this.canvas.addEventListener('mousedown', this.handleMouseDown);
    this.canvas.addEventListener('mouseup', this.handleMouseUp);
    this.canvas.addEventListener('mouseleave', this.handleMouseUp);
    this.canvas.addEventListener('click', this.handleClick);
  }

  private handleResize(width: number, height: number) {
    if (!this.canvas || !this.ctx) return;

    const oldWidth = this.width || width;
    const oldHeight = this.height || height;

    // Update dimensions
    this.width = width;
    this.height = height;

    // Reset canvas buffer with new dimensions (this is the key fix)
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;

    // Reset and scale context
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(dpr, dpr);

    // Auto-adjust physics based on min dimension
    this.autoScalePhysics();

    // Scale and spread node positions to new dimensions
    if (oldWidth > 0 && oldHeight > 0 && this.nodes.length > 0) {
      const scaleX = width / oldWidth;
      const scaleY = height / oldHeight;

      for (const node of this.nodes) {
        // Scale position
        node.x *= scaleX;
        node.y *= scaleY;

        // Add some velocity to spread out
        node.vx += (node.x - width / 2) * 0.1;
        node.vy += (node.y - height / 2) * 0.1;
      }
    }
  }

  private autoScalePhysics() {
    if (!this.autoScale) return;

    const minDim = Math.min(this.width, this.height);

    // Piecewise interpolation:
    // t=0 at 200px (collapsed), t=1 at 800px (expanded), continues beyond
    const t = Math.max(0, (minDim - 200) / 600);

    // Tuned values from user testing:
    // Collapsed (200px, t=0): spacing 51, maxNodes 17, center 4.0, repel 22, nodeSize 7, sizeScale 9, damping 50%
    // Expanded (800px, t=1):  spacing 51, maxNodes 70, center 1.0, repel 1, nodeSize 8, sizeScale 12, damping 60%
    // Large (1172px, t=1.62): spacing 149, maxNodes ~100, etc - continues scaling

    // Spacing: stays at 51 until 800px, then accelerates
    if (t <= 1) {
      this.minDistance = 51;                                       // constant 51 for small/medium
    } else {
      this.minDistance = Math.round(51 + (t - 1) * 160);           // 51 -> 149 -> ...
    }

    // Max nodes: starts small, grows
    this.maxNodes = Math.round(17 + t * 53);                       // 17 -> 70 -> 123 -> ...

    // Center: high when collapsed, low when expanded
    this.centerPull = Math.max(0.001, 0.004 - t * 0.003);          // 4.0 -> 1.0 -> 0.5 (floor 1.0)

    // Repel: HIGH when collapsed (22), LOW when expanded (1)
    this.repulsion = Math.max(0.01, 0.22 - t * 0.21);              // 22 -> 1 -> 0.5 (floor 1.0)

    // Node visuals
    this.nodeBaseSize = Math.round(7 + t * 1);                     // 7 -> 8 -> 9 -> ...
    this.nodeSizeScale = Math.round(9 + t * 3);                    // 9 -> 12 -> 15 -> ...
    this.damping = Math.min(0.85, 0.50 + t * 0.10);                // 50% -> 60% -> 70% (cap 85%)
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

    // If dragging, move the node
    if (this.draggedNode) {
      this.draggedNode.x = x;
      this.draggedNode.y = y;
      this.draggedNode.vx = 0;
      this.draggedNode.vy = 0;
      this.canvas!.style.cursor = 'grabbing';
      return;
    }

    let found: IntelNode | null = null;
    for (const node of this.getFilteredNodes()) {
      const size = this.getNodeSize(node);
      const dx = node.x - x;
      const dy = node.y - y;
      if (Math.sqrt(dx * dx + dy * dy) < size + 5) {
        found = node;
        break;
      }
    }

    this.hoveredNode = found;
    this.tooltipX = e.clientX - rect.left + 15;
    this.tooltipY = e.clientY - rect.top + 15;
    this.canvas!.style.cursor = found ? 'grab' : 'default';
  };

  private handleMouseDown = (e: MouseEvent) => {
    if (this.hoveredNode) {
      this.draggedNode = this.hoveredNode;
      this.canvas!.style.cursor = 'grabbing';
      e.preventDefault();
    }
  };

  private handleMouseUp = () => {
    if (this.draggedNode) {
      this.draggedNode = null;
      this.canvas!.style.cursor = this.hoveredNode ? 'grab' : 'default';
    }
  };

  private handleClick = () => {
    if (this.hoveredNode) {
      this.dispatchEvent(new CustomEvent('entity-select', {
        detail: { entity: this.hoveredNode },
        bubbles: true,
        composed: true
      }));
    }
  };

  private getFilteredNodes(): IntelNode[] {
    let filtered: IntelNode[];
    switch (this.filter) {
      case 'high':
        filtered = this.nodes.filter(n => n.confidence >= 0.5);
        break;
      case 'uncorroborated':
        filtered = this.nodes.filter(n => n.category === 'UNCORROBORATED');
        break;
      case 'echo':
        filtered = this.nodes.filter(n => n.isEchoChamber);
        break;
      default:
        filtered = this.nodes;
    }
    // Dynamic max based on canvas area (roughly 1 node per 5000 sq pixels)
    const dynamicMax = Math.max(30, Math.floor((this.width * this.height) / 5000));
    const limit = Math.min(this.maxNodes, dynamicMax);

    // Limit nodes for readability - show highest confidence first
    if (filtered.length > limit) {
      filtered = [...filtered].sort((a, b) => b.confidence - a.confidence).slice(0, limit);
    }
    return filtered;
  }

  private getNodeSize(node: IntelNode): number {
    // Size based on confidence (bigger = more confident)
    return this.nodeBaseSize + node.confidence * this.nodeSizeScale;
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
    const nodes = this.getFilteredNodes();
    if (nodes.length === 0) return;

    const centerX = this.width / 2;
    const centerY = this.height / 2;
    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    const nodeIds = new Set(nodes.map(n => n.id));

    // Get normalized edge weights
    const { normalize: normalizeWeight } = this.normalizeEdgeWeights();

    for (const node of nodes) {
      // Skip if being dragged
      if (this.draggedNode?.id === node.id) continue;

      // Center pull
      node.vx += (centerX - node.x) * this.centerPull;
      node.vy += (centerY - node.y) * this.centerPull;

      // Repulsion from other nodes
      for (const other of nodes) {
        if (node.id === other.id) continue;

        const dx = node.x - other.x;
        const dy = node.y - other.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Repel if within range
        if (dist < this.minDistance * 3 && dist > 0) {
          const force = (this.minDistance * 3 - dist) / dist;
          node.vx += dx * force * this.repulsion;
          node.vy += dy * force * this.repulsion;
        }
      }

      // Soft boundary - just prevent going off screen
      const margin = 40;
      if (node.x < margin) node.vx += 1;
      if (node.x > this.width - margin) node.vx -= 1;
      if (node.y < margin) node.vy += 1;
      if (node.y > this.height - margin) node.vy -= 1;
    }

    // Edge-based attraction (strong edges pull nodes together)
    for (const edge of this.edges) {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) continue;
      if (this.draggedNode?.id === source.id || this.draggedNode?.id === target.id) continue;

      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > 0) {
        // Attraction force based on normalized weight
        const weightNorm = normalizeWeight(edge.weight);
        // Ideal distance: strong edges = closer (minDistance), weak = further (minDistance * 2.5)
        const idealDist = this.minDistance * (2.5 - 1.5 * weightNorm);
        const attraction = (dist - idealDist) * 0.01 * (0.5 + weightNorm * 0.5);

        const fx = (dx / dist) * attraction;
        const fy = (dy / dist) * attraction;

        source.vx += fx;
        source.vy += fy;
        target.vx -= fx;
        target.vy -= fy;
      }
    }

    // Apply velocities
    for (const node of nodes) {
      if (this.draggedNode?.id === node.id) continue;

      node.vx *= this.damping;
      node.vy *= this.damping;
      node.x += node.vx;
      node.y += node.vy;

      // Hard boundary clamp
      const padding = 40;
      node.x = Math.max(padding, Math.min(this.width - padding, node.x));
      node.y = Math.max(padding, Math.min(this.height - padding, node.y));
    }
  }

  private draw() {
    // Try to get canvas if not yet initialized
    if (!this.canvas) {
      this.canvas = this.shadowRoot?.querySelector('canvas') || null;
      if (this.canvas) {
        this.ctx = this.canvas.getContext('2d');
        this.setupCanvasResize();
      }
    }
    if (!this.ctx || !this.canvas) return;

    // Check if canvas size changed (CSS resize)
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0 &&
        (Math.abs(rect.width - this.width) > 5 || Math.abs(rect.height - this.height) > 5)) {
      this.handleResize(rect.width, rect.height);
    }

    const nodes = this.getFilteredNodes();
    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    const nodeIds = new Set(nodes.map(n => n.id));

    this.ctx.clearRect(0, 0, this.width, this.height);

    // Get normalizers
    const normalizeConf = this.getConfidenceNormalizer(nodes);
    const { normalize: normalizeWeight } = this.normalizeEdgeWeights();

    // Compute graph depth from hovered node (if any)
    const depths = this.hoveredNode
      ? this.computeGraphDepth(this.hoveredNode.id, nodeIds)
      : new Map<string, number>();

    // Draw edges - only between visible nodes
    const visibleEdges = this.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));

    // Draw all edges with strength-based opacity
    for (const edge of visibleEdges) {
      const source = nodeMap.get(edge.source)!;
      const target = nodeMap.get(edge.target)!;

      const isHighlighted = this.hoveredNode?.id === source.id || this.hoveredNode?.id === target.id;

      // Compute opacity from weight (normalized) and highlight state
      const weightNorm = normalizeWeight(edge.weight);
      const baseOpacity = 0.08 + weightNorm * 0.15;  // 0.08 - 0.23 for non-highlighted
      const opacity = isHighlighted ? 0.6 + weightNorm * 0.3 : baseOpacity;
      const lineWidth = isHighlighted ? 1.5 + weightNorm * 1.5 : 0.5 + weightNorm * 0.5;

      this.ctx.beginPath();
      this.ctx.moveTo(source.x, source.y);
      this.ctx.lineTo(target.x, target.y);
      this.ctx.strokeStyle = `rgba(88, 166, 255, ${opacity})`;
      this.ctx.lineWidth = lineWidth;
      this.ctx.stroke();
    }

    // Draw nodes
    for (const node of nodes) {
      const size = this.getNodeSize(node);
      // Temperature color based on normalized confidence (continuous heat map)
      const normConf = normalizeConf(node.confidence);
      const color = this.getTemperatureColor(normConf);
      const isHovered = this.hoveredNode?.id === node.id;
      const depth = depths.get(node.id);  // undefined if no hovered node or not connected

      // Graph depth ring (when hovering another node)
      if (depth !== undefined && depth > 0) {
        // Color by depth: depth 1 = bright, depth 5 = dim
        const depthColors = ['#58a6ff', '#388bfd', '#1f6feb', '#1158c7', '#0d419d'];
        const depthColor = depthColors[Math.min(depth - 1, depthColors.length - 1)];
        const depthOpacity = 1 - (depth - 1) * 0.15;

        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, size + 3 + depth, 0, Math.PI * 2);
        this.ctx.strokeStyle = depthColor;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = depthOpacity;
        this.ctx.stroke();
        this.ctx.globalAlpha = 1;
      }

      // Echo chamber warning ring
      if (node.isEchoChamber) {
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, size + 4, 0, Math.PI * 2);
        this.ctx.strokeStyle = '#f85149';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
      }

      // Node circle
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, size, 0, Math.PI * 2);

      // Gradient fill based on confidence
      const gradient = this.ctx.createRadialGradient(
        node.x - size/3, node.y - size/3, 0,
        node.x, node.y, size
      );
      gradient.addColorStop(0, this.lightenColor(color, 30));
      gradient.addColorStop(1, color);
      this.ctx.fillStyle = gradient;
      this.ctx.fill();

      // Category indicator (thin outline)
      if (node.category !== 'UNPROCESSED') {
        this.ctx.strokeStyle = this.categoryColors[node.category];
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
      }

      // Hover highlight
      if (isHovered) {
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 2.5;
        this.ctx.stroke();
      }

      // Confidence indicator (arc on the edge)
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, size + 1, -Math.PI/2, -Math.PI/2 + (Math.PI * 2 * node.confidence));
      this.ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      // Label
      if (this.showLabels || isHovered) {
        this.ctx.font = `${isHovered ? 'bold ' : ''}11px -apple-system, sans-serif`;
        this.ctx.fillStyle = '#f0f6fc';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';

        const label = node.text.length > 18 ? node.text.slice(0, 15) + '...' : node.text;
        this.ctx.fillText(label, node.x, node.y + size + 6);
      }
    }
  }

  private lightenColor(hex: string, percent: number): string {
    const num = parseInt(hex.slice(1), 16);
    const r = Math.min(255, (num >> 16) + percent);
    const g = Math.min(255, ((num >> 8) & 0x00FF) + percent);
    const b = Math.min(255, (num & 0x0000FF) + percent);
    return `rgb(${r}, ${g}, ${b})`;
  }

  private setFilter(f: typeof this.filter) {
    this.filter = f;
  }

  async importCorpus() {
    if (!this.projectName) {
      console.error('No project name set');
      return;
    }
    this.loading = true;
    try {
      const resp = await fetch(`/api/intel/${this.projectName}/import`, { method: 'POST' });
      await resp.json();
      await this.loadData();
    } catch (err) {
      console.error('Import failed:', err);
    }
    this.loading = false;
  }

  async categorizeEntities() {
    if (!this.projectName || this.categorizing) return;

    this.categorizing = true;
    this.categorizeProgress = 'Starting...';

    try {
      const resp = await fetch(`/api/intel/${this.projectName}/categorize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 100 })
      });

      const reader = resp.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n').filter(l => l.startsWith('data: '));

        for (const line of lines) {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'progress') {
            this.categorizeProgress = `${data.done}/${data.total}: ${data.entity} → ${data.category}`;
          } else if (data.type === 'complete') {
            this.categorizeProgress = `Done: ${data.stats.corroborated} corroborated, ${data.stats.uncorroborated} uncorroborated`;
          }
        }
      }

      await this.loadData();
    } catch (err) {
      console.error('Categorize failed:', err);
      this.categorizeProgress = 'Error';
    }

    this.categorizing = false;
  }

  render() {
    const nodes = this.getFilteredNodes();

    if (this.loading) {
      return html`
        <div class="graph-container">
          <div class="empty">
            <span>Loading intel data...</span>
          </div>
        </div>
      `;
    }

    if (this.nodes.length === 0) {
      return html`
        <div class="graph-container">
          <div class="empty">
            <span>No intel data for ${this.projectName || '(no project)'}</span>
            <button @click=${this.importCorpus} ?disabled=${!this.projectName}>
              Import from Corpus
            </button>
          </div>
        </div>
      `;
    }

    return html`
      <div class="graph-container">
        <canvas></canvas>

        <div class="controls">
          <button
            class="control-btn ${this.filter === 'all' ? 'active' : ''}"
            @click=${() => this.setFilter('all')}
          >All (${nodes.length}${this.nodes.length > this.maxNodes ? `/${this.nodes.length}` : ''})</button>
          <button
            class="control-btn ${this.filter === 'high' ? 'active' : ''}"
            @click=${() => this.setFilter('high')}
          >High Conf</button>
          <button
            class="control-btn ${this.filter === 'uncorroborated' ? 'active' : ''}"
            @click=${() => this.setFilter('uncorroborated')}
          >Uncorroborated</button>
          <button
            class="control-btn ${this.filter === 'echo' ? 'active' : ''}"
            @click=${() => this.setFilter('echo')}
          >Echo Chambers</button>
          <button
            class="control-btn"
            @click=${this.categorizeEntities}
            ?disabled=${this.categorizing}
            style="margin-left: 16px; background: ${this.categorizing ? '#238636' : 'rgba(35, 134, 54, 0.6)'}"
          >${this.categorizing ? 'Categorizing...' : 'Categorize'}</button>
        </div>

        <div class="stats">
          ${nodes.length} entities shown
          ${this.summary ? html`
            <br>
            <span style="color: #3fb950">${this.summary.confidence_distribution?.high || 0} HIGH</span> /
            <span style="color: #d29922">${this.summary.confidence_distribution?.medium || 0} MED</span> /
            <span style="color: #f85149">${this.summary.confidence_distribution?.low || 0} LOW</span>
          ` : ''}
          ${this.categorizeProgress ? html`<br><span style="color: #58a6ff; font-size: 10px">${this.categorizeProgress}</span>` : ''}
        </div>

        <div class="physics-controls" style="position: absolute; bottom: 50px; left: 8px; background: rgba(0,0,0,0.8); border-radius: 6px; font-size: 10px; color: #8b949e;">
          <button
            @click=${() => this.showControls = !this.showControls}
            style="width: 100%; padding: 6px 10px; background: transparent; border: none; color: #58a6ff; cursor: pointer; text-align: left; font-size: 10px;">
            ${this.showControls ? '▼' : '▶'} Physics ${Math.min(this.width, this.height).toFixed(0)}px
          </button>
          ${this.showControls ? html`
            <div style="padding: 8px 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px;">
              <div>
                <label>Spacing: ${this.minDistance}</label><br>
                <input type="range" min="30" max="500" .value=${this.minDistance}
                  @input=${(e: Event) => this.minDistance = +(e.target as HTMLInputElement).value}
                  style="width: 80px;">
              </div>
              <div>
                <label>Max Nodes: ${this.maxNodes}</label><br>
                <input type="range" min="10" max="150" .value=${this.maxNodes}
                  @input=${(e: Event) => this.maxNodes = +(e.target as HTMLInputElement).value}
                  style="width: 80px;">
              </div>
              <div>
                <label>Center: ${(this.centerPull * 1000).toFixed(1)}</label><br>
                <input type="range" min="0" max="20" step="0.5" .value=${this.centerPull * 1000}
                  @input=${(e: Event) => this.centerPull = +(e.target as HTMLInputElement).value / 1000}
                  style="width: 80px;">
              </div>
              <div>
                <label>Repel: ${(this.repulsion * 100).toFixed(0)}</label><br>
                <input type="range" min="1" max="30" .value=${this.repulsion * 100}
                  @input=${(e: Event) => this.repulsion = +(e.target as HTMLInputElement).value / 100}
                  style="width: 80px;">
              </div>
              <div>
                <label>Node Size: ${this.nodeBaseSize}</label><br>
                <input type="range" min="3" max="30" .value=${this.nodeBaseSize}
                  @input=${(e: Event) => this.nodeBaseSize = +(e.target as HTMLInputElement).value}
                  style="width: 80px;">
              </div>
              <div>
                <label>Size Scale: ${this.nodeSizeScale}</label><br>
                <input type="range" min="0" max="50" .value=${this.nodeSizeScale}
                  @input=${(e: Event) => this.nodeSizeScale = +(e.target as HTMLInputElement).value}
                  style="width: 80px;">
              </div>
              <div>
                <label>Damping: ${(this.damping * 100).toFixed(0)}%</label><br>
                <input type="range" min="50" max="99" .value=${this.damping * 100}
                  @input=${(e: Event) => this.damping = +(e.target as HTMLInputElement).value / 100}
                  style="width: 80px;">
              </div>
              <div style="display: flex; align-items: center; gap: 4px;">
                <input type="checkbox" .checked=${this.showLabels}
                  @change=${(e: Event) => this.showLabels = (e.target as HTMLInputElement).checked}>
                <label>Labels</label>
              </div>
              <div style="display: flex; align-items: center; gap: 4px;">
                <input type="checkbox" .checked=${this.autoScale}
                  @change=${(e: Event) => { this.autoScale = (e.target as HTMLInputElement).checked; if (this.autoScale) this.autoScalePhysics(); }}>
                <label>Auto</label>
              </div>
            </div>
          ` : ''}
        </div>

        <div class="legend">
          <div class="legend-item" style="gap: 2px;">
            <span style="font-size: 9px; color: #6e7681;">Confidence:</span>
            <span class="legend-dot" style="background: ${this.getTemperatureColor(0)}"></span>
            <span class="legend-dot" style="background: ${this.getTemperatureColor(0.25)}"></span>
            <span class="legend-dot" style="background: ${this.getTemperatureColor(0.5)}"></span>
            <span class="legend-dot" style="background: ${this.getTemperatureColor(0.75)}"></span>
            <span class="legend-dot" style="background: ${this.getTemperatureColor(1)}"></span>
          </div>
          <div style="width: 1px; background: #30363d; margin: 0 4px;"></div>
          <div class="legend-item" title="Outline color = category">
            <span class="legend-dot" style="background: transparent; border: 2px solid ${this.categoryColors.CORROBORATED}; width: 6px; height: 6px;"></span>
            Corr
          </div>
          <div class="legend-item">
            <span class="legend-dot" style="background: transparent; border: 2px solid ${this.categoryColors.UNCORROBORATED}; width: 6px; height: 6px;"></span>
            Uncorr
          </div>
          <div class="legend-item">
            <span class="legend-dot" style="background: transparent; border: 2px solid ${this.categoryColors.CONTRADICTED}; width: 6px; height: 6px;"></span>
            Contrad
          </div>
          <div class="legend-item">
            <span class="legend-dot echo"></span>
            Echo
          </div>
        </div>

        ${this.hoveredNode ? html`
          <div class="tooltip" style="left: ${this.tooltipX}px; top: ${this.tooltipY}px">
            <h4>${this.hoveredNode.text}</h4>
            <span class="category ${this.hoveredNode.category}">${this.hoveredNode.category}</span>

            <div class="stat">
              <span>Confidence</span>
              <span class="stat-value">${(this.hoveredNode.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="stat">
              <span>Sources</span>
              <span class="stat-value">${this.hoveredNode.sources} model(s)</span>
            </div>
            <div class="stat">
              <span>Mentions</span>
              <span class="stat-value">${this.hoveredNode.mentions}x</span>
            </div>
            ${this.hoveredNode.drillTotal !== null ? html`
              <div class="stat">
                <span>Drill Score</span>
                <span class="stat-value">${this.hoveredNode.drillTotal.toFixed(0)}/60</span>
              </div>
            ` : ''}

            ${this.hoveredNode.isEchoChamber ? html`
              <div class="echo-warning">Same model echo chamber</div>
            ` : ''}
          </div>
        ` : ''}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'intel-graph': IntelGraph;
  }
}
