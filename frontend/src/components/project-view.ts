import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { navigateTo, probeState, groundTruthState, resetProbeState, type ProbeState } from '../state';
import { getProject, getFindings, startProbe, generateQuestions, updateProject, curateProject } from '../api';
import type { Project, Findings, SSEEvent, GeneratedQuestion, ProbeResponse, EntityMatch } from '../types';

@customElement('project-view')
export class ProjectView extends LitElement {
  static styles = css`
    :host {
      display: block;
      background: #0d1117;
      min-height: 100vh;
    }

    /* Top bar with controls */
    .top-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 24px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .top-left {
      display: flex;
      align-items: center;
      gap: 16px;
      flex: 1;
      min-width: 0;
    }

    .back-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      background: transparent;
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #8b949e;
      cursor: pointer;
      font-size: 16px;
    }

    .back-btn:hover {
      border-color: #58a6ff;
      color: #f0f6fc;
    }

    h1 {
      font-size: 18px;
      font-weight: 600;
      color: #f0f6fc;
      margin: 0;
    }

    .question-field {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 1;
      min-width: 0;
    }

    .question-label {
      font-size: 14px;
      font-weight: 500;
      color: #8b949e;
      white-space: nowrap;
    }

    .topic-input {
      font-size: 18px;
      font-weight: 600;
      color: #f0f6fc;
      background: transparent;
      border: none;
      border-bottom: 2px solid #30363d;
      padding: 4px 0;
      flex: 1;
      min-width: 0;
      outline: none;
      transition: border-color 150ms;
      resize: none;
      overflow-y: auto;
      min-height: 42px;
      max-height: 200px;
      line-height: 1.4;
      font-family: inherit;
    }

    .topic-input:hover {
      border-bottom-color: #484f58;
    }

    .topic-input:focus {
      border-bottom-color: #58a6ff;
    }

    .topic-input::placeholder {
      color: #6e7681;
    }

    .topic-input.saved {
      border-bottom-color: #3fb950;
    }

    .topic-display {
      font-size: 18px;
      font-weight: 600;
      color: #f0f6fc;
      cursor: pointer;
      padding: 4px 0;
      border-bottom: 2px solid transparent;
      transition: border-color 150ms, color 150ms;
    }

    .topic-display:hover {
      border-bottom-color: #484f58;
      color: #58a6ff;
    }

    .topic-saved-indicator {
      font-size: 10px;
      color: #3fb950;
      position: absolute;
      right: 4px;
      bottom: 4px;
      opacity: 0;
      transition: opacity 200ms;
    }

    .topic-saved-indicator.show {
      opacity: 1;
    }

    .question-field {
      position: relative;
    }

    .regen-btn {
      background: transparent;
      border: 1px solid #484f58;
      color: #8b949e;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 10px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 200ms;
      margin-left: 8px;
    }

    .regen-btn.show {
      opacity: 1;
    }

    .regen-btn:hover {
      border-color: #58a6ff;
      color: #58a6ff;
    }

    .top-controls {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* Main layout: Question queue LEFT, everything else RIGHT */
    .main-layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      height: calc(100vh - 65px);
      overflow: hidden;
    }

    @media (max-width: 900px) {
      .main-layout {
        grid-template-columns: 1fr;
      }
      .questions-column {
        display: none;
      }
    }

    /* LEFT: Question queue - full height */
    .questions-column {
      height: 100%;
      overflow-y: auto;
      border-right: 1px solid #30363d;
      background: #0d1117;
      padding: 16px;
    }

    /* RIGHT: Resizable panels for findings + responses */
    .right-panel {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }

    resizable-panel {
      flex: 1;
      overflow: hidden;
    }

    /* Top: Interactive findings/word cloud */
    .findings-top {
      padding: 8px;
      background: #161b22;
      height: 100%;
      box-sizing: border-box;
      overflow: hidden;
    }

    .findings-top findings-panel {
      height: 100%;
      display: block;
    }

    /* Bottom: Response stream */
    .responses-area {
      padding: 24px;
      height: 100%;
      overflow: auto;
    }

    /* Config drawer */
    .config-drawer {
      position: fixed;
      top: 65px;
      left: 0;
      width: 350px;
      height: calc(100vh - 65px);
      background: #161b22;
      border-right: 1px solid #30363d;
      transform: translateX(-100%);
      transition: transform 200ms ease;
      z-index: 40;
      overflow-y: auto;
      padding: 16px;
    }

    .config-drawer.open {
      transform: translateX(0);
    }

    .config-overlay {
      position: fixed;
      top: 65px;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.5);
      z-index: 39;
      opacity: 0;
      pointer-events: none;
      transition: opacity 200ms ease;
    }

    .config-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }

    .drawer-close {
      position: absolute;
      top: 8px;
      right: 8px;
      background: none;
      border: none;
      color: #8b949e;
      font-size: 20px;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 4px;
    }

    .drawer-close:hover {
      background: #21262d;
      color: #c9d1d9;
    }

    .config-btn {
      padding: 8px 16px;
      background: #238636;
      border: none;
      border-radius: 6px;
      color: white;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }

    .config-btn:hover {
      background: #2ea043;
    }

    .config-btn.secondary {
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
    }

    /* Android Studio style run controls */
    .run-controls {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      background: #21262d;
      border-radius: 6px;
    }

    .run-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      background: transparent;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background 150ms;
    }

    .run-btn:hover {
      background: #30363d;
    }

    .run-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .run-btn.play {
      color: #3fb950;
    }

    .run-btn.play:hover:not(:disabled) {
      background: rgba(63, 185, 80, 0.2);
    }

    .run-btn.stop {
      color: #f85149;
    }

    .run-btn.stop:hover:not(:disabled) {
      background: rgba(248, 81, 73, 0.2);
    }

    .run-btn.preview {
      color: #58a6ff;
    }

    .run-btn.preview:hover:not(:disabled) {
      background: rgba(88, 166, 255, 0.2);
    }

    .run-btn svg {
      width: 16px;
      height: 16px;
      fill: currentColor;
    }

    .divider {
      width: 1px;
      height: 20px;
      background: #30363d;
      margin: 0 4px;
    }

    .run-counter {
      font-size: 13px;
      font-weight: 600;
      color: #6e7681;
      font-family: monospace;
      min-width: 50px;
      text-align: center;
      padding: 4px 8px;
      border-radius: 4px;
      transition: all 150ms;
    }

    .run-counter:hover {
      background: #30363d;
    }

    .run-counter.active {
      color: #3fb950;
      animation: pulse-counter 1s ease-in-out infinite;
    }

    .run-counter.infinite {
      color: #58a6ff;
    }

    @keyframes pulse-counter {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }

    .toggle-badge {
      font-size: 10px;
      font-weight: 600;
      padding: 3px 6px;
      border-radius: 3px;
      cursor: pointer;
      user-select: none;
      background: #21262d;
      color: #6e7681;
      border: 1px solid #30363d;
      transition: all 150ms;
    }

    .toggle-badge:hover {
      background: #30363d;
      color: #8b949e;
    }

    .toggle-badge.active {
      background: rgba(63, 185, 80, 0.2);
      color: #3fb950;
      border-color: #3fb950;
    }

    .toggle-badge.curating {
      background: rgba(88, 166, 255, 0.2);
      color: #58a6ff;
      border-color: #58a6ff;
      animation: pulse 1s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }

    /* Recycle bin */
    .recycle-bin {
      position: relative;
    }

    .bin-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      background: transparent;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      color: #6e7681;
      font-size: 16px;
      transition: all 150ms;
    }

    .bin-btn:hover {
      background: #30363d;
      color: #f85149;
    }

    .bin-btn.has-items {
      color: #f0883e;
    }

    .bin-count {
      position: absolute;
      top: -4px;
      right: -4px;
      min-width: 16px;
      height: 16px;
      padding: 0 4px;
      background: #f85149;
      border-radius: 8px;
      font-size: 10px;
      font-weight: 600;
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .bin-dropdown {
      position: absolute;
      top: 100%;
      right: 0;
      margin-top: 8px;
      width: 280px;
      max-height: 300px;
      overflow-y: auto;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      z-index: 100;
    }

    .bin-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border-bottom: 1px solid #21262d;
      font-size: 12px;
      color: #8b949e;
    }

    .bin-empty-btn {
      background: none;
      border: none;
      color: #f85149;
      font-size: 11px;
      cursor: pointer;
    }

    .bin-empty-btn:hover {
      text-decoration: underline;
    }

    .bin-items {
      padding: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .bin-item {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      font-size: 12px;
      color: #8b949e;
      cursor: pointer;
      user-select: none;
      transition: all 150ms;
    }

    .bin-item:hover {
      background: #30363d;
      color: #c9d1d9;
      border-color: #58a6ff;
    }

    .bin-item:active {
      background: rgba(248, 81, 73, 0.2);
      border-color: #f85149;
    }

    .bin-hint {
      padding: 8px 12px;
      font-size: 10px;
      color: #484f58;
      text-align: center;
      border-top: 1px solid #21262d;
    }

    /* Error toast */
    .error-toast {
      position: fixed;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(248, 81, 73, 0.95);
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      z-index: 1000;
      animation: toast-in 0.3s ease-out;
      box-shadow: 0 4px 16px rgba(248, 81, 73, 0.4);
    }

    @keyframes toast-in {
      from { opacity: 0; transform: translateX(-50%) translateY(20px); }
      to { opacity: 1; transform: translateX(-50%) translateY(0); }
    }

    .loading {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
      color: #8b949e;
    }
  `;

  @property({ type: String })
  projectName: string | null = null;

  @state()
  private project: Project | null = null;

  @state()
  private findings: Findings | null = null;

  @state()
  private isLoading = true;

  @state()
  private _probeState: ProbeState = probeState.get();

  private _unsubscribe?: () => void;

  // Track what each model has seen for first_mention detection during streaming
  private _modelContext: Record<string, Set<string>> = {};

  async connectedCallback() {
    super.connectedCallback();
    this._unsubscribe = probeState.subscribe((s) => {
      this._probeState = s;
      this.requestUpdate();  // Force re-render on state change
    });
    document.addEventListener('click', this.handleClickOutside);
    await this.loadProject();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribe?.();
    document.removeEventListener('click', this.handleClickOutside);
  }

  async updated(changed: Map<string, unknown>) {
    if (changed.has('projectName') && this.projectName) {
      await this.loadProject();
    }
    // Auto-resize topic textarea to fit content
    requestAnimationFrame(() => {
      const textarea = this.shadowRoot?.querySelector('.topic-input') as HTMLTextAreaElement;
      if (textarea && textarea.value) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
      }
    });
  }

  async loadProject() {
    if (!this.projectName) return;

    this.isLoading = true;

    // CRITICAL: Reset state before loading new project to prevent cross-project leakage
    resetProbeState();
    this._modelContext = {};  // Reset model context for first_mention tracking

    try {
      this.project = await getProject(this.projectName);

      // Load saved state into probe state
      // Default topic from project name if not set
      const defaultTopic = this.projectName
        .replace(/-/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase());

      probeState.update((s) => ({
        ...s,
        topic: this.project!.topic || defaultTopic
      }));

      if (this.project.angles) {
        probeState.update((s) => ({ ...s, angles: this.project!.angles || [] }));
      }
      if (this.project.ground_truth) {
        groundTruthState.update((s) => ({
          ...s,
          facts: this.project!.ground_truth || [],
        }));
      }

      // Load hidden and promoted entities (negative/positive like Stable Diffusion)
      // Also load saved models and questions
      probeState.update((s) => ({
        ...s,
        hiddenEntities: this.project!.hidden_entities || [],
        promotedEntities: this.project!.promoted_entities || [],
        narrative: this.project!.narrative || '',
        narrativeUpdated: this.project!.narrative_updated ? new Date(this.project!.narrative_updated).getTime() : null,
        userNotes: this.project!.user_notes || '',
        selectedModels: this.project!.selected_models?.length ? this.project!.selected_models : s.selectedModels,
        questions: this.project!.questions || [],
      }));

      // Load findings
      try {
        this.findings = await getFindings(this.projectName);
        probeState.update((s) => ({ ...s, findings: this.findings }));

        // Auto-curate on load if enabled and has data
        const curState = probeState.get();
        if (curState.autoCurate && this.findings?.entities && Object.keys(this.findings.entities).length > 10) {
          console.log('Auto-curating on load...');
          this.handleCurate();
        }
      } catch {
        // No findings yet, that's fine
      }

      // Check for autostart flag
      const state = probeState.get();
      if (state.shouldAutostart && state.topic) {
        // Clear the flag and kick off probe
        probeState.update(s => ({ ...s, shouldAutostart: false }));
        // Small delay to let UI render first
        setTimeout(() => this.handleRun(), 100);
      }
    } catch (err) {
      console.error('Failed to load project:', err);
    } finally {
      this.isLoading = false;
    }
  }

  @state()
  private showConfig = false;

  @state()
  private showBin = false;

  @state()
  private binPressStart = 0;

  @state()
  private isEditingTopic = false;

  @state()
  private errorToast: string | null = null;

  private showErrorToast(message: string) {
    // Simplify error messages for common cases
    let displayMsg = message;
    if (message.includes('401') || message.includes('Unauthorized')) {
      displayMsg = 'API key invalid or expired - check your keys';
    } else if (message.includes('402') || message.includes('Payment')) {
      displayMsg = 'API payment required - check billing';
    } else if (message.includes('429') || message.includes('rate limit')) {
      displayMsg = 'Rate limited - will retry with other models';
    } else if (message.includes('500') || message.includes('Internal')) {
      displayMsg = 'Model server error - continuing with others';
    }

    this.errorToast = displayMsg;
    setTimeout(() => {
      this.errorToast = null;
    }, 5000);
  }

  private _abortController: AbortController | null = null;

  private async handlePreview() {
    const state = probeState.get();
    if (!state.topic) return;

    probeState.update(s => ({ ...s, isGenerating: true }));
    try {
      const result = await generateQuestions(
        state.topic,
        state.angles,
        state.questionCount,
        state.techniquePreset,
        Object.keys(state.findings?.entities || {}).slice(0, 20),
        this.projectName || undefined  // Pass project for banned/promoted entity context
      );
      const questions = result.questions as GeneratedQuestion[];
      probeState.update(s => ({ ...s, questions, isGenerating: false }));
      // Persist questions to project
      if (this.projectName) {
        updateProject(this.projectName, { questions });
      }
    } catch (err) {
      console.error('Failed to generate questions:', err);
      probeState.update(s => ({ ...s, isGenerating: false }));
    }
  }

  private async handleRun() {
    const state = probeState.get();
    if (!this.projectName || !state.topic) return;

    // Save config
    await updateProject(this.projectName, {
      topic: state.topic,
      angles: state.angles,
      ground_truth: groundTruthState.get().facts,
    });

    probeState.update(s => ({ ...s, isRunning: true, responses: [] }));

    this._abortController = startProbe(
      {
        project: this.projectName,
        topic: state.topic,
        angles: state.angles,
        models: state.selectedModels,
        runs_per_question: state.runsPerQuestion,
        questions_count: state.questionCount,
        technique_preset: state.techniquePreset,
        questions: state.questions.length > 0 ? state.questions : Array(state.questionCount).fill(null),  // Full objects
        negative_entities: state.hiddenEntities,    // Entities to AVOID
        positive_entities: state.promotedEntities,  // Entities to FOCUS ON
        auto_curate: state.autoCurate,              // Let AI clean up while running
        infinite_mode: state.infiniteMode,          // Keep running until stopped
      },
      (event: SSEEvent) => {
        // Debug: trace ALL events
        console.log('[SSE event]', event.type, event.type === 'response' ? '(response data)' : event);
        if (event.type === 'questions') {
          probeState.update(s => ({ ...s, questions: event.data as GeneratedQuestion[] }));
        } else if (event.type === 'response') {
          const resp = event.data as ProbeResponse;

          // Track model context OUTSIDE state update (side effect)
          const modelKey = resp.model;
          if (!this._modelContext[modelKey]) this._modelContext[modelKey] = new Set();

          // Pre-compute first mentions before state update
          const entityFirstMentions: Record<string, boolean> = {};
          for (const entity of (resp.entities || [])) {
            entityFirstMentions[entity] = !this._modelContext[modelKey].has(entity.toLowerCase());
          }

          // Update model context with new entities
          for (const entity of (resp.entities || [])) {
            this._modelContext[modelKey].add(entity.toLowerCase());
          }

          // Now do pure state update
          probeState.update(s => {
            const newResponses = [...s.responses, resp];
            const currentMatches = s.findings?.entity_matches || {};
            const updatedMatches = { ...currentMatches };

            for (const entity of (resp.entities || [])) {
              if (!updatedMatches[entity]) updatedMatches[entity] = [];

              let context = '';
              if (resp.response) {
                for (const sentence of resp.response.replace(/\n/g, '. ').split('. ')) {
                  if (sentence.toLowerCase().includes(entity.toLowerCase())) {
                    context = sentence.trim().substring(0, 150);
                    break;
                  }
                }
                if (!context) context = resp.response.substring(0, 100);
              }

              updatedMatches[entity].push({
                model: resp.model,
                question: resp.question?.substring(0, 100) || '',
                context,
                is_refusal: resp.is_refusal || false,
                is_first_mention: entityFirstMentions[entity]
              });
            }

            // Create or update findings with entity_matches
            const updatedFindings: Findings = s.findings
              ? { ...s.findings, entity_matches: updatedMatches as Record<string, EntityMatch[]> }
              : {
                  entities: {},
                  by_model: {},
                  by_question: {},
                  corpus_size: 0,
                  refusal_rate: 0,
                  entity_matches: updatedMatches as Record<string, EntityMatch[]>
                };

            // Also update entities count for word cloud
            for (const entity of (resp.entities || [])) {
              updatedFindings.entities[entity] = (updatedFindings.entities[entity] || 0) + 1;
            }

            return { ...s, responses: newResponses, findings: updatedFindings };
          });
        } else if (event.type === 'findings_update') {
          // Merge SSE findings with our accumulated entity_matches
          const newFindings = event.data as Findings;
          const existingMatches = this.findings?.entity_matches || probeState.get().findings?.entity_matches;
          if (existingMatches) {
            newFindings.entity_matches = existingMatches;
          }
          probeState.update(s => ({ ...s, findings: newFindings }));
          this.findings = newFindings;
        } else if (event.type === 'curate_ban') {
          // AI auto-banned some noise entities
          const banned = (event as any).entities as string[];
          probeState.update(s => ({
            ...s,
            hiddenEntities: [...new Set([...s.hiddenEntities, ...banned])]
          }));
          console.log('Auto-banned:', banned);
        } else if (event.type === 'curate_promote') {
          // AI auto-promoted some promising entities
          const promoted = (event as any).entities as string[];
          probeState.update(s => ({
            ...s,
            promotedEntities: [...new Set([...s.promotedEntities, ...promoted])]
          }));
          console.log('Auto-promoted:', promoted);
        } else if (event.type === 'models_selected') {
          // Auto-survey picked these models - update UI
          const models = (event as any).models as string[];
          probeState.update(s => ({ ...s, selectedModels: models }));
          // Save to project
          if (this.projectName) {
            updateProject(this.projectName, { selected_models: models });
          }
        } else if (event.type === 'narrative') {
          // Interrogator's updated working theory
          const text = (event.data as any)?.text || (event as any).text || (event.data as any)?.narrative || (event as any).narrative || '';
          console.log('[SSE] narrative event received:', text.substring(0, 100) + '...');
          probeState.update(s => ({ ...s, narrative: text, narrativeUpdated: Date.now() }));
          console.log('[SSE] probeState.narrative updated to:', probeState.get().narrative?.substring(0, 50));
        } else if (event.type === 'error') {
          // Model or API error - log but keep running
          const msg = (event as any).message || (event.data as any)?.message || 'Unknown error';
          console.warn('Probe error (continuing):', msg);
          // Show brief toast-like notification, don't stop the probe
          this.showErrorToast(msg);
        } else if (event.type === 'run_start') {
          // New question starting - update the current question index
          const qIdx = (event as any).question_index as number;
          console.log('[SSE] run_start: updating currentQuestionIndex to', qIdx);
          probeState.update(s => ({ ...s, currentQuestionIndex: qIdx }));
        } else if (event.type === 'model_active') {
          // Highlight which model is currently being queried
          const model = (event as any).model as string;
          probeState.update(s => ({ ...s, activeModel: model }));
        } else if (event.type === 'entity_verification') {
          // PUBLIC vs PRIVATE entity verification from web search
          const verification = event.data as any;
          probeState.update(s => ({ ...s, entityVerification: verification }));
          console.log('Entity verification:', verification?.summary);
        } else if (event.type === 'complete') {
          probeState.update(s => ({ ...s, isRunning: false, activeModel: null }));
          this._abortController = null;
          // Reload project to get any narratives generated during probe
          if (this.projectName) {
            getProject(this.projectName).then(proj => {
              if (proj.narrative) {
                probeState.update(s => ({ ...s, narrative: proj.narrative || '' }));
              }
            }).catch(() => {});
          }
        }
      },
      (error) => {
        console.error('Probe error:', error);
        probeState.update(s => ({ ...s, isRunning: false }));
        this._abortController = null;
      }
    );
  }

  private handleStop() {
    if (this._abortController) {
      this._abortController.abort();
      this._abortController = null;
      probeState.update(s => ({ ...s, isRunning: false }));
    }
  }

  @state()
  private isCurating = false;

  private async handleCurate() {
    if (!this.projectName || this.isCurating) return;

    this.isCurating = true;
    try {
      // Curation now cleans data at the source (probe_corpus)
      const result = await curateProject(this.projectName);
      console.log('Curate result:', result);

      // Refresh findings - data is already cleaned by backend
      this.findings = await getFindings(this.projectName);

      // Reload project to get updated working_theory
      this.project = await getProject(this.projectName);

      // Update state with fresh data
      probeState.update(s => ({
        ...s,
        findings: this.findings,
        promotedEntities: this.project?.promoted_entities || [],
      }));

      const changes = result.changes || {};
      console.log(`Curate: banned ${changes.banned || 0} mentions, merged ${changes.merged || 0}, relations ${changes.relations_added || 0}`);
    } catch (err) {
      console.error('Curate failed:', err);
    } finally {
      this.isCurating = false;
    }
  }

  private handleBinItemMouseDown(entity: string) {
    this.binPressStart = Date.now();
  }

  private async handleBinItemMouseUp(entity: string) {
    const duration = Date.now() - this.binPressStart;
    this.binPressStart = 0;

    if (duration < 300) {
      // Tap - restore to visibility (remove from hidden list)
      probeState.update(s => ({
        ...s,
        hiddenEntities: s.hiddenEntities.filter(e => e !== entity)
      }));

      // Persist removal from hidden list and reload findings
      if (this.projectName) {
        await updateProject(this.projectName, {
          hidden_entities: probeState.get().hiddenEntities,
        });
        // Reload findings to include restored entity
        const findings = await getFindings(this.projectName);
        probeState.update(s => ({ ...s, findings }));
      }
    }
    // Long press - keep it hidden (no action needed, it stays banned)
    // The entity remains in hiddenEntities, so it stays filtered out
  }

  private async emptyBin() {
    probeState.update(s => ({ ...s, hiddenEntities: [] }));
    this.showBin = false;

    // Persist to backend and reload findings
    if (this.projectName) {
      await updateProject(this.projectName, {
        hidden_entities: [],
      });
      // Reload findings to include all restored entities
      const findings = await getFindings(this.projectName);
      probeState.update(s => ({ ...s, findings }));
    }
  }

  private handleClickOutside = (e: MouseEvent) => {
    if (this.showBin) {
      const bin = this.shadowRoot?.querySelector('.recycle-bin');
      if (bin && !bin.contains(e.target as Node)) {
        this.showBin = false;
      }
    }
  };

  render() {
    if (this.isLoading) {
      return html`<div class="loading">Loading...</div>`;
    }

    if (!this.project) {
      return html`<div class="loading">Project not found</div>`;
    }

    const isRunning = this._probeState.isRunning;

    return html`
      <!-- Top bar -->
      <div class="top-bar">
        <div class="top-left">
          <button class="back-btn" @click=${() => navigateTo('projects')}>←</button>
          <div class="question-field">
            <span class="question-label">Q:</span>
            ${this.isEditingTopic ? html`
              <textarea
                class="topic-input"
                rows="1"
                .value=${this._probeState.topic || this.project.name}
                placeholder="What do you want to extract?"
                @input=${(e: Event) => {
                  const textarea = e.target as HTMLTextAreaElement;
                  const value = textarea.value;
                  probeState.update(s => ({ ...s, topic: value }));
                  // Auto-resize
                  textarea.style.height = 'auto';
                  textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
                }}
                @blur=${async () => {
                  // Auto-save topic on blur and exit edit mode
                  this.isEditingTopic = false;
                  if (this.projectName) {
                    await updateProject(this.projectName, { topic: this._probeState.topic });
                  }
                }}
                @keydown=${(e: KeyboardEvent) => {
                  if (e.key === 'Escape') {
                    this.isEditingTopic = false;
                  } else if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    (e.target as HTMLTextAreaElement).blur();
                  }
                }}
              ></textarea>
            ` : html`
              <span
                class="topic-display"
                @click=${() => {
                  this.isEditingTopic = true;
                  // Focus textarea after render
                  requestAnimationFrame(() => {
                    const textarea = this.shadowRoot?.querySelector('.topic-input') as HTMLTextAreaElement;
                    if (textarea) {
                      textarea.focus();
                      textarea.select();
                    }
                  });
                }}
              >${this._probeState.topic || this.project.name || 'Click to set topic...'}</span>
            `}
          </div>
        </div>
        <div class="top-controls">
          <!-- Android Studio style run controls -->
          <div class="run-controls">
            <span
              class="toggle-badge"
              title="Models: ${this._probeState.selectedModels.length === 0 ? 'AUTO-SURVEY (all models)' : this._probeState.selectedModels.join(', ')}"
              style="font-size: 9px; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
            >
              ${this._probeState.selectedModels.length === 0 ? 'AUTO' : this._probeState.selectedModels.length + ' models'}
            </span>
            <button
              class="run-btn preview"
              title="Preview Questions"
              ?disabled=${isRunning || this._probeState.isGenerating}
              @click=${this.handlePreview}
            >
              <svg viewBox="0 0 16 16"><path d="M8 3a5 5 0 100 10A5 5 0 008 3zM1 8a7 7 0 1114 0A7 7 0 011 8zm8-1V5H7v2H5v2h2v2h2V9h2V7H9z"/></svg>
            </button>
            <div class="divider"></div>
            <button
              class="run-btn play"
              title="Run Probe"
              ?disabled=${isRunning}
              @click=${this.handleRun}
            >
              <svg viewBox="0 0 16 16"><path d="M4 2l10 6-10 6V2z"/></svg>
            </button>
            <span
              class="run-counter ${isRunning ? 'active' : ''} ${this._probeState.infiniteMode ? 'infinite' : ''}"
              title="Click to toggle infinite mode. ${this._probeState.selectedModels.length === 0 ? 'Auto-survey mode: will sample all models' : `Models: ${this._probeState.selectedModels.join(', ')}`}"
              @click=${() => probeState.update(s => ({ ...s, infiniteMode: !s.infiniteMode }))}
              style="cursor: pointer;"
            >
              ${this._probeState.responses.length}:${this._probeState.infiniteMode ? '∞' : this._probeState.runsPerQuestion * this._probeState.questionCount * Math.max(this._probeState.selectedModels.length, 1)}
            </span>
            <span
              class="toggle-badge ${this._probeState.autoCurate ? 'active' : ''} ${this.isCurating ? 'curating' : ''}"
              title="Click to curate now. When active, also auto-curates during probes."
              @click=${this.handleCurate}
              @contextmenu=${(e: Event) => { e.preventDefault(); probeState.update(s => ({ ...s, autoCurate: !s.autoCurate })); }}
            >
              ${this.isCurating ? 'CURATING...' : 'CURATE'}
            </span>
            <button
              class="run-btn stop"
              title="Stop"
              ?disabled=${!isRunning}
              @click=${this.handleStop}
            >
              <svg viewBox="0 0 16 16"><rect x="3" y="3" width="10" height="10" rx="1"/></svg>
            </button>
            <div class="divider"></div>
            <!-- Recycle bin -->
            <div class="recycle-bin">
              <button
                class="bin-btn ${this._probeState.hiddenEntities.length > 0 ? 'has-items' : ''}"
                title="Removed entities"
                @click=${(e: Event) => { e.stopPropagation(); this.showBin = !this.showBin; }}
              >
                🗑️
                ${this._probeState.hiddenEntities.length > 0 ? html`
                  <span class="bin-count">${this._probeState.hiddenEntities.length}</span>
                ` : null}
              </button>
              ${this.showBin ? html`
                <div class="bin-dropdown">
                  <div class="bin-header">
                    <span>Removed (${this._probeState.hiddenEntities.length})</span>
                    ${this._probeState.hiddenEntities.length > 0 ? html`
                      <button class="bin-empty-btn" @click=${this.emptyBin}>Empty all</button>
                    ` : null}
                  </div>
                  <div class="bin-items">
                    ${this._probeState.hiddenEntities.length === 0 ? html`
                      <span style="color: #484f58; font-size: 12px; padding: 12px;">No removed entities</span>
                    ` : this._probeState.hiddenEntities.map(entity => html`
                      <span
                        class="bin-item"
                        @mousedown=${() => this.handleBinItemMouseDown(entity)}
                        @mouseup=${() => this.handleBinItemMouseUp(entity)}
                        @mouseleave=${() => this.binPressStart = 0}
                      >
                        ${entity}
                      </span>
                    `)}
                  </div>
                  <div class="bin-hint">tap = restore · hold = delete permanently</div>
                </div>
              ` : null}
            </div>
          </div>
          <button
            class="config-btn secondary"
            @click=${() => this.showConfig = !this.showConfig}
          >
            ⚙️ Config
          </button>
        </div>
      </div>

      <!-- Main layout: Questions LEFT, everything else RIGHT -->
      <div class="main-layout">
        <!-- LEFT: Question queue (full height) -->
        <div class="questions-column">
          <question-queue></question-queue>
        </div>

        <!-- RIGHT: Findings + Responses -->
        <div class="right-panel">
          <resizable-panel .initialHeight=${280} .minHeight=${150} .maxHeight=${600}>
            <!-- TOP: Interactive word cloud -->
            <div slot="top" class="findings-top">
              <findings-panel .findings=${this.findings}></findings-panel>
            </div>

            <!-- BOTTOM: Response stream -->
            <div slot="bottom" class="responses-area">
              <response-stream .projectName=${this.projectName} @stop-probe=${this.handleStop}></response-stream>
            </div>
          </resizable-panel>
        </div>
      </div>

      <!-- Config overlay (click to close) -->
      <div
        class="config-overlay ${this.showConfig ? 'open' : ''}"
        @click=${() => this.showConfig = false}
      ></div>

      <!-- Config drawer (slides from left) -->
      <div class="config-drawer ${this.showConfig ? 'open' : ''}">
        <button class="drawer-close" @click=${() => this.showConfig = false}>&times;</button>
        <probe-controls .projectName=${this.projectName}></probe-controls>
      </div>

      <!-- Error toast -->
      ${this.errorToast ? html`
        <div class="error-toast">${this.errorToast}</div>
      ` : null}
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'project-view': ProjectView;
  }
}
