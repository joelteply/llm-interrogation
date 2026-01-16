import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { probeState, type ProbeState } from '../state';
import { getTranscript, updateProject, getAvailableModels, type TranscriptQuestion } from '../api';
import type { ProbeResponse, ModelInfo, GeneratedQuestion, EntityVerification } from '../types';

@customElement('response-stream')
export class ResponseStream extends LitElement {
  @property({ type: String })
  projectName: string | null = null;
  static styles = css`
    :host {
      display: block;
    }

    .card {
      background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
      border: 2px solid #30363d;
      border-radius: 12px;
      padding: 0;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px;
      position: sticky;
      top: 0;
      background: linear-gradient(180deg, rgba(13, 17, 23, 0.98) 0%, rgba(13, 17, 23, 0.95) 100%);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid #21262d;
      z-index: 10;
    }

    .header-title {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    h3 {
      font-size: 16px;
      font-weight: 700;
      color: #f0f6fc;
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .live-badge {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: rgba(248, 81, 73, 0.2);
      border: 1px solid rgba(248, 81, 73, 0.4);
      border-radius: 20px;
      font-size: 10px;
      font-weight: 600;
      color: #f85149;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .live-dot {
      width: 6px;
      height: 6px;
      background: #f85149;
      border-radius: 50%;
      animation: pulse-live 1.5s ease-in-out infinite;
    }

    @keyframes pulse-live {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(0.8); }
    }

    .model-tags {
      display: flex;
      align-items: flex-start;
      gap: 4px;
      flex-wrap: wrap;
      max-height: 32px;
      overflow: hidden;
      transition: max-height 0.2s ease;
    }

    .model-tags.expanded {
      max-height: 200px;
      overflow-y: auto;
    }

    .model-tags-toggle {
      padding: 2px 6px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      color: #8b949e;
      cursor: pointer;
      font-size: 10px;
      flex-shrink: 0;
    }

    .model-tags-toggle:hover {
      background: #30363d;
      color: #c9d1d9;
    }

    .model-tag {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      padding: 2px 6px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 3px;
      font-size: 10px;
      color: #8b949e;
    }

    .model-tag .name {
      color: #c9d1d9;
    }

    .model-tag.active {
      background: #238636;
      border-color: #2ea043;
      animation: pulse-active 0.8s ease-in-out infinite;
    }

    .model-tag.active .name {
      color: #fff;
      font-weight: 600;
    }

    @keyframes pulse-active {
      0%, 100% { box-shadow: 0 0 0 0 rgba(35, 134, 54, 0.7); }
      50% { box-shadow: 0 0 8px 4px rgba(35, 134, 54, 0.4); }
    }

    .model-tag .close {
      cursor: pointer;
      opacity: 0.6;
      font-size: 14px;
      line-height: 1;
    }

    .model-tag .close:hover {
      opacity: 1;
      color: #f85149;
    }

    .model-tag.groq { border-color: rgba(63, 185, 80, 0.4); }
    .model-tag.deepseek { border-color: rgba(88, 166, 255, 0.4); }
    .model-tag.openai { border-color: rgba(163, 113, 247, 0.4); }
    .model-tag.anthropic { border-color: rgba(212, 165, 116, 0.4); }
    .model-tag.xai { border-color: rgba(248, 81, 73, 0.4); }
    .model-tag.mistral { border-color: rgba(255, 123, 0, 0.4); }
    .model-tag.google { border-color: rgba(66, 133, 244, 0.4); }
    .model-tag.together { border-color: rgba(232, 121, 249, 0.4); }
    .model-tag.cohere { border-color: rgba(57, 211, 83, 0.4); }

    .add-all-btn {
      padding: 2px 8px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      color: #8b949e;
      cursor: pointer;
      font-size: 11px;
      transition: all 150ms;
    }

    .add-all-btn:hover {
      background: #58a6ff;
      border-color: #58a6ff;
      color: white;
    }

    .add-model-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      background: #21262d;
      border: 1px dashed #30363d;
      border-radius: 4px;
      color: #6e7681;
      cursor: pointer;
      font-size: 14px;
      transition: all 150ms;
    }

    .add-model-btn:hover {
      background: #30363d;
      border-color: #58a6ff;
      color: #58a6ff;
    }

    .model-dropdown {
      position: relative;
    }

    .model-dropdown-menu {
      position: absolute;
      top: 100%;
      left: 0;
      margin-top: 4px;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      z-index: 100;
      min-width: 180px;
      max-height: 20dvh;
      overflow-y: auto;
    }

    .model-option {
      padding: 8px 12px;
      font-size: 12px;
      color: #c9d1d9;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }

    .model-option:hover {
      background: #21262d;
    }

    .model-option.selected {
      background: #1f6feb33;
      border-left: 2px solid #58a6ff;
      color: #58a6ff;
      cursor: default;
    }

    .model-option.selected:hover {
      background: #1f6feb33;
    }

    .model-option:first-child {
      border-radius: 6px 6px 0 0;
    }

    .model-option:last-child {
      border-radius: 0 0 6px 6px;
    }

    .model-option .provider {
      font-size: 10px;
      color: #6e7681;
    }

    .model-option.selected .provider {
      color: #58a6ff99;
    }

    .stats {
      display: flex;
      gap: 16px;
      font-size: 12px;
    }

    .stat {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 6px;
    }

    .stat-label {
      color: #6e7681;
    }

    .stat-value {
      color: #f0f6fc;
      font-weight: 700;
      font-size: 14px;
    }

    .content {
      padding: 20px;
    }

    .empty {
      padding: 60px 32px;
      text-align: center;
      color: #6e7681;
      font-size: 14px;
    }

    .empty-icon {
      font-size: 48px;
      margin-bottom: 16px;
      opacity: 0.5;
    }

    .running-banner {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 14px;
      background: linear-gradient(90deg, rgba(88, 166, 255, 0.08) 0%, rgba(163, 113, 247, 0.08) 100%);
      border: 1px solid rgba(88, 166, 255, 0.2);
      border-radius: 6px;
      margin-bottom: 16px;
      font-size: 12px;
      line-height: 1.5;
    }

    .running-banner:not(.idle) {
      animation: glow-pulse 2s ease-in-out infinite;
    }

    .running-banner.idle {
      background: rgba(110, 118, 129, 0.1);
      border-color: rgba(110, 118, 129, 0.2);
    }

    .notes-tabs {
      display: flex;
      gap: 0;
      margin-bottom: 0;
      border-bottom: 1px solid #30363d;
    }

    .notes-tab {
      padding: 8px 16px;
      font-size: 12px;
      font-weight: 600;
      color: #8b949e;
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      cursor: pointer;
      transition: all 0.15s;
    }

    .notes-tab:hover {
      color: #c9d1d9;
    }

    .notes-tab.active {
      color: #3fb950;
      border-bottom-color: #3fb950;
    }

    .notes-tab.notes-active {
      color: #58a6ff;
      border-bottom-color: #58a6ff;
    }

    .narrative-box {
      background: rgba(63, 185, 80, 0.08);
      border: 1px solid rgba(63, 185, 80, 0.25);
      border-radius: 0 0 8px 8px;
      padding: 12px 16px;
      margin-bottom: 16px;
    }

    .narrative-box.notes-box {
      background: rgba(88, 166, 255, 0.08);
      border-color: rgba(88, 166, 255, 0.25);
    }

    .narrative-header {
      font-size: 11px;
      font-weight: 700;
      color: #3fb950;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .expand-btn {
      background: none;
      border: none;
      color: #6e7681;
      cursor: pointer;
      font-size: 10px;
      padding: 2px 6px;
    }

    .expand-btn:hover {
      color: #c9d1d9;
    }

    .narrative-content {
      font-size: 13px;
      color: #c9d1d9;
      line-height: 1.6;
      white-space: pre-wrap;
      max-height: 150px;
      overflow-y: auto;
      cursor: pointer;
    }

    .narrative-content:hover {
      background: rgba(63, 185, 80, 0.05);
      border-radius: 4px;
      margin: -4px;
      padding: 4px;
    }

    .narrative-edit {
      width: 100%;
      min-height: 250px;
      max-height: 60vh;
      padding: 12px;
      background: #0d1117;
      border: 1px solid rgba(63, 185, 80, 0.4);
      border-radius: 6px;
      font-size: 13px;
      font-family: inherit;
      color: #c9d1d9;
      line-height: 1.6;
      resize: vertical;
    }

    .narrative-edit:focus {
      outline: none;
      border-color: #3fb950;
      box-shadow: 0 0 0 2px rgba(63, 185, 80, 0.2);
    }

    .narrative-header {
      cursor: pointer;
    }

    .narrative-header:hover {
      color: #56d364;
    }

    .running-indicator {
      margin-left: 8px;
      color: #f85149;
      animation: blink 1s ease-in-out infinite;
    }

    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    @keyframes glow-pulse {
      0%, 100% { box-shadow: 0 0 20px rgba(88, 166, 255, 0.1); }
      50% { box-shadow: 0 0 40px rgba(88, 166, 255, 0.2); }
    }

    .spinner {
      width: 20px;
      height: 20px;
      border: 3px solid #21262d;
      border-top-color: #58a6ff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .running-text {
      font-size: 12px;
      font-weight: 500;
      color: #8b949e;
      white-space: pre-wrap;
      flex: 1;
    }

    .running-banner:not(.idle) .running-text {
      color: #58a6ff;
    }

    /* Interrogation transcript style */
    .transcript {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .exchange {
      border-left: 3px solid #30363d;
      padding-left: 20px;
      animation: fade-in 0.3s ease-out;
    }

    @keyframes fade-in {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .exchange.active {
      border-left-color: #58a6ff;
    }

    .exchange.past {
      opacity: 0.8;
    }

    .exchange.past .question-text {
      background: rgba(110, 118, 129, 0.1);
      border-color: rgba(110, 118, 129, 0.2);
    }

    .interrogator {
      margin-bottom: 16px;
    }

    .interrogator-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 10px;
      font-weight: 700;
      color: #f85149;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 8px;
    }

    .question-text {
      font-size: 15px;
      font-weight: 500;
      color: #f0f6fc;
      line-height: 1.5;
      padding: 12px 16px;
      background: rgba(248, 81, 73, 0.05);
      border: 1px solid rgba(248, 81, 73, 0.2);
      border-radius: 8px;
    }

    .responses-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .subject-response {
      padding: 16px;
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 8px;
      transition: all 200ms;
    }

    .subject-response:hover {
      border-color: #30363d;
      background: #1c2128;
    }

    .subject-response.new {
      animation: highlight-new 1s ease-out;
    }

    @keyframes highlight-new {
      0% { background: rgba(88, 166, 255, 0.2); border-color: #58a6ff; }
      100% { background: #161b22; border-color: #21262d; }
    }

    .response-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }

    .subject-label {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .model-badge {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: #21262d;
      border-radius: 6px;
    }

    .model-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .model-indicator.groq { background: #3fb950; box-shadow: 0 0 8px rgba(63, 185, 80, 0.5); }
    .model-indicator.deepseek { background: #58a6ff; box-shadow: 0 0 8px rgba(88, 166, 255, 0.5); }
    .model-indicator.openai { background: #a371f7; box-shadow: 0 0 8px rgba(163, 113, 247, 0.5); }
    .model-indicator.anthropic { background: #d4a574; box-shadow: 0 0 8px rgba(212, 165, 116, 0.5); }
    .model-indicator.xai { background: #f85149; box-shadow: 0 0 8px rgba(248, 81, 73, 0.5); }
    .model-indicator.mistral { background: #ff7b00; box-shadow: 0 0 8px rgba(255, 123, 0, 0.5); }
    .model-indicator.google { background: #4285f4; box-shadow: 0 0 8px rgba(66, 133, 244, 0.5); }
    .model-indicator.together { background: #e879f9; box-shadow: 0 0 8px rgba(232, 121, 249, 0.5); }
    .model-indicator.cohere { background: #39d353; box-shadow: 0 0 8px rgba(57, 211, 83, 0.5); }

    .model-name {
      font-size: 12px;
      font-weight: 600;
      color: #8b949e;
    }

    .run-badge {
      font-size: 10px;
      color: #6e7681;
      padding: 2px 6px;
      background: #0d1117;
      border-radius: 4px;
    }

    .response-status {
      font-size: 10px;
      padding: 4px 8px;
      border-radius: 4px;
    }

    .response-status.talking {
      background: rgba(63, 185, 80, 0.15);
      color: #3fb950;
    }

    .response-status.refused {
      background: rgba(248, 81, 73, 0.15);
      color: #f85149;
    }

    .response-text {
      font-size: 14px;
      color: #c9d1d9;
      line-height: 1.6;
    }

    .response-text.refusal {
      color: #6e7681;
      font-style: italic;
    }

    .extracted-intel {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #21262d;
    }

    .intel-label {
      font-size: 10px;
      color: #6e7681;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-right: 4px;
    }

    .intel-item {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      background: rgba(63, 185, 80, 0.1);
      border: 1px solid rgba(63, 185, 80, 0.3);
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      color: #3fb950;
    }

    .intel-item.hot {
      background: rgba(248, 81, 73, 0.1);
      border-color: rgba(248, 81, 73, 0.3);
      color: #f85149;
      animation: pulse-hot 1.5s ease-in-out infinite;
    }

    @keyframes pulse-hot {
      0%, 100% { box-shadow: 0 0 0 rgba(248, 81, 73, 0); }
      50% { box-shadow: 0 0 12px rgba(248, 81, 73, 0.3); }
    }

    .intel-item.private {
      background: rgba(163, 113, 247, 0.15);
      border-color: rgba(163, 113, 247, 0.4);
      color: #a371f7;
    }

    .intel-item.private::before {
      content: "PRIVATE ";
      font-size: 9px;
      opacity: 0.8;
    }

    .intel-item.public {
      background: rgba(139, 148, 158, 0.1);
      border-color: rgba(139, 148, 158, 0.3);
      color: #8b949e;
      opacity: 0.7;
    }

    .intel-item.introduced {
      background: rgba(110, 118, 129, 0.08);
      border-color: rgba(110, 118, 129, 0.2);
      color: #6e7681;
      opacity: 0.6;
      font-weight: 400;
    }

    .intel-item.introduced::before {
      content: "↩ ";
      font-size: 10px;
      opacity: 0.7;
    }

    .intel-item.discovered {
      background: rgba(63, 185, 80, 0.15);
      border-color: rgba(63, 185, 80, 0.4);
      color: #3fb950;
      font-weight: 600;
    }

    .intel-item.discovered::before {
      content: "★ ";
      font-size: 10px;
    }
  `;

  @state()
  private _probeState: ProbeState = probeState.get();

  @state()
  private pastTranscript: TranscriptQuestion[] = [];

  @state()
  private isLoadingTranscript = false;

  @state()
  private showModelDropdown = false;

  @state()
  private modelsExpanded = false;

  @state()
  private editingUserNotes = false;

  @state()
  private notesTab: 'theory' | 'notes' = 'theory';

  @state()
  private theoryExpanded = false;

  @state()
  private availableModels: ModelInfo[] = [];

  @state()
  private _tick = 0;  // Forces re-render on interval

  private _unsubscribe?: () => void;

  private handleClickOutside = (e: MouseEvent) => {
    if (this.showModelDropdown) {
      const dropdown = this.shadowRoot?.querySelector('.model-dropdown');
      if (dropdown && !dropdown.contains(e.target as Node)) {
        this.showModelDropdown = false;
      }
    }
  };

  private _timestampInterval: number | null = null;

  connectedCallback() {
    super.connectedCallback();
    // Update timestamps every 30 seconds - increment tick to force re-render
    this._timestampInterval = window.setInterval(() => { this._tick++; }, 30000);
    // Fetch available models from API (not hardcoded)
    getAvailableModels().then(models => {
      this.availableModels = models;
    }).catch(err => {
      console.error('Failed to fetch models:', err);
    });
    this._unsubscribe = probeState.subscribe((s) => {
      const prevNarrative = this._probeState.narrative;
      const prevActive = this._probeState.activeModel;
      this._probeState = s;
      this.requestUpdate();  // Force re-render for active model highlighting

      // Log narrative changes
      if (s.narrative !== prevNarrative) {
        console.log('[response-stream] narrative changed:', s.narrative?.substring(0, 50) + '...');
      }

      // Scroll active model into view when it changes
      if (s.activeModel && s.activeModel !== prevActive) {
        requestAnimationFrame(() => {
          const activeTag = this.shadowRoot?.querySelector(`[data-model="${s.activeModel}"]`);
          activeTag?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
        });
      }
    });
    document.addEventListener('click', this.handleClickOutside);
    this.loadPastTranscript();
  }

  async updated(changed: Map<string, unknown>) {
    if (changed.has('projectName') && this.projectName) {
      await this.loadPastTranscript();
    }
  }

  private async loadPastTranscript() {
    if (!this.projectName) return;
    this.isLoadingTranscript = true;
    try {
      const data = await getTranscript(this.projectName);
      this.pastTranscript = data.transcript;
    } catch (e) {
      console.error('Failed to load transcript:', e);
    } finally {
      this.isLoadingTranscript = false;
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this._unsubscribe?.();
    if (this._timestampInterval) clearInterval(this._timestampInterval);
    document.removeEventListener('click', this.handleClickOutside);
  }

  private async removeModel(model: string) {
    const current = probeState.get();
    const remaining = current.selectedModels.filter(m => m !== model);

    if (remaining.length === 0 && current.isRunning) {
      // Stop the probe if we remove the last model
      probeState.update(s => ({
        ...s,
        selectedModels: remaining,
        isRunning: false
      }));
      // Dispatch stop event for parent to handle abort
      this.dispatchEvent(new CustomEvent('stop-probe', { bubbles: true, composed: true }));
    } else {
      probeState.update(s => ({
        ...s,
        selectedModels: remaining
      }));
    }
    // Save to project
    if (this.projectName) {
      updateProject(this.projectName, { selected_models: remaining });
    }
  }

  private async addModel(model: string) {
    const newModels = probeState.get().selectedModels.includes(model)
      ? probeState.get().selectedModels
      : [...probeState.get().selectedModels, model];

    probeState.update(s => ({
      ...s,
      selectedModels: newModels
    }));
    this.showModelDropdown = false;

    // Save to project
    if (this.projectName) {
      updateProject(this.projectName, { selected_models: newModels });
    }
  }

  private async addAllModels() {
    const allModelIds = this.availableModels.map(m => m.id);

    probeState.update(s => ({
      ...s,
      selectedModels: allModelIds
    }));

    // Save to project
    if (this.projectName) {
      updateProject(this.projectName, { selected_models: allModelIds });
    }
  }

  // Working theory is now read-only (AI-generated)

  private async saveUserNotes(e: Event) {
    const textarea = e.target as HTMLTextAreaElement;
    const newNotes = textarea.value;

    probeState.update(s => ({
      ...s,
      userNotes: newNotes
    }));

    this.editingUserNotes = false;

    // Save to project
    if (this.projectName) {
      updateProject(this.projectName, { user_notes: newNotes });
    }
  }

  private getModelProvider(model: string): string {
    if (model.includes('groq')) return 'groq';
    if (model.includes('deepseek')) return 'deepseek';
    if (model.includes('openai')) return 'openai';
    if (model.includes('anthropic') || model.includes('claude')) return 'anthropic';
    if (model.includes('xai') || model.includes('grok')) return 'xai';
    if (model.includes('mistral')) return 'mistral';
    if (model.includes('google') || model.includes('gemini')) return 'google';
    if (model.includes('together') || model.includes('meta-llama')) return 'together';
    if (model.includes('cohere') || model.includes('command')) return 'cohere';
    return 'other';
  }

  private getModelDisplayName(model: string): string {
    const parts = model.split('/');
    return parts[parts.length - 1].split('-').slice(0, 2).join(' ');
  }

  private formatTimeAgo(timestamp: number): string {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 10) return 'just now';
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  }

  private extractHeadline(narrative: string): { headline: string; subhead: string; analysis: string } {
    if (!narrative) return { headline: 'No theory generated yet', subhead: 'Run the probe to build one.', analysis: '' };

    // Try to parse structured HEADLINE/SUBHEAD/ANALYSIS format
    const headlineMatch = narrative.match(/HEADLINE:?\s*\n?([^\n]+)/i);
    // Capture subhead until ANALYSIS section or double newline
    const subheadMatch = narrative.match(/SUBHEAD:?\s*\n?([\s\S]*?)(?=\n\n|\nANALYSIS|\nCLAIMS|\nNEXT|\nKEY|\n#|$)/i);
    // Capture analysis section
    const analysisMatch = narrative.match(/ANALYSIS:?\s*\n?([\s\S]*?)$/i);

    if (headlineMatch) {
      const headline = headlineMatch[1].trim().replace(/^\[|\]$/g, '').replace(/^\*+|\*+$/g, '');
      const subhead = subheadMatch
        ? subheadMatch[1].trim().replace(/^\[|\]$/g, '').replace(/^\*+|\*+$/g, '')
        : '';
      const analysis = analysisMatch
        ? analysisMatch[1].trim()
        : '';
      return { headline, subhead, analysis };
    }

    // Fallback: extract headline from first good line, subhead from rest
    const skipPatterns = [
      /list specific/i, /your task/i, /format:/i, /output/i,
      /be specific/i, /extract/i, /analyze/i, /cross-reference/i,
      /^#+\s/, // Skip markdown headers
      /^project\/program/i, /^people\/org/i, /^dates/i, /^locations/i,
      /^key relation/i, /^unique claim/i, /^pattern/i, /^question/i,
      /names surfaced/i, /organizations/i, /timeline/i
    ];

    const lines = narrative.split('\n');
    for (let i = 0; i < lines.length; i++) {
      let content = lines[i].trim().replace(/\*\*/g, '').replace(/^[•\-\*]\s*/, '').replace(/^#+\s*/, '');
      // Skip headers, short lines, lines ending with colon
      if (content.length > 25 && !content.endsWith(':') && !content.startsWith('#') && !skipPatterns.some(p => p.test(content))) {
        // Check if this line has a sentence ending
        const sentenceEnd = content.search(/[.!?]\s|[.!?]$/);
        if (sentenceEnd > 30) {
          // Line has sentence - split it
          const headline = content.substring(0, sentenceEnd + 1);
          const restOfLine = content.substring(sentenceEnd + 1).trim();
          const remaining = lines.slice(i + 1).map(l => l.trim()).filter(l => l.length > 0).join('\n');
          const subhead = restOfLine || remaining.substring(0, 300);
          return { headline, subhead, analysis: remaining };
        }
        // No sentence in this line - use it as headline, grab next content as subhead/analysis
        const remaining = lines.slice(i + 1)
          .map(l => l.trim())
          .filter(l => l.length > 0)
          .join('\n');
        const subhead = remaining.substring(0, 300);
        return { headline: content, subhead, analysis: remaining };
      }
    }

    return { headline: 'Analyzing findings...', subhead: '', analysis: '' };
  }

  private groupResponsesByQuestion(responses: ProbeResponse[]): Map<number, ProbeResponse[]> {
    const groups = new Map<number, ProbeResponse[]>();
    for (const response of responses) {
      const existing = groups.get(response.question_index) || [];
      groups.set(response.question_index, [...existing, response]);
    }
    return groups;
  }

  private isHotEntity(entity: string): boolean {
    // Count how many times this entity appears across all responses
    const count = this._probeState.responses.filter(r =>
      r.entities?.includes(entity)
    ).length;
    return count >= 3;
  }

  private getEntityVerificationClass(entity: string): string {
    const verification = this._probeState.entityVerification;
    if (!verification) return '';

    // Check if entity is in verified (PUBLIC) list
    const isPublic = verification.verified?.some(v =>
      (typeof v === 'string' ? v : v.entity)?.toLowerCase() === entity.toLowerCase()
    );
    if (isPublic) return 'public';

    // Check if entity is in unverified (PRIVATE) list - the interesting ones!
    const isPrivate = verification.unverified?.some(v =>
      (typeof v === 'string' ? v : v.entity)?.toLowerCase() === entity.toLowerCase()
    );
    if (isPrivate) return 'private';

    return '';
  }

  render() {
    const { responses, questions, isRunning } = this._probeState;
    const grouped = this.groupResponsesByQuestion(responses);

    // Combine current session + past transcripts
    const hasCurrentSession = responses.length > 0;
    const hasPastData = this.pastTranscript.length > 0;
    const totalResponses = responses.length + this.pastTranscript.reduce((sum, q) => sum + q.responses.length, 0);

    return html`
      <div class="card">
        <div class="header">
          <div class="header-title">
            <h3>${isRunning ? 'Interrogating' : 'Interrogation'}</h3>
            ${isRunning ? html`
              <div class="live-badge">
                <span class="live-dot"></span>
                Live
              </div>
            ` : null}
            <button
              class="model-tags-toggle"
              @click=${() => this.modelsExpanded = !this.modelsExpanded}
              title="${this.modelsExpanded ? 'Collapse' : 'Expand'} models"
            >${this._probeState.selectedModels.length} models ${this.modelsExpanded ? '▲' : '▼'}</button>
            <div class="model-tags ${this.modelsExpanded ? 'expanded' : ''}">
              ${this._probeState.selectedModels.map(model => {
                const provider = this.getModelProvider(model);
                const name = this.getModelDisplayName(model);
                const isActive = this._probeState.activeModel === model;
                return html`
                  <span
                    class="model-tag ${provider} ${isActive ? 'active' : ''}"
                    data-model="${model}"
                  >
                    <span class="name">${name}</span>
                    <span class="close" @click=${() => this.removeModel(model)}>×</span>
                  </span>
                `;
              })}
              <button
                class="add-all-btn"
                @click=${this.addAllModels}
                title="Add all models"
              >All</button>
              <div class="model-dropdown">
                <button
                  class="add-model-btn"
                  @click=${(e: Event) => { e.stopPropagation(); this.showModelDropdown = !this.showModelDropdown; }}
                  title="Add model"
                >+</button>
                ${this.showModelDropdown ? html`
                  <div class="model-dropdown-menu">
                    ${this.availableModels.map(m => {
                      const isSelected = this._probeState.selectedModels.includes(m.id);
                      return html`
                        <div
                          class="model-option ${isSelected ? 'selected' : ''}"
                          @click=${(e: Event) => { e.stopPropagation(); if (!isSelected) this.addModel(m.id); }}
                        >
                          <span>${m.name}</span>
                          <span class="provider">${m.provider}</span>
                        </div>
                      `;
                    })}
                  </div>
                ` : null}
              </div>
            </div>
          </div>
          <div class="stats">
            <div class="stat">
              <span class="stat-label">Responses</span>
              <span class="stat-value">${totalResponses}</span>
            </div>
            <div class="stat">
              <span class="stat-label">Questions</span>
              <span class="stat-value">${questions.length + this.pastTranscript.length}</span>
            </div>
          </div>
        </div>

        <div class="content">
          <!-- Tabbed notes section -->
          <div class="notes-tabs">
            <button
              class="notes-tab ${this.notesTab === 'theory' ? 'active' : ''}"
              @click=${() => this.notesTab = 'theory'}
            >
              📓 Working Theory
              ${isRunning ? html`<span style="color: #3fb950; margin-left: 4px;">●</span>` : ''}
            </button>
            <button
              class="notes-tab ${this.notesTab === 'notes' ? 'notes-active' : ''}"
              @click=${() => this.notesTab = 'notes'}
            >
              💡 Your Notes
            </button>
          </div>

          ${this.notesTab === 'theory' ? html`
            <!-- Working Theory (AI-generated, read-only with expand) -->
            <div class="narrative-box">
              <div class="narrative-header">
                <span style="font-weight: 400; color: #8b949e; font-size: 11px;" data-tick=${this._tick}>
                  ${this._probeState.narrativeUpdated
                    ? `Updated ${this.formatTimeAgo(this._probeState.narrativeUpdated)}`
                    : ''}
                </span>
                <button class="expand-btn" @click=${() => this.theoryExpanded = !this.theoryExpanded}>
                  ${this.theoryExpanded ? '▲' : '▼'}
                </button>
              </div>
              ${(() => {
                const { headline, subhead, analysis } = this.extractHeadline(this._probeState.narrative);
                return this.theoryExpanded ? html`
                  <div class="narrative-headline" style="font-size: 18px; font-weight: 700; color: #3fb950; line-height: 1.3; margin-bottom: 8px;">
                    ${headline}
                  </div>
                  ${subhead ? html`<div style="font-size: 14px; color: #c9d1d9; line-height: 1.5; margin-bottom: 12px;">${subhead}</div>` : ''}
                  ${analysis ? html`
                    <div class="narrative-content" style="cursor: default; border-top: 1px solid #30363d; padding-top: 12px; margin-top: 8px; font-size: 13px; color: #8b949e; line-height: 1.6; white-space: pre-wrap;">
                      ${analysis}
                    </div>
                  ` : ''}
                ` : html`
                  <div class="narrative-headline" style="font-size: 18px; font-weight: 700; color: #3fb950; line-height: 1.3;">
                    ${headline}
                  </div>
                  ${subhead ? html`<div style="font-size: 13px; color: #c9d1d9; line-height: 1.4; margin-top: 8px;">${subhead}</div>` : ''}
                `;
              })()}
            </div>
          ` : html`
            <!-- User Notes (editable, fed back to AI) -->
            <div class="narrative-box notes-box">
              <div class="narrative-header" style="color: #58a6ff;">
                <span>Your hunches & leads (fed to AI)</span>
              </div>
              ${this.editingUserNotes ? html`
                <textarea
                  class="narrative-edit"
                  style="border-color: #58a6ff;"
                  .value=${this._probeState.userNotes || ''}
                  @blur=${this.saveUserNotes}
                  @keydown=${(e: KeyboardEvent) => { if (e.key === 'Escape') this.editingUserNotes = false; }}
                  placeholder="Add your wild theories, hunches, things to explore. This is fed back to the AI to guide its questioning."
                ></textarea>
              ` : html`
                <div class="narrative-content" @click=${() => this.editingUserNotes = true} style="cursor: pointer;">
                  ${this._probeState.userNotes || 'Click to add your hunches, wild theories, leads to explore...'}
                </div>
              `}
            </div>
          `}

          ${!hasCurrentSession && !hasPastData && !isRunning && !this.isLoadingTranscript ? html`
            <div class="empty">
              <div class="empty-icon">🎯</div>
              <div>Click "Run Probe" to begin interrogation</div>
              <div style="font-size: 12px; margin-top: 8px; color: #484f58;">
                Questions will be generated and fired at the selected models
              </div>
            </div>
          ` : null}

          ${isRunning && !hasCurrentSession && !this.isLoadingTranscript ? html`
            <div class="running-banner">
              <div class="spinner"></div>
              <div class="running-text">
                Starting probe... generating questions and connecting to models
              </div>
            </div>
          ` : null}

          ${this.isLoadingTranscript ? html`
            <div class="empty">
              <div class="spinner"></div>
              <div style="margin-top: 16px;">Loading past interrogations...</div>
            </div>
          ` : null}

          <div class="transcript">
            <!-- Current session responses (newest first) -->
            ${Array.from(grouped.entries()).reverse().map(([questionIndex, questionResponses]) => {
              // Get question text from response itself (reliable), fallback to questions array
              const questionText = questionResponses[0]?.question || questions[questionIndex]?.question || `Question ${questionIndex + 1}`;
              const questionMeta = questions[questionIndex];  // For template/color info
              const isActive = isRunning && questionIndex === Math.max(...Array.from(grouped.keys()));
              return this.renderExchange(questionText, questionResponses, isActive, true, questionMeta);
            })}

            <!-- Past transcript -->
            ${this.pastTranscript.slice().reverse().map(q =>
              this.renderExchange(q.question, q.responses.map(r => ({
                ...r,
                question: q.question,
                question_index: q.question_index
              })), false, false)
            )}
          </div>
        </div>
      </div>
    `;
  }

  private renderExchange(question: string, responses: ProbeResponse[], isActive: boolean, isCurrent: boolean, meta?: GeneratedQuestion) {
    const templateStyle = meta?.color
      ? `background: rgba(${parseInt(meta.color.slice(1, 3), 16)}, ${parseInt(meta.color.slice(3, 5), 16)}, ${parseInt(meta.color.slice(5, 7), 16)}, 0.15); border: 1px solid rgba(${parseInt(meta.color.slice(1, 3), 16)}, ${parseInt(meta.color.slice(3, 5), 16)}, ${parseInt(meta.color.slice(5, 7), 16)}, 0.3); color: ${meta.color}; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-left: 8px;`
      : '';

    return html`
      <div class="exchange ${isActive ? 'active' : ''} ${isCurrent ? 'current' : 'past'}">
        <div class="interrogator">
          <div class="interrogator-label">
            <span>⚡</span> Interrogator
            ${meta?.template ? html`<span style=${templateStyle}>${meta.template}</span>` : ''}
            ${!isCurrent ? html`<span style="opacity: 0.5; font-size: 9px;">(past)</span>` : ''}
          </div>
          <div class="question-text">
            "${question}"
          </div>
        </div>

        <div class="responses-list">
          ${responses.slice(-10).reverse().map((r: any, i: number) => {
            const isNew = i === 0 && isActive && isCurrent;
            return html`
              <div class="subject-response ${isNew ? 'new' : ''}">
                <div class="response-header">
                  <div class="subject-label">
                    <div class="model-badge">
                      <span class="model-indicator ${this.getModelProvider(r.model)}"></span>
                      <span class="model-name">${this.getModelDisplayName(r.model)}</span>
                    </div>
                    <span class="run-badge">Run #${(r.run_index || 0) + 1}</span>
                  </div>
                  <span class="response-status ${r.is_refusal ? 'refused' : 'talking'}">
                    ${r.is_refusal ? '🚫 Refused' : '💬 Talking'}
                  </span>
                </div>
                <div class="response-text ${r.is_refusal ? 'refusal' : ''}">
                  ${r.response}
                </div>
                ${(r.discovered_entities?.length || r.introduced_entities?.length || r.entities?.length) ? html`
                  <div class="extracted-intel">
                    ${r.discovered_entities?.length ? html`
                      <span class="intel-label">Discovered:</span>
                      ${r.discovered_entities.slice(0, 6).map((e: string) => html`
                        <span class="intel-item discovered ${this.isHotEntity(e) ? 'hot' : ''} ${this.getEntityVerificationClass(e)}">${e}</span>
                      `)}
                      ${r.discovered_entities.length > 6 ? html`<span class="intel-item" style="opacity: 0.5;">+${r.discovered_entities.length - 6}</span>` : ''}
                    ` : ''}
                    ${r.introduced_entities?.length ? html`
                      <span class="intel-label" style="margin-left: 8px;">From query:</span>
                      ${r.introduced_entities.slice(0, 4).map((e: string) => html`
                        <span class="intel-item introduced">${e}</span>
                      `)}
                      ${r.introduced_entities.length > 4 ? html`<span class="intel-item introduced" style="opacity: 0.5;">+${r.introduced_entities.length - 4}</span>` : ''}
                    ` : ''}
                    ${!r.discovered_entities?.length && !r.introduced_entities?.length && r.entities?.length ? html`
                      <span class="intel-label">Extracted:</span>
                      ${r.entities.slice(0, 8).map((e: string) => html`
                        <span class="intel-item ${this.isHotEntity(e) ? 'hot' : ''} ${this.getEntityVerificationClass(e)}">${e}</span>
                      `)}
                      ${r.entities.length > 8 ? html`<span class="intel-item" style="opacity: 0.5;">+${r.entities.length - 8} more</span>` : ''}
                    ` : ''}
                  </div>
                ` : null}
              </div>
            `;
          })}
          ${responses.length > 10 ? html`
            <div style="text-align: center; padding: 8px; color: #6e7681; font-size: 12px;">
              +${responses.length - 10} more responses...
            </div>
          ` : null}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'response-stream': ResponseStream;
  }
}
