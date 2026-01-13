import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { appState, type AppState } from '../state';

@customElement('app-shell')
export class AppShell extends LitElement {
  static styles = css`
    :host {
      display: block;
      min-height: 100vh;
      background: var(--bg-primary, #0d1117);
    }
  `;

  @state()
  private _appState: AppState = appState.get();

  private _unsubscribe?: () => void;

  connectedCallback(): void {
    super.connectedCallback();
    this._unsubscribe = appState.subscribe((s) => {
      this._appState = s;
    });
  }

  disconnectedCallback(): void {
    super.disconnectedCallback();
    this._unsubscribe?.();
  }

  render() {
    return html`
      ${this._appState.currentView === 'projects'
        ? html`<home-page></home-page>`
        : html`<project-view .projectName=${this._appState.currentProject}></project-view>`}
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'app-shell': AppShell;
  }
}
