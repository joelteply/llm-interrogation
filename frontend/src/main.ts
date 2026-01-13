import './styles/main.scss';
import { initFromUrl } from './state';

// Import components
import './components/app-shell';
import './components/home-page';
import './components/project-card';
import './components/project-view';
import './components/probe-controls';
import './components/question-queue';
import './components/response-stream';
import './components/findings-panel';
import './components/tag-input';
import './components/word-cloud';

// Initialize routing from URL
initFromUrl();
