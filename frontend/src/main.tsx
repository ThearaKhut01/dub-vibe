import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
// After index.css: the light theme re-points the palette utilities the app uses.
import './theme-light.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
