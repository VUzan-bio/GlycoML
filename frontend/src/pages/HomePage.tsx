import React from 'react';
import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="tag">Phase 3 Dashboard</p>
          <h1>GlycoML Fc-Fcgr Visual Lab</h1>
          <p className="subtitle">
            Explore Fc-Fcgr binding across glycoforms with structural context and
            quantitative affinity metrics.
          </p>
        </div>
        <div className="hero-card">
          <h3>Quick actions</h3>
          <ul>
            <li>
              <span>Single glycoform</span>
              <Link to="/single">Open viewer</Link>
            </li>
            <li>
              <span>Compare glycoforms</span>
              <Link to="/compare">Run comparison</Link>
            </li>
          </ul>
        </div>
      </header>

      <main className="content">
        <section className="panel">
          <div className="panel-header">
            <h2>What this view provides</h2>
            <p>
              Phase 3 focuses on Fcgr binding signals derived from Fc sequences
              and glycan structures. Use the viewer to inspect one glycoform or
              compare multiple glycoforms side by side.
            </p>
          </div>
          <div className="metrics-grid">
            <div className="metric">
              <span>Data source</span>
              <strong>phase3_fcgr_merged.csv</strong>
            </div>
            <div className="metric">
              <span>Primary outputs</span>
              <strong>Kd, log Kd, delta G</strong>
            </div>
            <div className="metric">
              <span>Visualization stack</span>
              <strong>Molstar + FastAPI</strong>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Next steps</h2>
            <p>
              Keep the Phase 2 glycan encoder separate. Phase 3 uses the Fcgr-Fc
              binding table and optional antibody metadata for context only.
            </p>
          </div>
          <div className="cta-row">
            <Link className="cta" to="/single">
              Launch single view
            </Link>
            <Link className="cta outline" to="/compare">
              Compare glycoforms
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
