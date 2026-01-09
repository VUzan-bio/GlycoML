import React from 'react';
import { Columns3, Eye } from 'lucide-react';
import { Link } from 'react-router-dom';
import buttonStyles from '../components/ui/Button.module.css';

export default function HomePage() {
  const primaryLink = `${buttonStyles.button} ${buttonStyles.primary} ${buttonStyles.withIcon}`;
  const secondaryLink = `${buttonStyles.button} ${buttonStyles.secondary} ${buttonStyles.withIcon}`;

  return (
    <div className="app-shell">
      <main className="main-grid" role="main">
        <aside className="sidebar">
          <div className="sidebar-panel">
            <div className="section-title">Explorer</div>
            <p className="helper-text">Choose a view to begin.</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '16px' }}>
              <Link className={primaryLink} to="/single">
                <span className={buttonStyles.iconGraphic} aria-hidden="true">
                  <Eye size={16} />
                </span>
                Viewer
              </Link>
              <Link className={secondaryLink} to="/compare">
                <span className={buttonStyles.iconGraphic} aria-hidden="true">
                  <Columns3 size={16} />
                </span>
                Compare
              </Link>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
