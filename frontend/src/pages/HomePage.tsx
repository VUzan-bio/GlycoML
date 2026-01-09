import React from 'react';
import { Activity, Atom, Columns3, FileText, Play } from 'lucide-react';
import { Link } from 'react-router-dom';
import buttonStyles from '../components/ui/Button.module.css';
import Card from '../components/ui/Card';
import styles from './HomePage.module.css';

export default function HomePage() {
  const primaryButton = `${buttonStyles.button} ${buttonStyles.primary} ${buttonStyles.withIcon}`;

  return (
    <div className={styles.shell}>
      <main className={styles.main} id="main-content" role="main">
        <section className={styles.hero}>
          <div className={styles.heroContent}>
            <span className={styles.heroBadge}>GlycoML Phase 3</span>
            <h1>Predict lectin-glycan binding with deep learning</h1>
            <p className={styles.heroSub}>
              AI-powered prediction of Fc-FcγR interactions across 13 glycoforms with
              live structural context and quantitative affinity metrics.
            </p>
            <div className={styles.heroActions}>
              <Link to="/predict" className={primaryButton}>
                <span className={buttonStyles.iconGraphic} aria-hidden="true">
                  <Play size={18} />
                </span>
                Start Prediction
              </Link>
              <Link to="/compare" className={styles.secondaryLink}>
                Compare View
              </Link>
            </div>
          </div>
          <Card className={styles.heroCard}>
            <div className={styles.heroCardHeader}>
              <Atom size={20} color="#5771FE" aria-hidden="true" />
              <span>Structural Preview</span>
            </div>
            <div className={styles.heroCardBody}>
              <div className={styles.previewLine} />
              <div className={styles.previewLine} />
              <div className={styles.previewLineShort} />
              <div className={styles.previewCanvas}>
                <div className={styles.previewHint}>Mol* viewer ready for loading</div>
              </div>
            </div>
          </Card>
        </section>

        <section className={styles.featureGrid}>
          <Card className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <Activity size={18} color="#5771FE" aria-hidden="true" />
              <h3>Multi-task learning</h3>
            </div>
            <p>Combines binary classification and regression to capture binding probability and K_D.</p>
          </Card>
          <Card className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <FileText size={18} color="#5771FE" aria-hidden="true" />
              <h3>Structural integration</h3>
            </div>
            <p>ESM2 embeddings fused with 3D context for Fc-FcγR contact-aware predictions.</p>
          </Card>
          <Card className={styles.featureCard}>
            <div className={styles.featureHeader}>
              <Columns3 size={18} color="#5771FE" aria-hidden="true" />
              <h3>Interactive 3D viewer</h3>
            </div>
            <p>Compare glycoforms side-by-side with live Mol* structures and affinity labels.</p>
          </Card>
        </section>

        <section className={styles.announcement}>
          <span className={styles.announcementTag}>New</span>
          <p>
            Phase 3 Explorer now flags missing glycan coordinates and highlights identical templates.
          </p>
        </section>
      </main>

      <footer className={styles.footer}>
        <div className={styles.footerLinks}>
          <a href="https://github.com/VUzan-bio/GlycoML" target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a href="https://github.com/VUzan-bio/GlycoML" target="_blank" rel="noreferrer">
            Documentation
          </a>
          <a href="https://github.com/VUzan-bio/GlycoML" target="_blank" rel="noreferrer">
            License
          </a>
        </div>
        <span className={styles.footerMeta}>GlycoML Research Platform</span>
      </footer>
    </div>
  );
}
