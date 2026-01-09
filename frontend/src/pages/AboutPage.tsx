import React from 'react';
import { BookOpen, FlaskConical, ShieldCheck } from 'lucide-react';
import Card from '../components/ui/Card';
import styles from './AboutPage.module.css';

export default function AboutPage() {
  return (
    <div className="app-shell">
      <main className={styles.layout} role="main" id="main-content">
        <section className={styles.header}>
          <h2>About GlycoML</h2>
          <p className={styles.subhead}>
            GlycoML is a scientific interface for Fc-FcγR binding prediction, combining
            glycan-aware machine learning with structural visualization.
          </p>
        </section>

        <div className={styles.cardGrid}>
          <Card className={styles.card}>
            <div className={styles.cardHeader}>
              <FlaskConical size={18} color="#5771FE" aria-hidden="true" />
              <span>Scientific focus</span>
            </div>
            <p>
              Quantifies binding affinity shifts across FcγR allotypes and glycoform variants
              with clear, standardized metrics.
            </p>
          </Card>
          <Card className={styles.card}>
            <div className={styles.cardHeader}>
              <ShieldCheck size={18} color="#5771FE" aria-hidden="true" />
              <span>Reliable outputs</span>
            </div>
            <p>
              Uses consistent thresholds for affinity classes and provides provenance for
              predicted structures.
            </p>
          </Card>
          <Card className={styles.card}>
            <div className={styles.cardHeader}>
              <BookOpen size={18} color="#5771FE" aria-hidden="true" />
              <span>Open documentation</span>
            </div>
            <p>
              See the full methods, datasets, and release notes in the public documentation.
            </p>
            <a
              className={styles.link}
              href="https://github.com/VUzan-bio/GlycoML"
              target="_blank"
              rel="noreferrer"
            >
              View documentation
            </a>
          </Card>
        </div>
      </main>
    </div>
  );
}
