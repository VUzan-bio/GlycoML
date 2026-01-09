import React from 'react';
import styles from './StatCard.module.css';

type Tone = 'strong' | 'moderate' | 'weak' | 'neutral';

type Props = {
  label: string;
  value: string;
  subtitle?: string;
  tone?: Tone;
  icon?: React.ReactNode;
  badge?: string;
};

export default function StatCard({ label, value, subtitle, tone = 'neutral', icon, badge }: Props) {
  return (
    <div className={`${styles.card} ${styles[tone]}`}>
      <div className={styles.header}>
        <div className={styles.labelRow}>
          {icon ? <span className={styles.icon}>{icon}</span> : null}
          <span>{label}</span>
        </div>
        {badge ? <span className={styles.badge}>{badge}</span> : null}
      </div>
      <div className={styles.value}>{value}</div>
      {subtitle ? <div className={styles.subtitle}>{subtitle}</div> : null}
    </div>
  );
}
