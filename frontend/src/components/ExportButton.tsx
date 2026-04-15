import React from 'react';
import { Download } from 'lucide-react';
import { exportUrl } from '../api';
import buttonStyles from './ui/Button.module.css';

type Props = {
  disabled?: boolean;
  className?: string;
};

export default function ExportButton({ disabled, className }: Props) {
  const classes = [
    buttonStyles.button,
    buttonStyles.secondary,
    buttonStyles.withIcon,
    disabled ? buttonStyles.disabled : '',
    className || '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <a
      className={classes}
      href={disabled ? undefined : exportUrl()}
      aria-disabled={disabled}
      aria-label="Export structure data"
      tabIndex={disabled ? -1 : 0}
    >
      <span className={buttonStyles.iconGraphic} aria-hidden="true">
        <Download size={20} />
      </span>
      Export Structure
    </a>
  );
}
