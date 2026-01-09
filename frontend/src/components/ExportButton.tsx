import React from 'react';
import { exportUrl } from '../api';

type Props = {
  disabled?: boolean;
};

export default function ExportButton({ disabled }: Props) {
  return (
    <a
      className={`export ${disabled ? 'disabled' : ''}`}
      href={disabled ? undefined : exportUrl()}
      aria-disabled={disabled}
    >
      Export CSV
    </a>
  );
}
