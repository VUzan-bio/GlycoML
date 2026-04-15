import React from 'react';
import { ChevronDown } from 'lucide-react';
import styles from './Select.module.css';

type Option = {
  label: string;
  value: string;
};

type Props = {
  id: string;
  label: string;
  value: string;
  options: Option[];
  onChange: (value: string) => void;
  ariaLabel?: string;
  disabled?: boolean;
  hideLabel?: boolean;
};

export default function Select({
  id,
  label,
  value,
  options,
  onChange,
  ariaLabel,
  disabled,
  hideLabel,
}: Props) {
  return (
    <label className={styles.field} htmlFor={id}>
      <span className={`${styles.label} ${hideLabel ? 'sr-only' : ''}`}>{label}</span>
      <div className={styles.control}>
        <select
          id={id}
          className={styles.select}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          aria-label={ariaLabel || label}
          disabled={disabled}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDown className={styles.icon} aria-hidden="true" />
      </div>
    </label>
  );
}
