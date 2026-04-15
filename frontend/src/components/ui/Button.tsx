import React from 'react';
import styles from './Button.module.css';

type Variant = 'primary' | 'secondary' | 'icon';

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  icon?: React.ReactNode;
};

export default function Button({
  variant = 'primary',
  icon,
  className,
  children,
  type,
  ...rest
}: Props) {
  const variantClass = variant === 'icon' ? styles.iconVariant : styles[variant];
  const classes = [styles.button, variantClass, icon ? styles.withIcon : '', className]
    .filter(Boolean)
    .join(' ');

  return (
    <button className={classes} type={type || 'button'} {...rest}>
      {icon ? <span className={styles.iconGraphic}>{icon}</span> : null}
      {children}
    </button>
  );
}
