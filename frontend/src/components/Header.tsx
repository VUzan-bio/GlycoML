import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, Info, Menu } from 'lucide-react';
import { NavLink, useLocation } from 'react-router-dom';
import styles from './Header.module.css';

export default function Header() {
  const location = useLocation();
  const [isPredictOpen, setIsPredictOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);
  const isPredictActive = ['/predict', '/compare', '/explorer'].some((path) =>
    location.pathname.startsWith(path)
  );

  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      if (!dropdownRef.current) {
        return;
      }
      if (!dropdownRef.current.contains(event.target as Node)) {
        setIsPredictOpen(false);
      }
    };
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsPredictOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleEscape);
    };
  }, []);

  useEffect(() => {
    if (!isMenuOpen) {
      setIsPredictOpen(false);
    }
  }, [isMenuOpen]);

  const navLinkClass = ({ isActive }: { isActive: boolean }) =>
    `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`;

  return (
    <header className={styles.header}>
      <div className={styles.brand}>GlycoML</div>

      <nav
        className={`${styles.nav} ${isMenuOpen ? styles.navOpen : ''}`}
        aria-label="Primary"
        id="primary-nav"
      >
        <NavLink to="/" className={navLinkClass} onClick={() => setIsMenuOpen(false)}>
          Home
        </NavLink>

        <div className={styles.dropdown} ref={dropdownRef}>
          <button
            className={`${styles.dropdownButton} ${isPredictActive ? styles.dropdownButtonActive : ''}`}
            type="button"
            aria-haspopup="menu"
            aria-expanded={isPredictOpen}
            onClick={() => setIsPredictOpen((open) => !open)}
          >
            Predict
            <ChevronDown size={16} aria-hidden="true" />
          </button>
          {isPredictOpen && (
            <div className={styles.dropdownMenu} role="menu">
              <NavLink
                to="/predict"
                className={styles.dropdownItem}
                role="menuitem"
                onClick={() => {
                  setIsPredictOpen(false);
                  setIsMenuOpen(false);
                }}
              >
                Binding Prediction
              </NavLink>
              <NavLink
                to="/explorer"
                className={styles.dropdownItem}
                role="menuitem"
                onClick={() => {
                  setIsPredictOpen(false);
                  setIsMenuOpen(false);
                }}
              >
                FcγR Explorer
              </NavLink>
              <NavLink
                to="/compare"
                className={styles.dropdownItem}
                role="menuitem"
                onClick={() => {
                  setIsPredictOpen(false);
                  setIsMenuOpen(false);
                }}
              >
                Compare View
              </NavLink>
            </div>
          )}
        </div>

        <a
          href="https://github.com/VUzan-bio/GlycoML"
          target="_blank"
          rel="noreferrer"
          className={styles.navLink}
          onClick={() => setIsMenuOpen(false)}
        >
          Documentation
        </a>

        <NavLink to="/about" className={navLinkClass} onClick={() => setIsMenuOpen(false)}>
          About
        </NavLink>
      </nav>

      <div className={styles.actions}>
        <button
          className={styles.menuButton}
          type="button"
          aria-label="Toggle navigation"
          aria-expanded={isMenuOpen}
          aria-controls="primary-nav"
          onClick={() => setIsMenuOpen((open) => !open)}
        >
          <Menu size={18} aria-hidden="true" />
        </button>
        <button className={styles.iconBtn} type="button" title="Help" aria-label="Help">
          <Info size={18} aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
