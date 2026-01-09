import React from 'react';
import { Atom, Eye, Info, Table2 } from 'lucide-react';
import { NavLink } from 'react-router-dom';

export default function Header() {
  return (
    <header className="header">
      <div className="header-left">
        <Atom size={28} color="#4286F5" aria-hidden="true" />
        <div>
          <h1>Fcγ Binding Explorer</h1>
          <p>Fc-FcγR Interaction Predictor</p>
        </div>
      </div>

      <nav className="header-nav" aria-label="Primary">
        <NavLink to="/single" className={({ isActive }) => `nav-btn${isActive ? ' active' : ''}`}>
          <Eye size={20} aria-hidden="true" />
          Viewer
        </NavLink>
        <NavLink to="/compare" className={({ isActive }) => `nav-btn${isActive ? ' active' : ''}`}>
          <Table2 size={20} aria-hidden="true" />
          Compare
        </NavLink>
      </nav>

      <div className="header-right">
        <button className="icon-btn" type="button" title="Help" aria-label="Help">
          <Info size={20} aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
