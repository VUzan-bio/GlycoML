import React from 'react';
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import SingleView from './pages/SingleView';
import CompareView from './pages/CompareView';

export default function App() {
  return (
    <BrowserRouter>
      <nav className="main-nav">
        <Link to="/" className="brand">
          GlycoML Phase 3
        </Link>
        <div className="nav-links">
          <Link to="/single">Single view</Link>
          <Link to="/compare">Compare</Link>
        </div>
      </nav>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/single" element={<SingleView />} />
        <Route path="/compare" element={<CompareView />} />
      </Routes>
    </BrowserRouter>
  );
}
