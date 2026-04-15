import React from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import HomePage from './pages/HomePage';
import SingleView from './pages/SingleView';
import CompareView from './pages/CompareView';
import ExplorerView from './pages/ExplorerView';
import AboutPage from './pages/AboutPage';

export default function App() {
  return (
    <BrowserRouter>
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <Header />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/predict" element={<SingleView />} />
        <Route path="/single" element={<SingleView />} />
        <Route path="/compare" element={<CompareView />} />
        <Route path="/explorer" element={<ExplorerView />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </BrowserRouter>
  );
}
