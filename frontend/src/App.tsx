import React from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import SingleView from './pages/SingleView';
import CompareView from './pages/CompareView';

export default function App() {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<SingleView />} />
        <Route path="/single" element={<SingleView />} />
        <Route path="/compare" element={<CompareView />} />
      </Routes>
    </BrowserRouter>
  );
}
