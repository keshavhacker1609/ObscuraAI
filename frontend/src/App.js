import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import './index.css';

import Sidebar           from './components/Sidebar';
import Dashboard         from './pages/Dashboard';
import AuditPage         from './pages/AuditPage';
import MitigationPage    from './pages/MitigationPage';
import ReportPage        from './pages/ReportPage';
import VisualizationsPage from './pages/VisualizationsPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/"               element={<Dashboard />} />
            <Route path="/audit"          element={<AuditPage />} />
            <Route path="/mitigation"     element={<MitigationPage />} />
            <Route path="/report"         element={<ReportPage />} />
            <Route path="/visualizations" element={<VisualizationsPage />} />
          </Routes>
        </main>
      </div>
      <Toaster
        position="top-right"
        toastOptions={{
          style: { fontSize: 13, borderRadius: 10, border: '1px solid #e2e8f0', background: '#fff', color: '#0f172a' },
          success: { iconTheme: { primary: '#16a34a', secondary: '#fff' } },
          error:   { iconTheme: { primary: '#dc2626', secondary: '#fff' } },
        }}
      />
    </BrowserRouter>
  );
}
