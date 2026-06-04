import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { MdDownload, MdZoomIn, MdClose, MdRefresh } from 'react-icons/md';

const API = 'http://127.0.0.1:8000';

const CATEGORY_LABELS = {
  audit:      { label: 'Leakage Audit',   color: '#2563eb', bg: '#eff6ff' },
  mitigation: { label: 'Mitigation',      color: '#7c3aed', bg: '#f5f3ff' },
  fairness:   { label: 'Fairness',        color: '#059669', bg: '#f0fdf4' },
  general:    { label: 'General',         color: '#64748b', bg: '#f8fafc' },
};

function PlotCard({ plot, onExpand }) {
  const [loaded, setLoaded] = useState(false);
  const [error,  setError]  = useState(false);
  const imgUrl = plot.url.startsWith('http') ? plot.url : `${API}${plot.url}`;
  const cat    = CATEGORY_LABELS[plot.category] || CATEGORY_LABELS.general;

  const handleDownload = (e) => {
    e.stopPropagation();
    const a = document.createElement('a');
    a.href = imgUrl;
    a.download = plot.filename || `${plot.name}.png`;
    a.click();
  };

  return (
    <motion.div
      className="plot-card"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2, boxShadow: '0 8px 32px rgba(0,0,0,0.10)' }}
      onClick={() => !error && onExpand(imgUrl, plot.label)}
      style={{ cursor: error ? 'default' : 'zoom-in', position: 'relative' }}
    >
      {/* Category chip */}
      <div style={{
        position: 'absolute', top: 10, left: 10, zIndex: 2,
        background: cat.bg, color: cat.color,
        border: `1px solid ${cat.color}30`,
        borderRadius: 99, padding: '2px 8px', fontSize: 10.5, fontWeight: 700,
        letterSpacing: '0.04em', textTransform: 'uppercase',
      }}>
        {cat.label}
      </div>

      {/* Expand + Download buttons */}
      <div style={{
        position: 'absolute', top: 8, right: 8, zIndex: 2,
        display: 'flex', gap: 4, opacity: 0,
      }} className="plot-card-actions">
        <button
          title="Download PNG"
          onClick={handleDownload}
          style={{
            background: 'rgba(15,23,42,0.75)', border: 'none', borderRadius: 6,
            padding: '5px 7px', cursor: 'pointer', color: '#e2e8f0', fontSize: 14,
          }}
        >
          <MdDownload />
        </button>
        <button
          title="Expand"
          onClick={(e) => { e.stopPropagation(); onExpand(imgUrl, plot.label); }}
          style={{
            background: 'rgba(15,23,42,0.75)', border: 'none', borderRadius: 6,
            padding: '5px 7px', cursor: 'pointer', color: '#e2e8f0', fontSize: 14,
          }}
        >
          <MdZoomIn />
        </button>
      </div>

      {/* Image */}
      {!loaded && !error && (
        <div style={{
          height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: '#f1f5f9', borderRadius: '8px 8px 0 0',
        }}>
          <div style={{ fontSize: 28 }}>⏳</div>
        </div>
      )}
      {error && (
        <div style={{
          height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: '#fef2f2', borderRadius: '8px 8px 0 0', flexDirection: 'column', gap: 8,
        }}>
          <div style={{ fontSize: 28 }}>⚠️</div>
          <div style={{ fontSize: 12, color: '#dc2626' }}>Image not available</div>
        </div>
      )}
      <img
        src={imgUrl}
        alt={plot.label}
        loading="lazy"
        onLoad={() => setLoaded(true)}
        onError={() => { setLoaded(true); setError(true); }}
        style={{ display: loaded && !error ? 'block' : 'none', width: '100%', borderRadius: '8px 8px 0 0' }}
      />

      <div className="plot-card-label" style={{ fontWeight: 600, fontSize: 12.5 }}>
        📊 {plot.label}
      </div>
    </motion.div>
  );
}

export default function VisualizationsPage() {
  const [plots,    setPlots]    = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [lightbox, setLightbox] = useState(null); // {url, label}
  const [filter,   setFilter]   = useState('all');

  const fetchPlots = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API}/visualizations`);
      // Backend now returns array: [{name, filename, url, label, category}]
      setPlots(Array.isArray(data.plots) ? data.plots : []);
    } catch {
      setPlots([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchPlots(); }, [fetchPlots]);

  // Escape key closes lightbox
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') setLightbox(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const categories = ['all', 'audit', 'mitigation', 'fairness'];
  const visible = filter === 'all' ? plots : plots.filter(p => p.category === filter);

  return (
    <div className="fade-in">
      {/* ── Topbar ─────────────────────────────── */}
      <div className="topbar">
        <div className="topbar-breadcrumb">
          ObscuraAI › <strong>Research Visualizations</strong>
        </div>
        <div className="topbar-right">
          <span className="topbar-badge">{plots.length} Plots</span>
          <span className="topbar-badge">IEEE Publication Quality</span>
          <button
            className="btn btn-secondary btn-sm"
            onClick={fetchPlots}
            title="Refresh plot list"
          >
            <MdRefresh /> Refresh
          </button>
        </div>
      </div>

      <div className="page-content">
        <div className="page-header" style={{ marginTop: 8 }}>
          <div className="module-badge" style={{ background: '#fdf4ff', color: '#7e22ce', border: '1px solid #e9d5ff' }}>
            VISUALIZATIONS
          </div>
          <h1 className="page-title">Research Visualizations Gallery</h1>
          <p className="page-subtitle">
            7 IEEE-publication-quality figures generated by the ML pipeline. Click any plot to enlarge or download.
          </p>
        </div>

        {/* ── Category Filter ──────────────────── */}
        {plots.length > 0 && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
            {categories.map(cat => {
              const meta = CATEGORY_LABELS[cat] || { color: '#64748b', bg: '#f8fafc' };
              const isActive = filter === cat;
              return (
                <button
                  key={cat}
                  onClick={() => setFilter(cat)}
                  style={{
                    padding: '6px 16px', borderRadius: 99, border: `1.5px solid ${isActive ? meta.color : '#e2e8f0'}`,
                    background: isActive ? meta.bg : '#fff',
                    color: isActive ? meta.color : '#64748b',
                    fontWeight: isActive ? 700 : 500, fontSize: 12.5, cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  {cat === 'all' ? `All (${plots.length})` : `${cat.charAt(0).toUpperCase() + cat.slice(1)} (${plots.filter(p => p.category === cat).length})`}
                </button>
              );
            })}
          </div>
        )}

        {/* ── Loading ──────────────────────────── */}
        {loading && (
          <div className="card">
            <div className="empty-state">
              <div className="empty-state-icon">🔄</div>
              <h3>Loading plots…</h3>
            </div>
          </div>
        )}

        {/* ── Empty ────────────────────────────── */}
        {!loading && plots.length === 0 && (
          <div className="card">
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <h3>No Plots Yet</h3>
              <p>Run the pipeline from the Overview Dashboard to generate all 7 visualizations.</p>
            </div>
          </div>
        )}

        {/* ── Plot Grid ────────────────────────── */}
        {visible.length > 0 && (
          <div className="plot-grid" style={{ '--plot-cols': visible.length >= 3 ? '3' : String(visible.length) }}>
            {visible.map((plot, i) => (
              <PlotCard
                key={plot.name}
                plot={plot}
                onExpand={(url, label) => setLightbox({ url, label })}
              />
            ))}
          </div>
        )}

        {/* ── Caption Guide ────────────────────── */}
        {plots.length > 0 && (
          <motion.div
            className="card card-padded"
            style={{ marginTop: 20, background: '#fdf4ff', borderColor: '#e9d5ff' }}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
          >
            <div className="section-label" style={{ color: '#7e22ce' }}>Plot Reference Guide</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 28px', fontSize: 12.5, color: '#475569', lineHeight: 1.85 }}>
              <div>📈 <strong>ROC Curves</strong> — LR + MLP attacker per attribute, baseline vs. post-mitigation</div>
              <div>📊 <strong>Leakage Bars</strong> — AUC comparison across attributes with risk thresholds</div>
              <div>🕸️ <strong>Radar Chart</strong> — Multi-attribute leakage profile (AUC per attacker × attribute)</div>
              <div>📉 <strong>Trade-off Curves</strong> — Privacy–utility Pareto frontier (λ-sweep + σ-sweep)</div>
              <div>⚖️ <strong>Before/After</strong> — Mitigation effectiveness: baseline → adversarial → noise</div>
              <div>📓 <strong>Training History</strong> — Adversarial disentangler loss convergence (60 epochs)</div>
              <div>🧮 <strong>Fairness Disparity</strong> — Max Demographic Disparity by ethnicity group</div>
            </div>
          </motion.div>
        )}
      </div>

      {/* ── Lightbox ─────────────────────────── */}
      <AnimatePresence>
        {lightbox && (
          <motion.div
            className="lightbox"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setLightbox(null)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              onClick={e => e.stopPropagation()}
              style={{ position: 'relative', maxWidth: '90vw', maxHeight: '90vh' }}
            >
              <button
                onClick={() => setLightbox(null)}
                style={{
                  position: 'absolute', top: -16, right: -16, zIndex: 10,
                  background: '#0f172a', border: 'none', borderRadius: '50%',
                  width: 32, height: 32, cursor: 'pointer', color: '#e2e8f0',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16,
                }}
              >
                <MdClose />
              </button>
              <img
                src={lightbox.url}
                alt={lightbox.label}
                style={{ maxWidth: '90vw', maxHeight: '90vh', borderRadius: 10, boxShadow: '0 24px 60px rgba(0,0,0,0.5)' }}
              />
              {lightbox.label && (
                <div style={{
                  position: 'absolute', bottom: 8, left: '50%', transform: 'translateX(-50%)',
                  background: 'rgba(15,23,42,0.8)', color: '#e2e8f0', padding: '6px 16px',
                  borderRadius: 99, fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap',
                }}>
                  {lightbox.label}
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
