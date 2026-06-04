import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import Badge from '../components/Badge';
import {
  MdPlayArrow, MdRefresh, MdArrowForward,
  MdAnalytics, MdSecurity, MdAssessment, MdSearch, MdTune, MdFlash
} from 'react-icons/md';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line
} from 'recharts';

const API = 'http://127.0.0.1:8000';
const ATTR_COLORS  = { gender: '#2563eb', age_group: '#7c3aed', ethnicity: '#059669' };
const RISK_PALETTE = { CRITICAL: '#dc2626', HIGH: '#f59e0b', MEDIUM: '#3b82f6', LOW: '#22c55e', NEGLIGIBLE: '#64748b' };
const LEVEL_COLOR  = { INFO: '#2563eb', ERROR: '#dc2626', WARNING: '#f59e0b' };

function PipelineBanner({ status, onRun, loading }) {
  const cls = status?.status === 'running'   ? 'running'
            : status?.status === 'completed' ? 'completed'
            : status?.status === 'error'     ? 'error' : '';
  return (
    <div className={`pipeline-banner ${cls}`} style={{ marginBottom: 20 }}>
      <span className={`dot dot-${status?.status || 'idle'}`} />
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 700, fontSize: 13, color: '#0f172a' }}>
          Pipeline Status: <span style={{ textTransform: 'capitalize' }}>
            {status?.status || 'idle'}
          </span>
          {status?.status === 'running' && status?.progress > 0 && ` — ${status.progress.toFixed(0)}%`}
          {status?.run_id && (
            <span style={{ fontFamily: 'monospace', fontSize: 10, marginLeft: 10, color: '#64748b' }}>
              Run #{status.run_id}
            </span>
          )}
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 1 }}>
          {status?.message || 'Ready to execute'}
        </div>
        {status?.status === 'running' && (
          <div className="progress-wrap" style={{ maxWidth: 360, marginTop: 6 }}>
            <div className="progress-fill" style={{ width: `${status.progress || 0}%` }} />
          </div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          className="btn btn-primary btn-sm"
          onClick={() => onRun(true)}
          disabled={loading || status?.status === 'running'}
          id="run-pipeline-btn"
        >
          {status?.status === 'running'
            ? <><MdRefresh /> Running…</>
            : <><MdPlayArrow /> Deep Run Pipeline</>}
        </button>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => onRun(false)}
          disabled={loading || status?.status === 'running'}
          id="fast-pipeline-btn"
          title="Skips λ/σ sweep — runs in ~45 s. Ideal for demos."
        >
          <MdFlash /> Fast Demo
        </button>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [overview,  setOverview]  = useState(null);
  const [pStatus,   setPStatus]   = useState(null);
  const [auditSum,  setAuditSum]  = useState(null);
  const [logEntries, setLogEntries] = useState([]);
  const [loading,   setLoading]   = useState(false);
  const navigate = useNavigate();

  const fetchAll = useCallback(async () => {
    const [ov, ps, au, lg] = await Promise.allSettled([
      axios.get(`${API}/metrics/overview`),
      axios.get(`${API}/pipeline/status`),
      axios.get(`${API}/audit/summary`),
      axios.get(`${API}/logs?last_n=8`),
    ]);
    if (ov.status === 'fulfilled') setOverview(ov.value.data);
    if (ps.status === 'fulfilled') setPStatus(ps.value.data);
    if (au.status === 'fulfilled') setAuditSum(au.value.data);
    if (lg.status === 'fulfilled') setLogEntries((lg.value.data?.entries || []).reverse());
  }, []);

  useEffect(() => {
    fetchAll();
    const iv = setInterval(fetchAll, 4000);
    return () => clearInterval(iv);
  }, [fetchAll]);

  const handleRun = async (deepSweep = true) => {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/pipeline/run?dataset=synthetic&sweep=${deepSweep}`);
      toast.success(
        deepSweep
          ? `Deep pipeline started (Run #${data.run_id || '?'}). ~3–5 min.`
          : `Fast pipeline started (Run #${data.run_id || '?'}). ~45 s.`
      );
    } catch { toast.error('Could not start pipeline. Is the backend running?'); }
    setLoading(false);
  };

  const ov       = overview?.data || overview || {};
  const attrs    = auditSum?.attributes || {};
  const hasData  = ov.overall_risk && ov.overall_risk !== 'UNKNOWN';

  const barData = ['gender', 'age_group', 'ethnicity'].map(a => ({
    name: a.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()),
    auc:  +(attrs[a]?.best_auc || 0).toFixed(3),
    attr: a,
  }));

  return (
    <div className="fade-in">
      {/* ── Topbar ──────────────────────────────── */}
      <div className="topbar">
        <div className="topbar-breadcrumb">
          ObscuraAI <span style={{ color: '#cbd5e1' }}>›</span> <strong>Overview Dashboard</strong>
        </div>
        <div className="topbar-right">
          <span className="topbar-badge">12k Synthetic Embeddings</span>
          <span className="topbar-badge">3 Attributes Monitored</span>
          <span className="topbar-badge">512-dim Embeddings</span>
        </div>
      </div>

      <div className="page-content">
        {/* ── Header ─────────────────────────────── */}
        <div className="page-header" style={{ marginTop: 8 }}>
          <div className="module-badge">Adversarial Demographic Leakage Auditing &amp; Mitigation</div>
          <h1 className="page-title">ObscuraAI — Unified Privacy Dashboard</h1>
          <p className="page-subtitle">
            End-to-end pipeline: Synthetic Embeddings → LR+MLP Audit → GRL Mitigation → Fairness → IEEE Model Card
          </p>
        </div>

        <PipelineBanner status={pStatus} onRun={handleRun} loading={loading} />

        {/* ── Pipeline Architecture Flow ──────────── */}
        <div className="card card-padded" style={{ marginBottom: 20 }}>
          <div className="section-label">Pipeline Architecture</div>
          <div className="pipeline-flow" style={{ flexWrap: 'wrap', gap: 6 }}>
            {[
              ['📂', 'Dataset Gen', 'Synthetic CelebA-like\nN=12k, D=512'],
              ['🔍', 'LR+MLP Audit', 'AUC leakage\nattacker models'],
              ['🎭', 'GRL Disentangle', 'λ-sweep\nadversarial'],
              ['💨', 'Noise Injection', 'σ-sweep\nGaussian'],
              ['⚖️', 'Fairness', 'MDD + EOD\nper-group'],
              ['📋', 'Model Card', 'IEEE report\nSHA-256 signed'],
            ].map(([icon, step, sub], i, arr) => (
              <React.Fragment key={step}>
                <div className="flow-step" title={sub}>{icon} {step}</div>
                {i < arr.length - 1 && <span className="flow-arrow">→</span>}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* ── KPI Cards ──────────────────────────── */}
        <div className="metrics-grid">
          {[
            {
              icon: '🎯', label: 'Baseline Leakage', cls: 'red',
              value: hasData ? `${((ov.mean_leakage_baseline || 0)*100).toFixed(1)}%` : '—',
              sub: 'Pre-mitigation mean', bg: '#fff5f5', iconBg: '#fee2e2',
            },
            {
              icon: '📉', label: 'Avg AUC Reduction', cls: 'green',
              value: hasData ? `${(ov.avg_adv_reduction_pct || 0).toFixed(1)}%` : '—',
              sub: 'Adversarial method', bg: '#f0fdf4', iconBg: '#dcfce7',
            },
            {
              icon: '⚖️', label: 'Fairness MDD', cls: 'amber',
              value: hasData ? (ov.mdd || 0).toFixed(3) : '—',
              sub: 'Max demographic disparity', bg: '#fffbeb', iconBg: '#fef3c7',
            },
            {
              icon: '🔒', label: 'Overall Risk',
              value: hasData ? ov.overall_risk : '—',
              cls: hasData ? (ov.overall_risk === 'CRITICAL' || ov.overall_risk === 'HIGH' ? 'red' : 'green') : '',
              sub: 'Worst-attribute AUC', bg: '#fff5f5', iconBg: '#fee2e2',
            },
            {
              icon: '🎭', label: 'Identity Utility',
              value: hasData && ov.mean_genuine_similarity ? (ov.mean_genuine_similarity * 100).toFixed(1) + '%' : '—',
              cls: 'blue', sub: 'Genuine cosine similarity', bg: '#eff6ff', iconBg: '#dbeafe',
            },
            {
              icon: '📊', label: 'Attributes Audited',
              value: hasData ? (ov.n_attributes || 3) : '3',
              cls: 'blue', sub: 'Gender · Age · Ethnicity', bg: '#eff6ff', iconBg: '#dbeafe',
            },
          ].map(({ icon, label, value, sub, cls, bg, iconBg }, i) => (
            <motion.div key={label} className="metric-card" style={{ background: bg }}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
              <div className="metric-icon-wrap" style={{ background: iconBg }}>{icon}</div>
              <div className={`metric-value ${cls}`}>{value}</div>
              <div className="metric-label">{label}</div>
              <div className="metric-sub">{sub}</div>
            </motion.div>
          ))}
        </div>

        {/* ── Module Summary ──────────────────────── */}
        <div className="section-label">Module Summary</div>
        <div className="grid-3" style={{ marginBottom: 20 }}>
          {[
            {
              num: 'MODULE 01', name: 'Leakage Auditing', icon: <MdSearch size={16} />,
              desc: 'LR + MLP attackers on 512-dim face embeddings. Mutual information estimation. AUC-ROC leakage scoring per attribute.',
              stat: hasData ? `Best AUC: ${Math.max(...['gender','age_group','ethnicity'].map(a => attrs[a]?.best_auc || 0)).toFixed(3)}` : 'Not run',
              status: hasData ? ov.overall_risk : null,
              path: '/audit', color: '#2563eb',
            },
            {
              num: 'MODULE 02', name: 'Mitigation Engine', icon: <MdTune size={16} />,
              desc: 'Gradient Reversal Layer (GRL) adversarial disentanglement + Gaussian noise injection. λ-sweep and σ-sweep trade-off curves.',
              stat: hasData ? `Adv. ↓${(ov.avg_adv_reduction_pct || 0).toFixed(1)}% leakage` : 'Not run',
              status: null, path: '/mitigation', color: '#7c3aed',
            },
            {
              num: 'MODULE 03', name: 'Reporting & Model Cards', icon: <MdAssessment size={16} />,
              desc: 'IEEE-style model cards with SHA-256 signed audit hash. Fairness MDD + Equalised Odds. Cryptographic accountability chain.',
              stat: hasData ? 'Report + SHA-256 generated' : 'Not run',
              status: null, path: '/report', color: '#059669',
            },
          ].map((m, i) => (
            <motion.div key={m.num} className="module-card" initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 + i * 0.08 }}>
              <div className="module-number" style={{ color: m.color }}>{m.num}</div>
              <div className="module-name" style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <span style={{ color: m.color }}>{m.icon}</span> {m.name}
              </div>
              <div className="module-desc">{m.desc}</div>
              <div className="progress-wrap" style={{ marginBottom: 10 }}>
                <div className="progress-fill" style={{
                  width: hasData ? '100%' : '0%',
                  background: `linear-gradient(90deg, ${m.color}, ${m.color}88)`,
                }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, color: '#64748b' }}>{m.stat}</span>
                <button className="btn btn-xs btn-blue-ghost" onClick={() => navigate(m.path)}>
                  View <MdArrowForward />
                </button>
              </div>
            </motion.div>
          ))}
        </div>

        {/* ── Audit Chart + Per-Attr Leakage ─────── */}
        {hasData && (
          <div className="grid-2" style={{ marginBottom: 20 }}>
            <div className="card">
              <div className="card-header">
                <div className="card-title"><MdAnalytics /> Attacker AUC per Attribute</div>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>Baseline inference performance</span>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={barData} margin={{ top: 5, right: 16, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                    <YAxis domain={[0.4, 1.0]} tick={{ fontSize: 11, fill: '#64748b' }} />
                    <Tooltip
                      contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 12 }}
                      formatter={v => [v.toFixed(3), 'AUC']}
                    />
                    <Bar dataKey="auc" radius={[4,4,0,0]} maxBarSize={52}>
                      {barData.map(d => <Cell key={d.attr} fill={ATTR_COLORS[d.attr] || '#2563eb'} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-title"><MdSecurity /> Per-Attribute Leakage</div>
              </div>
              <div className="card-body">
                {['gender', 'age_group', 'ethnicity'].map(attr => {
                  const d     = attrs[attr] || {};
                  const auc   = d.best_auc || 0;
                  const color = ATTR_COLORS[attr];
                  return (
                    <div key={attr} className="attr-row">
                      <div className="attr-label">{attr.replace('_', ' ')}</div>
                      <div className="attr-bar-wrap">
                        <div className="auc-bar">
                          <div className="auc-bar-fill" style={{ width: `${auc * 100}%`, background: color }} />
                        </div>
                      </div>
                      <div className="attr-value" style={{ color }}>{auc.toFixed(3)}</div>
                      <Badge level={d.risk_level} />
                    </div>
                  );
                })}
                <div style={{ marginTop: 14, padding: '10px 14px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
                  <strong>Threshold Guide:</strong> AUC &gt;0.90 = CRITICAL · &gt;0.75 = HIGH · &gt;0.60 = MEDIUM · ≈0.50 = NEGLIGIBLE
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Live Audit Log ──────────────────────── */}
        <div className="card" style={{ marginBottom: 20, overflow: 'hidden' }}>
          <div style={{ background: '#0f172a', padding: '12px 18px' }}>
            <div style={{ color: '#f8fafc', fontWeight: 700, fontSize: 12.5, display: 'flex', alignItems: 'center', gap: 8 }}>
              🔒 Live Audit Log
              <span style={{ background: '#1e293b', color: '#64748b', border: '1px solid #334155', borderRadius: 6, padding: '2px 8px', fontSize: 10 }}>
                Last 8 entries
              </span>
            </div>
          </div>
          <div style={{ padding: '0 18px', maxHeight: 230, overflowY: 'auto' }}>
            {logEntries.length === 0 ? (
              <div style={{ padding: '16px 0', color: '#94a3b8', fontSize: 12.5, textAlign: 'center' }}>
                No audit entries yet — run the pipeline to see live log.
              </div>
            ) : (
              logEntries.map((entry, i) => (
                <div key={entry.seq ?? i} className="log-entry">
                  <div className="log-time" style={{ minWidth: 170 }}>
                    {new Date(entry.timestamp).toLocaleTimeString()} · #{entry.seq}
                  </div>
                  <div className="log-msg" style={{ flex: 1 }}>
                    <span style={{ fontSize: 10, fontWeight: 700, marginRight: 6, color: LEVEL_COLOR[entry.level] || '#64748b' }}>
                      [{entry.module}]
                    </span>
                    {entry.message}
                  </div>
                  <span style={{ fontSize: 9, fontFamily: 'monospace', color: '#475569', minWidth: 80 }}>
                    {entry.entry_hash?.slice(0, 8)}…
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {!hasData && (
          <div className="card">
            <div className="empty-state">
              <div className="empty-state-icon">🚀</div>
              <h3>No Results Yet</h3>
              <p>
                Click <strong>Deep Run Pipeline</strong> for full results (~3–5 min), or{' '}
                <strong>Fast Demo</strong> for a quick preview (~45 s).
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
