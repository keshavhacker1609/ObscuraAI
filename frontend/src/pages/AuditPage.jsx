import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import Badge from '../components/Badge';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine, RadarChart,
  Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

const API = 'http://127.0.0.1:8000';
const ATTR_COLORS = { gender: '#2563eb', age_group: '#7c3aed', ethnicity: '#059669' };
const ATTR_LABELS = { gender: 'Gender', age_group: 'Age Group', ethnicity: 'Ethnicity' };

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tt">
      <div className="custom-tt-label">{payload[0]?.payload?.name || payload[0]?.name}</div>
      <div className="custom-tt-value">{payload[0]?.value?.toFixed(4)}</div>
    </div>
  );
};

export default function AuditPage() {
  const [audit,   setAudit]   = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      axios.get(`${API}/audit`),
      axios.get(`${API}/audit/summary`),
    ]).then(([a, s]) => {
      if (a.status === 'fulfilled' && a.value.data.data) setAudit(a.value.data.data);
      if (s.status === 'fulfilled') setSummary(s.value.data);
      setLoading(false);
    });
  }, []);

  const attrs = ['gender', 'age_group', 'ethnicity'];

  if (loading) return <div className="page-content" style={{padding: 40, color: '#64748b'}}>Loading audit data…</div>;

  if (!audit) return (
    <div className="fade-in">
      <div className="topbar">
        <div className="topbar-breadcrumb">Privacy Framework › <strong>Module 1: Leakage Auditing</strong></div>
        <div className="topbar-right"><span className="topbar-badge">MODULE 01</span></div>
      </div>
      <div className="page-content"><div className="card"><div className="empty-state">
        <div className="empty-state-icon">🔍</div>
        <h3>No Audit Results</h3>
        <p>Return to the dashboard and run the pipeline first.</p>
      </div></div></div>
    </div>
  );

  // Bar chart data
  const barData = attrs.flatMap(attr => [
    { name: `${ATTR_LABELS[attr]} LR`,  value: audit[attr]?.logistic_regression?.auc_roc || 0, attr, type: 'lr' },
    { name: `${ATTR_LABELS[attr]} MLP`, value: audit[attr]?.mlp?.auc_roc || 0,                 attr, type: 'mlp' },
  ]);

  // Radar data
  const radarData = attrs.map(attr => ({
    attribute: ATTR_LABELS[attr],
    LR:  Math.round((audit[attr]?.logistic_regression?.auc_roc || 0.5) * 100),
    MLP: Math.round((audit[attr]?.mlp?.auc_roc || 0.5) * 100),
  }));

  return (
    <div className="fade-in">
      {/* ── Top bar ─────────────────────── */}
      <div className="topbar">
        <div className="topbar-breadcrumb">Privacy Framework › <strong>Module 1: Leakage Auditing</strong></div>
        <div className="topbar-right">
          <span className="topbar-badge">MODULE 01</span>
          <Badge level={summary?.overall_risk} />
        </div>
      </div>

      <div className="page-content">
        <div className="page-header" style={{ marginTop: 8 }}>
          <div className="module-badge" style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #dbeafe' }}>MODULE 01</div>
          <h1 className="page-title">Attribute Leakage Auditing</h1>
          <p className="page-subtitle">Measuring sensitive attribute inference from 512-dim face embeddings using LR &amp; MLP attacker classifiers</p>
        </div>

        {/* ── Overall Summary ──────────────────────────── */}
        <div className="card card-padded" style={{ marginBottom: 18, borderLeft: '4px solid #dc2626', background: summary?.overall_risk === 'CRITICAL' ? '#fff5f5' : '#f0fdf4' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <div style={{ fontSize: 36 }}>🛡️</div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
                Overall Privacy Risk Assessment
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                <Badge level={summary?.overall_risk} />
                <span style={{ fontSize: 13, color: '#334155' }}>
                  Mean Leakage: <strong>{(summary?.mean_leakage || 0).toFixed(4)}</strong>
                </span>
                <span style={{ fontSize: 13, color: '#334155' }}>
                  Max AUC: <strong>{(summary?.max_auc || 0).toFixed(4)}</strong>
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ── Per-Attribute Cards ───────────────────────── */}
        <div className="section-label">Per-Attribute Leakage Metrics</div>
        <div className="grid-3" style={{ marginBottom: 20 }}>
          {attrs.map((attr, i) => {
            const d = audit[attr];
            if (!d) return null;
            const color = ATTR_COLORS[attr];
            const auc = Math.max(d.logistic_regression?.auc_roc || 0, d.mlp?.auc_roc || 0);
            return (
              <motion.div key={attr} className="card" initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }}>
                <div className="card-header" style={{ borderLeftColor: color, borderLeftWidth: 3, borderLeftStyle: 'solid' }}>
                  <div className="card-title">{ATTR_LABELS[attr]}</div>
                  <Badge level={d.risk_level} />
                </div>
                <div className="card-body">
                  {/* AUC display */}
                  <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
                    {[
                      { lbl: 'LR AUC',  val: d.logistic_regression?.auc_roc },
                      { lbl: 'MLP AUC', val: d.mlp?.auc_roc },
                    ].map(({ lbl, val }) => (
                      <div key={lbl} style={{ flex: 1, background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 12px', textAlign: 'center' }}>
                        <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 3, fontWeight: 600 }}>{lbl}</div>
                        <div style={{ fontSize: 20, fontWeight: 800, color, letterSpacing: '-0.02em' }}>{(val || 0).toFixed(3)}</div>
                      </div>
                    ))}
                  </div>

                  {/* Progress bar */}
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                      <span>Leakage Severity</span><span>{(auc * 100).toFixed(1)}%</span>
                    </div>
                    <div className="auc-bar">
                      <div className="auc-bar-fill" style={{ width: `${auc * 100}%`, background: `linear-gradient(90deg, ${color}, ${color}99)` }} />
                    </div>
                  </div>

                  {/* Detailed stats grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                    {[
                      ['Accuracy',      d.mlp?.accuracy],
                      ['Balanced Acc.', d.mlp?.balanced_accuracy],
                      ['Leakage Score', d.mlp?.leakage_score],
                      ['Mutual Info',   d.mlp?.mutual_information],
                    ].map(([label, val]) => (
                      <div key={label} style={{ background: '#f8fafc', borderRadius: 6, padding: '7px 10px' }}>
                        <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 1 }}>{label}</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: '#334155' }}>{(val || 0).toFixed(4)}</div>
                      </div>
                    ))}
                  </div>

                  {/* Class labels */}
                  <div style={{ marginTop: 10, display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                    {(d.labels || []).map(l => (
                      <span key={l} style={{ background: `${color}12`, color, border: `1px solid ${color}30`, borderRadius: 99, padding: '2px 7px', fontSize: 10, fontWeight: 600 }}>
                        {l}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* ── AUC Bar + Radar ──────────────────────────── */}
        <div className="grid-2" style={{ marginBottom: 20 }}>
          <div className="card">
            <div className="card-header">
              <div className="card-title">AUC Comparison</div>
              <span style={{ fontSize: 11, color: '#94a3b8' }}>LR vs. MLP Attacker · per Attribute</span>
            </div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={barData} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#94a3b8' }} angle={-15} textAnchor="end" />
                  <YAxis domain={[0.4, 1.0]} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={0.5}  stroke="#e2e8f0" strokeDasharray="4" />
                  <ReferenceLine y={0.75} stroke="#fcd34d" strokeDasharray="4" />
                  <ReferenceLine y={0.90} stroke="#fca5a5" strokeDasharray="4" />
                  <Bar dataKey="value" radius={[4,4,0,0]} maxBarSize={40}>
                    {barData.map((d, i) => <Cell key={i} fill={ATTR_COLORS[d.attr] || '#2563eb'} opacity={d.type === 'lr' ? 1 : 0.6} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title">Leakage Profile Radar</div>
              <span style={{ fontSize: 11, color: '#94a3b8' }}>Multi-attribute · AUC%</span>
            </div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={230}>
                <RadarChart cx="50%" cy="50%" outerRadius={90} data={radarData}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="attribute" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[40, 100]} tick={{ fontSize: 9, fill: '#94a3b8' }} />
                  <Radar name="LR"  dataKey="LR"  stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} strokeWidth={2} />
                  <Radar name="MLP" dataKey="MLP" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.1}  strokeWidth={2} strokeDasharray="4" />
                  <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 12 }} formatter={v => [`${v}%`, 'AUC']} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* ── Interpretation ───────────────────────────── */}
        <div className="card card-padded" style={{ background: '#eff6ff', borderColor: '#dbeafe' }}>
          <div className="section-label" style={{ color: '#1d4ed8' }}>Interpretation Guide</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 32px', fontSize: 13, color: '#334155', lineHeight: 1.7 }}>
            <div>🔴 <strong>AUC &gt; 0.90 (CRITICAL)</strong> — Strong linear separability in embedding space</div>
            <div>🟡 <strong>AUC &gt; 0.75 (HIGH)</strong> — Non-trivial leakage; adversarial mitigation required</div>
            <div>🔵 <strong>AUC &gt; 0.60 (MEDIUM)</strong> — Moderate; noise injection may suffice</div>
            <div>🟢 <strong>AUC ≈ 0.50 (LOW)</strong> — Near-random; attribute information suppressed</div>
          </div>
          <div style={{ marginTop: 10, fontSize: 11.5, color: '#64748b', fontStyle: 'italic' }}>
            Ref: Morales et al. (2020) "SensitiveNets", IEEE TPAMI · Terhorst et al. (2021) "Comprehensive Study of Face Recognition Biases", IEEE TIFS
          </div>
        </div>
      </div>
    </div>
  );
}
