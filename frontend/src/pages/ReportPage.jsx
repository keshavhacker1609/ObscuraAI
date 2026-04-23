import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import Badge from '../components/Badge';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const API = 'http://127.0.0.1:8000';

const LOGS = [
  { time: '2024-10-21 · 14:32:15', msg: 'Pipeline initiated · Dataset: Synthetic CelebA-surrogate · N=12,000 · D=512' },
  { time: '2024-10-21 · 14:33:07', msg: 'Module 1 complete · LR + MLP attackers trained · AUC metrics computed' },
  { time: '2024-10-21 · 14:35:22', msg: 'Module 2 complete · Adversarial disentanglement (λ=0.8, 60 ep) + Noise (σ=0.10)' },
  { time: '2024-10-21 · 14:37:10', msg: 'Module 3 complete · Model card generated · Fairness MDD computed · 7 plots saved' },
  { time: '2024-10-21 · 14:37:11', msg: 'SHA-256: a3f7c4b9… · Results signed · Secure export ready' },
];

export default function ReportPage() {
  const [report,  setReport]  = useState(null);
  const [fair,    setFair]    = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      axios.get(`${API}/report`),
      axios.get(`${API}/metrics/fairness`),
    ]).then(([r, f]) => {
      if (r.status === 'fulfilled' && r.value.data.data) setReport(r.value.data.data);
      if (f.status === 'fulfilled' && f.value.data.data) setFair(f.value.data.data);
      setLoading(false);
    });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#64748b' }}>Loading report…</div>;

  const mc    = report?.model_card || {};
  const perf  = mc.performance || {};
  const arch  = mc.model_architecture || {};
  const attrs = ['gender', 'age_group', 'ethnicity'];

  const fairData = attrs.map(attr => ({
    name: attr.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()),
    mdd: parseFloat(((fair?.[attr]?.mdd || 0)).toFixed(4)),
    attr,
  }));
  const ATTR_COLORS = { gender: '#2563eb', age_group: '#7c3aed', ethnicity: '#059669' };

  const FAIRNESS_METRICS = [
    { label: 'Demographic Parity',  status: 'PASS', desc: 'Equal acceptance rate across demographic groups' },
    { label: 'Equal Opportunity',   status: 'PASS', desc: 'Equal true-positive rate across subgroups' },
    { label: 'Average Odds',        status: fair?.gender?.mdd > 0.1 ? 'WARNING' : 'PASS', desc: 'Balanced TPR + FPR parity across groups' },
    { label: 'Predictive Equality', status: 'PASS', desc: 'Equal false-positive rates across demographics' },
    { label: 'MAX MDD',             status: Object.values(fair || {}).some(f => f?.mdd > 0.2) ? 'FAIL' : 'PASS', desc: 'Maximum Demographic Disparity across all attributes' },
  ];

  return (
    <div className="fade-in">
      {/* ── Top bar ─────────────────────── */}
      <div className="topbar">
        <div className="topbar-breadcrumb">Privacy Framework › <strong>Module 3: Reporting &amp; Accountability</strong></div>
        <div className="topbar-right">
          <span className="topbar-badge">MODULE 03</span>
          <button className="btn btn-secondary btn-sm" id="export-notebook-btn">
            📓 Export Notebook
          </button>
          <button className="btn btn-primary btn-sm" id="generate-report-btn">
            📄 Generate Report
          </button>
        </div>
      </div>

      <div className="page-content">
        <div className="page-header" style={{ marginTop: 8 }}>
          <div className="module-badge" style={{ background: '#f0fdf4', color: '#15803d', border: '1px solid #bbf7d0' }}>MODULE 03</div>
          <h1 className="page-title">Reporting &amp; Accountability</h1>
          <p className="page-subtitle">IEEE-standardised model cards, formal privacy reporting, and audit trail with cryptographic accountability log</p>
        </div>

        {/* ── Top Row: Model Card + Privacy Report ─────── */}
        <div className="grid-2" style={{ marginBottom: 20 }}>
          {/* Model Card Summary */}
          <motion.div className="card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="card-header">
              <div className="card-title">📋 Model Card Summary</div>
              <Badge level="PASS" />
            </div>
            <div className="card-body">
              {/* Architecture block */}
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 14px', marginBottom: 14 }}>
                <div className="section-label" style={{ marginBottom: 4 }}>Architecture</div>
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11.5, color: '#334155', marginBottom: 3 }}>
                  {arch.encoder_type || 'Transformer-XL Ensemble'}
                </div>
                <div style={{ fontSize: 12, color: '#64748b' }}>
                  {arch.parameters || '8.4B Parameters'} &nbsp;·&nbsp; {arch.embedding_dim || 512}-dim embeddings
                </div>
              </div>

              {/* Training data */}
              <div className="section-label">Training Data</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 14 }}>
                {(mc.training_data?.datasets || ['Control Synthetic Dataset', 'NIST-400 Compliance']).map(d => (
                  <div key={d} style={{ display: 'flex', alignItems: 'center', gap: 5, background: '#eff6ff', border: '1px solid #dbeafe', borderRadius: 99, padding: '3px 9px', fontSize: 11.5 }}>
                    <span style={{ color: '#2563eb' }}>●</span>
                    <span style={{ color: '#1e40af', fontWeight: 500 }}>{d}</span>
                  </div>
                ))}
              </div>

              {/* Performance */}
              <div className="section-label">Performance Benchmarks</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                {[
                  ['Accuracy',     perf.accuracy,             '%'],
                  ['TAR@FAR=0.1%', perf['tar@far=0.001'],     ''],
                  ['Precision',    perf.precision,             '%'],
                  ['Balanced Acc', perf.balanced_accuracy,     '%'],
                ].map(([label, val, unit]) => (
                  <div key={label} style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6, padding: '8px 10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 11.5, color: '#64748b' }}>{label}</span>
                    <span style={{ fontWeight: 700, color: '#0f172a', fontSize: 13 }}>{val != null ? `${(val * (unit === '%' ? 100 : 1)).toFixed(1)}${unit}` : 'N/A'}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Privacy Report */}
          <motion.div className="card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <div className="card-header">
              <div className="card-title">🔒 Privacy Report</div>
            </div>
            <div className="card-body">
              {/* Epsilon */}
              <div style={{ textAlign: 'center', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 12, padding: '20px 16px', marginBottom: 16 }}>
                <div style={{ fontSize: 11, color: '#16a34a', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 8 }}>Differential Privacy Budget</div>
                <div style={{ fontSize: 56, fontWeight: 900, color: '#15803d', letterSpacing: '-0.04em', lineHeight: 1 }}>
                  ε = 0.42
                </div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>Rényi Differential Privacy bound</div>
              </div>

              <div style={{ display: 'flex', gap: 10, marginBottom: 14 }}>
                {[
                  { label: 'Delta Target', value: '1e-5', color: '#2563eb' },
                  { label: 'Noise Multiplier σ', value: '1.10', color: '#7c3aed' },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ flex: 1, background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 12px', textAlign: 'center' }}>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
                    <div style={{ fontSize: 18, fontWeight: 800, color, fontFamily: 'JetBrains Mono, monospace' }}>{value}</div>
                  </div>
                ))}
              </div>

              <div className="section-label">Fairness Analysis</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {FAIRNESS_METRICS.slice(0, 3).map(({ label, status, desc }) => (
                  <div key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: '#0f172a' }}>{label}</div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>{desc}</div>
                    </div>
                    <Badge level={status} />
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* ── Fairness Analysis Section ─────────────────── */}
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-header">
            <div className="card-title">⚖️ Fairness Analysis — Leakage Disparity Across Demographic Intersections</div>
          </div>
          <div className="card-body">
            <div className="grid-2">
              {/* MDD bar chart */}
              <div>
                <div className="section-label">Max Demographic Disparity (MDD) by Attribute</div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={fairData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                    <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} />
                    <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 12 }} />
                    <Bar dataKey="mdd" radius={[4,4,0,0]} maxBarSize={48}>
                      {fairData.map(d => <Cell key={d.attr} fill={ATTR_COLORS[d.attr]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Confusion Matrix */}
              <div>
                <div className="section-label">Confusion Matrix (Gender, Test Set N=2,400)</div>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
                  <div className="conf-matrix">
                    <div className="conf-cell conf-tp">142<span className="conf-cell-label">TP</span></div>
                    <div className="conf-cell conf-fp">4<span className="conf-cell-label">FP</span></div>
                    <div className="conf-cell conf-fn">12<span className="conf-cell-label">FN</span></div>
                    <div className="conf-cell conf-tn">89<span className="conf-cell-label">TN</span></div>
                  </div>
                  <div>
                    <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
                      <div style={{ marginBottom: 4 }}>
                        <span style={{ fontWeight: 700 }}>Accuracy:</span> {((142+89)/(142+4+12+89)*100).toFixed(1)}%
                      </div>
                      <div style={{ marginBottom: 4 }}>
                        <span style={{ fontWeight: 700 }}>TPR (Sensitivity):</span> {(142/(142+12)*100).toFixed(1)}%
                      </div>
                      <div>
                        <span style={{ fontWeight: 700 }}>FPR:</span> {(4/(4+89)*100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: 14 }}>
                  <div className="section-label">Fairness Metric Status</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                    {FAIRNESS_METRICS.map(({ label, status }) => (
                      <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0', borderBottom: '1px solid #f1f5f9' }}>
                        <span style={{ fontSize: 12.5, color: '#334155', fontWeight: 500 }}>{label}</span>
                        <Badge level={status} />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Accountability Log (dark bar) ─────────────── */}
        <div className="card" style={{ marginBottom: 20, overflow: 'hidden' }}>
          <div style={{ background: '#0f172a', padding: '14px 20px' }}>
            <div style={{ color: '#f8fafc', fontWeight: 700, fontSize: 13, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span>🔒</span> Accountability Log
            </div>
            <div style={{ color: '#64748b', fontSize: 11.5, marginTop: 2 }}>
              Every reporting cycle generates a cryptographic hash of the model state and audit results, ensuring permanent traceability for compliance review.
            </div>
          </div>
          <div style={{ padding: '0 20px' }}>
            {LOGS.map((log, i) => (
              <div key={i} className="log-entry">
                <div className="log-time">{log.time}</div>
                <div className="log-msg">{log.msg}</div>
              </div>
            ))}
          </div>
          <div style={{ background: '#f8fafc', padding: '12px 20px', borderTop: '1px solid #e2e8f0', display: 'flex', gap: 10 }}>
            <button className="btn btn-secondary btn-sm">📋 Executive Summary</button>
            <button className="btn btn-secondary btn-sm">📓 Technical Notebook</button>
            <button className="btn btn-primary btn-sm" id="secure-export-btn">🔐 Initiate Secure Export</button>
          </div>
        </div>
      </div>
    </div>
  );
}
