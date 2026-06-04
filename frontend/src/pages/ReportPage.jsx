import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import Badge from '../components/Badge';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, Legend
} from 'recharts';
import toast from 'react-hot-toast';
import { MdDownload, MdVerified, MdWarning } from 'react-icons/md';

const API = 'http://127.0.0.1:8000';
const ATTR_COLORS = { gender: '#2563eb', age_group: '#7c3aed', ethnicity: '#059669' };
const RISK_COLOR  = { CRITICAL: '#dc2626', HIGH: '#f59e0b', MEDIUM: '#3b82f6', LOW: '#22c55e', NEGLIGIBLE: '#64748b' };

function IntegrityChip({ valid }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '3px 10px', borderRadius: 99, fontSize: 11, fontWeight: 700,
      background: valid ? '#f0fdf4' : '#fef2f2',
      color:      valid ? '#15803d' : '#dc2626',
      border:     `1px solid ${valid ? '#bbf7d0' : '#fecaca'}`,
    }}>
      {valid ? <MdVerified /> : <MdWarning />}
      {valid ? 'Chain Intact' : 'Chain Broken!'}
    </span>
  );
}

export default function ReportPage() {
  const [report,      setReport]      = useState(null);
  const [fairness,    setFairness]    = useState(null);
  const [auditTrail,  setAuditTrail]  = useState(null);
  const [loading,     setLoading]     = useState(true);
  const [generating,  setGenerating]  = useState(false);

  const fetchAll = useCallback(async () => {
    const [r, f, at] = await Promise.allSettled([
      axios.get(`${API}/report`),
      axios.get(`${API}/metrics/fairness`),
      axios.get(`${API}/report/audit-trail`),
    ]);
    if (r.status === 'fulfilled'  && r.value.data?.data)  setReport(r.value.data.data);
    if (f.status === 'fulfilled'  && f.value.data?.data)  setFairness(f.value.data.data);
    if (at.status === 'fulfilled' && at.value.data)       setAuditTrail(at.value.data);
  }, []);

  useEffect(() => {
    fetchAll().finally(() => setLoading(false));
  }, [fetchAll]);

  const handleGenerateReport = async () => {
    setGenerating(true);
    try {
      await axios.post(`${API}/pipeline/run?dataset=synthetic&sweep=false`);
      toast.success('Pipeline started — report will update when complete.');

      const poll = setInterval(async () => {
        try {
          const { data } = await axios.get(`${API}/pipeline/status`);
          if (data.status === 'completed') {
            clearInterval(poll);
            await fetchAll();
            setGenerating(false);
            toast.success('Report generated successfully!');
          } else if (data.status === 'error') {
            clearInterval(poll);
            setGenerating(false);
            toast.error('Pipeline failed — check backend logs.');
          }
        } catch { clearInterval(poll); setGenerating(false); }
      }, 3000);
    } catch {
      setGenerating(false);
      toast.error('Could not start pipeline. Is the backend running?');
    }
  };

  const handleDownload = async () => {
    try {
      const resp = await axios.get(`${API}/report/download`, { responseType: 'blob' });
      const url  = URL.createObjectURL(new Blob([resp.data], { type: 'application/json' }));
      const a    = document.createElement('a');
      a.href     = url;
      a.download = 'obscura_model_card.json';
      a.click();
      URL.revokeObjectURL(url);
      toast.success('Model card downloaded.');
    } catch {
      toast.error('No report available to download yet.');
    }
  };

  if (loading) return <div style={{ padding: 40, color: '#64748b' }}>Loading report…</div>;

  // ── Derived data from correct API shape ─────────────────────────────────────
  const privacy    = report?.privacy_analysis   || {};
  const mitigation = report?.mitigation          || {};
  const utility    = report?.utility             || {};
  const fairData   = report?.fairness            || {};
  const baseline   = privacy.baseline            || {};

  // Per-attribute rows (from baseline privacy analysis)
  const attrRows = ['gender', 'age_group', 'ethnicity'].map(attr => ({
    name:    attr.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()),
    attr,
    best_auc:     baseline[attr]?.best_auc          || 0,
    leakage:      baseline[attr]?.leakage_score      || 0,
    risk:         baseline[attr]?.risk_level         || 'UNKNOWN',
  }));

  // Group-level fairness data for chart
  const groupData = Object.entries(fairData.per_group_leakage || {}).map(([group, vals]) => ({
    name:   group,
    mean:   parseFloat((vals.mean_leakage || 0).toFixed(4)),
    max:    parseFloat((vals.max_leakage  || 0).toFixed(4)),
  }));

  // Comparison table (from mitigation)
  const comparison = mitigation.comparison_table || [];

  // TAR@FAR utility metrics
  const tarFarEntries = Object.entries(utility)
    .filter(([k]) => k.startsWith('TAR@'))
    .map(([k, v]) => ({ metric: k, value: typeof v === 'number' ? v.toFixed(4) : v }));

  // Audit trail
  const integrity = auditTrail?.integrity || {};
  const logEntries = (auditTrail?.entries || []).slice(-15).reverse();

  const LEVEL_COLOR = { INFO: '#2563eb', ERROR: '#dc2626', WARNING: '#f59e0b' };

  return (
    <div className="fade-in">
      {/* ── Topbar ─────────────────────────── */}
      <div className="topbar">
        <div className="topbar-breadcrumb">
          ObscuraAI › <strong>Module 3: Reporting &amp; Accountability</strong>
        </div>
        <div className="topbar-right">
          <span className="topbar-badge">MODULE 03</span>
          {report?.audit_hash && (
            <span className="topbar-badge" style={{ fontFamily: 'monospace', fontSize: 10.5 }}>
              SHA-256: {report.audit_hash.slice(0, 12)}…
            </span>
          )}
          <button className="btn btn-secondary btn-sm" onClick={handleDownload}>
            <MdDownload /> Export JSON
          </button>
          <button
            className="btn btn-primary btn-sm"
            onClick={handleGenerateReport}
            disabled={generating}
            style={{ minWidth: 150, opacity: generating ? 0.7 : 1 }}
          >
            {generating ? '⏳ Generating…' : '📄 Generate Report'}
          </button>
        </div>
      </div>

      <div className="page-content">
        <div className="page-header" style={{ marginTop: 8 }}>
          <div className="module-badge" style={{ background: '#f0fdf4', color: '#15803d', border: '1px solid #bbf7d0' }}>
            MODULE 03
          </div>
          <h1 className="page-title">Reporting &amp; Accountability</h1>
          <p className="page-subtitle">
            IEEE-standardised model cards, formal privacy reporting, and a cryptographically chained audit trail
          </p>
        </div>

        {!report && !generating && (
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="empty-state">
              <div className="empty-state-icon">📋</div>
              <h3>No Report Yet</h3>
              <p>Click <strong>Generate Report</strong> to run the pipeline and produce the model card.</p>
            </div>
          </div>
        )}

        {report && (
          <>
            {/* ── Row 1: Model Card summary + Privacy metrics ── */}
            <div className="grid-2" style={{ marginBottom: 20 }}>

              {/* Model Card */}
              <motion.div className="card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="card-header">
                  <div className="card-title">📋 Model Card Summary</div>
                  <Badge level={privacy.overall_risk} />
                </div>
                <div className="card-body">
                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 14px', marginBottom: 14 }}>
                    <div className="section-label" style={{ marginBottom: 4 }}>Framework</div>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11.5, color: '#334155', marginBottom: 3 }}>
                      {report.framework || 'ObscuraAI Privacy Intelligence Framework'}
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>
                      Dataset: {report.dataset?.name || '—'} · {report.dataset?.n_samples?.toLocaleString() || '—'} samples · {report.dataset?.embedding_dim || 512}-dim
                    </div>
                  </div>

                  <div className="section-label">Mitigation Techniques Applied</div>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
                    {(mitigation.techniques_applied || []).map(t => (
                      <div key={t} style={{
                        background: '#eff6ff', border: '1px solid #dbeafe',
                        borderRadius: 99, padding: '4px 10px', fontSize: 11.5,
                        color: '#1e40af', fontWeight: 600,
                      }}>
                        {t.replace('_', ' ')}
                      </div>
                    ))}
                  </div>

                  <div className="section-label">Privacy Leakage — Per Attribute</div>
                  {attrRows.map(row => (
                    <div key={row.attr} className="attr-row" style={{ marginBottom: 6 }}>
                      <div className="attr-label" style={{ minWidth: 90 }}>{row.name}</div>
                      <div className="attr-bar-wrap">
                        <div className="auc-bar">
                          <div className="auc-bar-fill" style={{
                            width: `${row.best_auc * 100}%`,
                            background: RISK_COLOR[row.risk] || '#64748b',
                          }} />
                        </div>
                      </div>
                      <div className="attr-value" style={{ color: RISK_COLOR[row.risk] }}>
                        {row.best_auc.toFixed(3)}
                      </div>
                      <Badge level={row.risk} />
                    </div>
                  ))}

                  <div style={{
                    marginTop: 14, padding: '10px 14px',
                    background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#64748b',
                  }}>
                    <strong>Mean Leakage Score:</strong> {(privacy.mean_leakage_score || 0).toFixed(4)} &nbsp;|&nbsp;
                    <strong>Generated:</strong> {report.generated_at ? new Date(report.generated_at).toLocaleString() : '—'}
                  </div>
                </div>
              </motion.div>

              {/* Privacy Report */}
              <motion.div className="card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}>
                <div className="card-header">
                  <div className="card-title">🔒 Privacy Report</div>
                  {report.audit_hash && <IntegrityChip valid={integrity.valid !== false} />}
                </div>
                <div className="card-body">
                  {/* Audit Hash Display */}
                  {report.audit_hash && (
                    <div style={{
                      background: '#0f172a', borderRadius: 8, padding: '12px 14px', marginBottom: 14,
                      fontFamily: 'JetBrains Mono, monospace', fontSize: 10.5,
                    }}>
                      <div style={{ color: '#64748b', fontSize: 10, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                        SHA-256 Audit Hash
                      </div>
                      <div style={{ color: '#34d399', wordBreak: 'break-all' }}>
                        {report.audit_hash}
                      </div>
                    </div>
                  )}

                  {/* TAR@FAR utility */}
                  {tarFarEntries.length > 0 && (
                    <>
                      <div className="section-label">Face Verification Utility (TAR@FAR)</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 14 }}>
                        {tarFarEntries.map(({ metric, value }) => (
                          <div key={metric} style={{
                            background: '#f8fafc', border: '1px solid #e2e8f0',
                            borderRadius: 6, padding: '8px 10px',
                            display: 'flex', justifyContent: 'space-between',
                          }}>
                            <span style={{ fontSize: 11, color: '#64748b' }}>{metric}</span>
                            <span style={{ fontWeight: 800, color: '#2563eb', fontSize: 14 }}>{value}</span>
                          </div>
                        ))}
                        {utility.mean_genuine_similarity != null && (
                          <div style={{
                            gridColumn: '1 / -1', background: '#f0fdf4', border: '1px solid #bbf7d0',
                            borderRadius: 6, padding: '8px 10px',
                            display: 'flex', justifyContent: 'space-between',
                          }}>
                            <span style={{ fontSize: 11, color: '#64748b' }}>Mean Genuine Similarity</span>
                            <span style={{ fontWeight: 800, color: '#15803d', fontSize: 14 }}>
                              {(utility.mean_genuine_similarity || 0).toFixed(4)}
                            </span>
                          </div>
                        )}
                      </div>
                    </>
                  )}

                  {/* Recommendations */}
                  <div className="section-label">Recommendations</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {(report.recommendations || []).map((rec, i) => {
                      const isWarn = rec.startsWith('CRITICAL') || rec.startsWith('HIGH') || rec.startsWith('FAIRNESS') || rec.startsWith('EQUALISED');
                      return (
                        <div key={i} style={{
                          padding: '8px 12px', borderRadius: 6,
                          background: isWarn ? '#fffbeb' : '#f8fafc',
                          border: `1px solid ${isWarn ? '#fde68a' : '#e2e8f0'}`,
                          fontSize: 12, color: '#334155', lineHeight: 1.5,
                        }}>
                          {isWarn ? '⚠️ ' : '✓ '}{rec}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </motion.div>
            </div>

            {/* ── Row 2: Fairness Analysis ─── */}
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header">
                <div className="card-title">⚖️ Fairness Analysis — Leakage Disparity Across Demographic Groups</div>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>
                  MDD = {(fairData.max_demographic_disparity || 0).toFixed(4)} &nbsp;|&nbsp;
                  EOD = {(fairData.equalised_odds_difference || 0).toFixed(4)}
                </span>
              </div>
              <div className="card-body">
                <div className="grid-2">
                  {/* Group leakage bar chart */}
                  <div>
                    <div className="section-label">Per-Group Leakage Score (Ethnicity Groups)</div>
                    {groupData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={groupData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                          <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} domain={[0, 1]} />
                          <Tooltip
                            contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 12 }}
                            formatter={v => [v.toFixed(4), '']}
                          />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar dataKey="mean" name="Mean Leakage" fill="#2563eb" radius={[3,3,0,0]} maxBarSize={40} />
                          <Bar dataKey="max"  name="Max Leakage"  fill="#7c3aed" radius={[3,3,0,0]} maxBarSize={40} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: 13 }}>
                        No group-level data available
                      </div>
                    )}
                  </div>

                  {/* Comparison table */}
                  <div>
                    <div className="section-label">Before / After AUC (Comparison Table)</div>
                    {comparison.length > 0 ? (
                      <div className="table-wrap" style={{ marginTop: 0 }}>
                        <table>
                          <thead>
                            <tr>
                              <th>Attribute</th>
                              <th>Baseline</th>
                              <th>Adversarial</th>
                              <th>Noise</th>
                              <th>Risk Change</th>
                            </tr>
                          </thead>
                          <tbody>
                            {comparison.map(r => (
                              <tr key={r.attribute}>
                                <td style={{ fontWeight: 700 }}>
                                  {r.attribute.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                </td>
                                <td style={{ color: '#dc2626', fontWeight: 700 }}>{r.baseline_auc?.toFixed(4)}</td>
                                <td style={{ color: '#2563eb', fontWeight: 700 }}>
                                  {r.adv_auc?.toFixed(4)}
                                  <span style={{ color: '#16a34a', fontSize: 10, marginLeft: 4 }}>↓{r.adv_reduction_pct?.toFixed(1)}%</span>
                                </td>
                                <td style={{ color: '#7c3aed', fontWeight: 700 }}>
                                  {r.noise_auc?.toFixed(4)}
                                  <span style={{ color: '#16a34a', fontSize: 10, marginLeft: 4 }}>↓{r.noise_reduction_pct?.toFixed(1)}%</span>
                                </td>
                                <td>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                    <Badge level={r.baseline_risk} />
                                    <span style={{ color: '#94a3b8' }}>→</span>
                                    <Badge level={r.adv_risk} />
                                  </div>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 20 }}>Run the pipeline to see comparison data.</div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* ── Attribute Narratives ──────────── */}
            {report.attribute_narratives && Object.keys(report.attribute_narratives).length > 0 && (
              <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header">
                  <div className="card-title">📝 Attribute Leakage Narratives</div>
                </div>
                <div className="card-body">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {Object.entries(report.attribute_narratives).map(([attr, text]) => (
                      <div key={attr} style={{
                        padding: '12px 16px', borderRadius: 8,
                        background: '#f8fafc', border: '1px solid #e2e8f0',
                        borderLeft: `4px solid ${ATTR_COLORS[attr] || '#64748b'}`,
                      }}>
                        <div style={{ fontWeight: 700, color: ATTR_COLORS[attr] || '#64748b', fontSize: 12, marginBottom: 4 }}>
                          {attr.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                        </div>
                        <div style={{ fontSize: 12.5, color: '#475569', lineHeight: 1.6 }}>{text}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* ── Cryptographic Audit Trail ──── */}
        <div className="card" style={{ marginBottom: 20, overflow: 'hidden' }}>
          <div style={{ background: '#0f172a', padding: '14px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ color: '#f8fafc', fontWeight: 700, fontSize: 13, display: 'flex', alignItems: 'center', gap: 8 }}>
                🔒 Cryptographic Audit Trail
                {auditTrail && <IntegrityChip valid={integrity.valid !== false} />}
              </div>
              <div style={{ color: '#64748b', fontSize: 11.5, marginTop: 2 }}>
                SHA-256 blockchain-style chained log — every entry hashes the previous entry's hash.
                {integrity.n_entries > 0 && ` ${integrity.n_entries} entries · Chain head: ${String(integrity.chain_head || '').slice(0, 16)}…`}
              </div>
            </div>
            <span style={{
              background: '#1e293b', color: '#64748b', border: '1px solid #334155',
              borderRadius: 6, padding: '4px 10px', fontSize: 10.5, fontFamily: 'monospace',
            }}>
              {logEntries.length} of {auditTrail?.total || 0} shown
            </span>
          </div>

          <div style={{ padding: '0 20px', maxHeight: 320, overflowY: 'auto' }}>
            {logEntries.length === 0 ? (
              <div style={{ padding: '20px 0', color: '#94a3b8', fontSize: 13, textAlign: 'center' }}>
                No audit entries yet — run the pipeline to generate a log.
              </div>
            ) : (
              logEntries.map((entry, i) => (
                <div key={entry.seq ?? i} className="log-entry" style={{ alignItems: 'flex-start', gap: 12 }}>
                  <div style={{ minWidth: 190, display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <div className="log-time">
                      {new Date(entry.timestamp).toLocaleString()}
                    </div>
                    <div style={{
                      fontSize: 9.5, fontFamily: 'monospace', color: '#475569',
                    }}>
                      #{entry.seq} · {entry.entry_hash?.slice(0, 12)}…
                    </div>
                  </div>
                  <div style={{ flex: 1 }}>
                    <span style={{
                      fontSize: 10, fontWeight: 700, marginRight: 7,
                      color: LEVEL_COLOR[entry.level] || '#64748b',
                      textTransform: 'uppercase',
                    }}>
                      [{entry.module}]
                    </span>
                    <span className="log-msg">{entry.message}</span>
                  </div>
                  <span style={{
                    fontSize: 10.5, fontWeight: 700,
                    color: LEVEL_COLOR[entry.level] || '#64748b',
                    minWidth: 48,
                  }}>
                    {entry.level}
                  </span>
                </div>
              ))
            )}
          </div>

          <div style={{ background: '#f8fafc', padding: '12px 20px', borderTop: '1px solid #e2e8f0', display: 'flex', gap: 10 }}>
            <button className="btn btn-secondary btn-sm" onClick={() => fetchAll()}>
              🔄 Refresh Log
            </button>
            <button className="btn btn-primary btn-sm" onClick={handleDownload}>
              📥 Download Model Card JSON
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
