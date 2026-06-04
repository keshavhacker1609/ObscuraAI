import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import Badge from '../components/Badge';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, Legend, ReferenceLine,
  ScatterChart, Scatter, ZAxis
} from 'recharts';

const API = 'http://127.0.0.1:8000';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tt">
      <div className="custom-tt-label">{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color, fontWeight: 700, fontSize: 13 }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
};

const SweepTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload || {};
  return (
    <div className="custom-tt" style={{ fontSize: 11 }}>
      {d.lambda  != null && <div>λ = {d.lambda}</div>}
      {d.sigma   != null && <div>σ = {d.sigma}</div>}
      <div style={{ color: '#dc2626' }}>Leakage: {(d.mean_leakage_score || 0).toFixed(4)}</div>
      <div style={{ color: '#2563eb' }}>Utility: {(d.identity_utility || 0).toFixed(4)}</div>
      {d.snr_db  != null && <div style={{ color: '#64748b' }}>SNR: {d.snr_db.toFixed(1)} dB</div>}
    </div>
  );
};

export default function MitigationPage() {
  const [mitig,   setMitig]   = useState(null);
  const [tarFar,  setTarFar]  = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      axios.get(`${API}/mitigate`),
      axios.get(`${API}/metrics/utility`),
    ]).then(([m, t]) => {
      if (m.status === 'fulfilled' && m.value.data?.data) setMitig(m.value.data.data);
      if (t.status === 'fulfilled' && t.value.data?.data) setTarFar(t.value.data.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#64748b' }}>Loading mitigation data…</div>;

  const comparison = mitig?.comparison  || [];
  const advSweep   = mitig?.adv_sweep   || [];
  const noiseSweep = mitig?.noise_sweep || [];

  const barData = comparison.map(r => ({
    attr:        r.attribute.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()),
    Baseline:    parseFloat((r.baseline_auc || 0).toFixed(4)),
    Adversarial: parseFloat((r.adv_auc || 0).toFixed(4)),
    Noise:       parseFloat((r.noise_auc || 0).toFixed(4)),
  }));

  // TAR@FAR entries
  const tarFarEntries = Object.entries(tarFar || {})
    .filter(([k]) => k.startsWith('TAR@'))
    .map(([k, v]) => ({ metric: k, value: typeof v === 'number' ? v : 0 }));

  if (!mitig) return (
    <div className="fade-in">
      <div className="topbar">
        <div className="topbar-breadcrumb">ObscuraAI › <strong>Module 2: Mitigation</strong></div>
      </div>
      <div className="page-content">
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">🛡️</div>
            <h3>No Mitigation Results</h3>
            <p>Run the full pipeline from the dashboard first.</p>
          </div>
        </div>
      </div>
    </div>
  );

  const avgAdv   = comparison.reduce((s, r) => s + (r.adv_reduction_pct   || 0), 0) / Math.max(comparison.length, 1);
  const avgNoise = comparison.reduce((s, r) => s + (r.noise_reduction_pct || 0), 0) / Math.max(comparison.length, 1);

  return (
    <div className="fade-in">
      {/* ── Topbar ──────────────────────────── */}
      <div className="topbar">
        <div className="topbar-breadcrumb">ObscuraAI › <strong>Module 2: Leakage Mitigation</strong></div>
        <div className="topbar-right">
          <span className="topbar-badge">MODULE 02</span>
          <span className="topbar-badge" style={{ background: '#f5f3ff', color: '#6d28d9', border: '1px solid #ddd6fe' }}>
            2 Techniques Applied
          </span>
        </div>
      </div>

      <div className="page-content">
        <div className="page-header" style={{ marginTop: 8 }}>
          <div className="module-badge" style={{ background: '#f5f3ff', color: '#6d28d9', border: '1px solid #ddd6fe' }}>MODULE 02</div>
          <h1 className="page-title">Leakage Mitigation Workbench</h1>
          <p className="page-subtitle">
            Adversarial Attribute Disentanglement (GRL + λ-sweep) &amp; Gaussian Noise Injection (σ-sweep).
            Utility–privacy Pareto frontier analysis with TAR@FAR identity preservation metrics.
          </p>
        </div>

        {/* ── Technique Cards ─────────────────── */}
        <div className="grid-2" style={{ marginBottom: 20 }}>
          <motion.div className="card" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}>
            <div className="card-header" style={{ borderLeft: '4px solid #2563eb' }}>
              <div className="card-title">🎭 Adversarial Disentanglement</div>
            </div>
            <div className="card-body">
              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.75, marginBottom: 14 }}>
                <div><strong>Architecture:</strong> EmbeddingProjector + GradientReversalLayer + AttributeDiscriminator</div>
                <div><strong>Loss:</strong> L<sub>total</sub> = L<sub>cosine</sub> + λ · L<sub>adversarial</sub></div>
                <div><strong>Config:</strong> λ = 0.8 · Epochs = 60 · Grid: {'{0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0}'}</div>
                <div><strong>Reference:</strong> Ganin &amp; Lempitsky (2015), Morales et al. (2020 IEEE TPAMI)</div>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {comparison.map(r => (
                  <div key={r.attribute} style={{
                    flex: 1, minWidth: 80, background: '#eff6ff', border: '1px solid #dbeafe',
                    borderRadius: 8, padding: '10px 12px', textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 10, color: '#64748b', marginBottom: 3, textTransform: 'uppercase', fontWeight: 600 }}>
                      {r.attribute.replace('_', ' ')}
                    </div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: '#1d4ed8' }}>
                      ↓{r.adv_reduction_pct?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 10, color: '#22c55e', fontWeight: 600 }}>AUC reduction</div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          <motion.div className="card" initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.08 }}>
            <div className="card-header" style={{ borderLeft: '4px solid #7c3aed' }}>
              <div className="card-title">💨 Gaussian Noise Injection</div>
            </div>
            <div className="card-body">
              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.75, marginBottom: 14 }}>
                <div><strong>Method:</strong> ε ~ N(0, σ²) added to L2-normalised embeddings</div>
                <div><strong>Best σ:</strong> 0.10 · Grid: {'[0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]'}</div>
                <div><strong>Metrics:</strong> SNR, cosine drift, attacker AUC reduction</div>
                <div><strong>Reference:</strong> Mirjalili et al. (2020 IEEE TIP), Dwork et al. (2014)</div>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {comparison.map(r => (
                  <div key={r.attribute} style={{
                    flex: 1, minWidth: 80, background: '#f5f3ff', border: '1px solid #ddd6fe',
                    borderRadius: 8, padding: '10px 12px', textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 10, color: '#64748b', marginBottom: 3, textTransform: 'uppercase', fontWeight: 600 }}>
                      {r.attribute.replace('_', ' ')}
                    </div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: '#6d28d9' }}>
                      ↓{r.noise_reduction_pct?.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 10, color: '#22c55e', fontWeight: 600 }}>AUC reduction</div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* ── KPI Row ─────────────────────────── */}
        <div className="metrics-grid" style={{ marginBottom: 20 }}>
          {[
            { icon: '📉', label: 'Avg Adv Reduction',  value: `${avgAdv.toFixed(1)}%`,   sub: 'Adversarial method',       cls: 'green', bg: '#f0fdf4', iconBg: '#dcfce7' },
            { icon: '💨', label: 'Avg Noise Reduction', value: `${avgNoise.toFixed(1)}%`, sub: 'Noise injection',           cls: '',      bg: '#fff',   iconBg: '#f1f5f9' },
            { icon: '🎯', label: 'Identity Utility',
              value: tarFar?.mean_genuine_similarity != null
                ? `${(tarFar.mean_genuine_similarity * 100).toFixed(1)}%`
                : '—',
              sub: 'Mean genuine cos. similarity', cls: 'blue', bg: '#eff6ff', iconBg: '#dbeafe',
            },
            ...(tarFarEntries.slice(0, 3).map(({ metric, value }) => ({
              icon: '✅', label: metric, value: value.toFixed(3),
              sub: 'Sanitised embeddings', cls: 'green', bg: '#f0fdf4', iconBg: '#dcfce7',
            }))),
          ].map(({ icon, label, value, sub, cls, bg, iconBg }) => (
            <div key={label} className="metric-card" style={{ background: bg }}>
              <div className="metric-icon-wrap" style={{ background: iconBg }}>{icon}</div>
              <div className={`metric-value ${cls}`}>{value}</div>
              <div className="metric-label">{label}</div>
              <div className="metric-sub">{sub}</div>
            </div>
          ))}
        </div>

        {/* ── Before / After Chart ─────────────── */}
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-header">
            <div className="card-title">AUC Comparison: Baseline vs. Mitigated</div>
            <span style={{ fontSize: 11, color: '#94a3b8' }}>Lower AUC = less leakage</span>
          </div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={barData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }} barGap={4} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="attr" tick={{ fontSize: 12, fill: '#64748b' }} />
                <YAxis domain={[0.4, 1.0]} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#64748b', paddingTop: 8 }} />
                <ReferenceLine y={0.5}  stroke="#e2e8f0" strokeDasharray="4" />
                <ReferenceLine y={0.75} stroke="#fde68a" strokeDasharray="3" />
                <Bar dataKey="Baseline"    fill="#dc2626" radius={[4,4,0,0]} maxBarSize={38} />
                <Bar dataKey="Adversarial" fill="#2563eb" radius={[4,4,0,0]} maxBarSize={38} />
                <Bar dataKey="Noise"       fill="#7c3aed" radius={[4,4,0,0]} maxBarSize={38} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ── Detailed Table ──────────────────── */}
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-header">
            <div className="card-title">Detailed Comparison Table</div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Attribute</th>
                  <th>Baseline AUC</th>
                  <th>Adv. AUC</th>
                  <th>Adv. Δ</th>
                  <th>Noise AUC</th>
                  <th>Noise Δ</th>
                  <th>Baseline Risk</th>
                  <th>Post-Adv. Risk</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map(r => (
                  <tr key={r.attribute}>
                    <td style={{ fontWeight: 700, color: '#0f172a' }}>
                      {r.attribute.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </td>
                    <td style={{ fontWeight: 700, color: '#dc2626' }}>{r.baseline_auc?.toFixed(4)}</td>
                    <td style={{ fontWeight: 700, color: '#2563eb' }}>{r.adv_auc?.toFixed(4)}</td>
                    <td style={{ fontWeight: 700, color: '#16a34a' }}>↓ {r.adv_reduction_pct?.toFixed(2)}%</td>
                    <td style={{ fontWeight: 700, color: '#7c3aed' }}>{r.noise_auc?.toFixed(4)}</td>
                    <td style={{ fontWeight: 700, color: '#16a34a' }}>↓ {r.noise_reduction_pct?.toFixed(2)}%</td>
                    <td><Badge level={r.baseline_risk} /></td>
                    <td><Badge level={r.adv_risk} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Trade-off Curves ──────────────── */}
        {(advSweep.length > 0 || noiseSweep.length > 0) && (
          <div className="grid-2" style={{ marginBottom: 20 }}>
            <div className="card">
              <div className="card-header">
                <div className="card-title">Privacy–Utility Pareto Frontier</div>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>Adversarial λ-sweep · {advSweep.length} points</span>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={230}>
                  <LineChart data={advSweep} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis
                      dataKey="mean_leakage_score" type="number" domain={['auto', 'auto']}
                      tick={{ fontSize: 10, fill: '#94a3b8' }}
                      label={{ value: 'Leakage ↓ better', position: 'insideBottomRight', offset: -5, fill: '#94a3b8', fontSize: 10 }}
                    />
                    <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }}
                      label={{ value: 'Identity Utility ↑', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 10 }}
                    />
                    <Tooltip content={<SweepTooltip />} />
                    <Line type="monotone" dataKey="identity_utility" stroke="#2563eb" strokeWidth={2.5}
                      dot={{ r: 5, fill: '#2563eb', strokeWidth: 0 }} name="Utility vs λ" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-title">Noise σ-Sweep Trade-off</div>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>Gaussian noise analysis · {noiseSweep.length} points</span>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={230}>
                  <LineChart data={noiseSweep} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis
                      dataKey="mean_leakage_score" type="number" domain={['auto', 'auto']}
                      tick={{ fontSize: 10, fill: '#94a3b8' }}
                      label={{ value: 'Leakage ↓ better', position: 'insideBottomRight', offset: -5, fill: '#94a3b8', fontSize: 10 }}
                    />
                    <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} />
                    <Tooltip content={<SweepTooltip />} />
                    <Line type="monotone" dataKey="identity_utility" stroke="#7c3aed" strokeWidth={2.5}
                      dot={{ r: 5, fill: '#7c3aed', strokeWidth: 0 }} name="Utility vs σ" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        <div className="card card-padded" style={{ background: '#f5f3ff', borderColor: '#ddd6fe' }}>
          <div className="section-label" style={{ color: '#6d28d9' }}>Methodology References</div>
          <div style={{ fontSize: 12.5, color: '#475569', lineHeight: 1.9 }}>
            🔸 Ganin &amp; Lempitsky (2015) <em>Unsupervised Domain Adaptation by Backpropagation</em>, ICML — Gradient Reversal Layer<br />
            🔸 Morales et al. (2020) <em>SensitiveNets</em>, IEEE TPAMI — Adversarial Attribute Suppression in Face Embeddings<br />
            🔸 Mirjalili et al. (2020) <em>PrivacyNet</em>, IEEE TIP — Semi-Adversarial Networks for Attribute Obfuscation<br />
            🔸 Dwork et al. (2014) <em>Algorithmic Foundations of Differential Privacy</em> — Noise Injection Theoretical Framework
          </div>
        </div>
      </div>
    </div>
  );
}
