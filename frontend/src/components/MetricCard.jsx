import React from 'react';
import { motion } from 'framer-motion';

export default function MetricCard({
  icon, label, value, unit = '', sub, color = 'var(--primary)', delay = 0
}) {
  return (
    <motion.div
      className="metric-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: 'easeOut' }}
      style={{ '--metric-color': color, '--metric-bg': `${color}18`, '--metric-border': `${color}30` }}
    >
      <div className="metric-icon">{icon}</div>
      <div className="metric-value">
        {typeof value === 'number' ? value.toFixed(value < 10 ? 3 : 1) : (value ?? '—')}
        {unit && <span style={{ fontSize: '16px', fontWeight: 500, color: 'var(--text-muted)', marginLeft: 2 }}>{unit}</span>}
      </div>
      <div className="metric-label">{label}</div>
      {sub && <div className="metric-change">{sub}</div>}
    </motion.div>
  );
}
