import React from 'react';

const MAP = {
  CRITICAL:    { cls: 'badge-red',   dot: '●', label: 'CRITICAL' },
  HIGH:        { cls: 'badge-amber', dot: '●', label: 'HIGH' },
  MEDIUM:      { cls: 'badge-blue',  dot: '●', label: 'MEDIUM' },
  LOW:         { cls: 'badge-green', dot: '●', label: 'LOW' },
  NEGLIGIBLE:  { cls: 'badge-slate', dot: '●', label: 'NEGLIGIBLE' },
  PASS:        { cls: 'badge-pass',  dot: '✓', label: 'PASS' },
  FAIL:        { cls: 'badge-fail',  dot: '✗', label: 'FAIL' },
  WARNING:     { cls: 'badge-warn',  dot: '⚠', label: 'WARNING' },
};

export default function Badge({ level }) {
  const m = MAP[level?.toUpperCase?.()] || { cls: 'badge-slate', dot: '●', label: level || 'N/A' };
  return <span className={`badge ${m.cls}`}>{m.dot} {m.label}</span>;
}
