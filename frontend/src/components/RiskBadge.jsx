import React from 'react';

const RISK_MAP = {
  CRITICAL:   'risk-critical',
  HIGH:       'risk-high',
  MEDIUM:     'risk-medium',
  LOW:        'risk-low',
  NEGLIGIBLE: 'risk-negligible',
  UNKNOWN:    'risk-unknown',
};

const DOT_MAP = {
  CRITICAL: '🔴',
  HIGH:     '🟠',
  MEDIUM:   '🟡',
  LOW:      '🟢',
  NEGLIGIBLE:'⚪',
  UNKNOWN:  '⚫',
};

export default function RiskBadge({ level }) {
  const cls  = RISK_MAP[level] || 'risk-unknown';
  const dot  = DOT_MAP[level]  || '⚫';
  return (
    <span className={`risk-badge ${cls}`}>
      {dot} {level || 'UNKNOWN'}
    </span>
  );
}
