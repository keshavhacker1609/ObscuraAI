import React from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip
} from 'recharts';

export default function LeakageRadar({ auditData = {} }) {
  const attrs = ['gender', 'age_group', 'ethnicity'];
  const labels = { gender: 'Gender\nLR', age_group: 'Age\nLR', ethnicity: 'Race\nLR' };

  const data = [];
  attrs.forEach(attr => {
    if (!auditData[attr]) return;
    data.push({
      metric: attr.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()) + ' LR',
      value:  Math.round((auditData[attr]?.logistic_regression?.auc_roc || 0.5) * 100),
    });
    data.push({
      metric: attr.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()) + ' MLP',
      value:  Math.round((auditData[attr]?.mlp?.auc_roc || 0.5) * 100),
    });
  });

  if (!data.length) return (
    <div className="empty-state" style={{ padding: 30 }}>
      <div className="empty-state-icon">🕷️</div>
      <p>No audit data yet.</p>
    </div>
  );

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart outerRadius={110} data={data}>
        <PolarGrid stroke="rgba(255,255,255,0.08)" />
        <PolarAngleAxis
          dataKey="metric"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
        />
        <PolarRadiusAxis
          angle={30}
          domain={[40, 100]}
          tick={{ fill: '#475569', fontSize: 9 }}
        />
        <Radar
          name="AUC (%)"
          dataKey="value"
          stroke="#6366f1"
          fill="#6366f1"
          fillOpacity={0.2}
          strokeWidth={2}
        />
        <Tooltip
          contentStyle={{
            background: '#1e293b',
            border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: 12,
            color: '#f8fafc',
            fontSize: 12,
          }}
          formatter={(v) => [`${v}%`, 'AUC']}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
