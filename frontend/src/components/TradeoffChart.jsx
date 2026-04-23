import React from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Line, ComposedChart, Area
} from 'recharts';

const CustomTooltip = ({ active, payload, xKey, yKey }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="custom-tooltip">
      <div className="custom-tooltip-label">{xKey} = {d?.[xKey]?.toFixed(3)}</div>
      <div className="custom-tooltip-value">Utility: {d?.[yKey]?.toFixed(4)}</div>
    </div>
  );
};

export default function TradeoffChart({ data = [], xKey, yKey, color = '#6366f1', label }) {
  if (!data.length) return (
    <div className="empty-state" style={{ padding: 30 }}>
      <div className="empty-state-icon">📊</div>
      <p>No sweep data yet. Run the pipeline with sweep enabled.</p>
    </div>
  );

  const chartData = data.map(d => ({
    ...d,
    [xKey]: parseFloat(d[xKey]?.toFixed ? d[xKey].toFixed(4) : d[xKey]),
    [yKey]: parseFloat(d[yKey]?.toFixed ? d[yKey].toFixed(4) : d[yKey]),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis
          dataKey={xKey}
          name={xKey}
          label={{ value: `${xKey} →`, position: 'insideBottomRight', offset: -10, fill: '#475569', fontSize: 11 }}
          tick={{ fill: '#64748b', fontSize: 11 }}
        />
        <YAxis
          dataKey={yKey}
          domain={['auto', 'auto']}
          label={{ value: 'Identity Utility', angle: -90, position: 'insideLeft', fill: '#475569', fontSize: 11 }}
          tick={{ fill: '#64748b', fontSize: 11 }}
        />
        <Tooltip content={<CustomTooltip xKey={xKey} yKey={yKey} />} />
        <Area type="monotone" dataKey={yKey} fill={`${color}12`} stroke="none" />
        <Line
          type="monotone"
          dataKey={yKey}
          stroke={color}
          strokeWidth={2.5}
          dot={{ r: 5, fill: color, strokeWidth: 0 }}
          activeDot={{ r: 7, fill: color }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
