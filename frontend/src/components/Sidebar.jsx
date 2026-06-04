import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  MdShield, MdSearch, MdTune, MdAssessment, MdImage, MdCircle, MdHome
} from 'react-icons/md';

const NAV = [
  { to: '/',               icon: <MdHome />,       label: 'Overview Dashboard',    end: true },
  { to: '/audit',          icon: <MdSearch />,     label: 'Module 1: Auditing' },
  { to: '/mitigation',     icon: <MdTune />,       label: 'Module 2: Mitigation' },
  { to: '/report',         icon: <MdAssessment />, label: 'Module 3: Reporting' },
  { to: '/visualizations', icon: <MdImage />,      label: 'Visualizations' },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">🔐</div>
        <div>
          <div className="sidebar-logo-title">ObscuraAI</div>
          <div className="sidebar-logo-sub">Privacy Intelligence v2.1</div>
        </div>
      </div>

      <div className="sidebar-section-label">Navigation</div>
      <nav className="sidebar-nav">
        {NAV.map(({ to, icon, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <span className="nav-icon">{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div style={{ fontWeight: 700, color: '#334155', fontSize: 12 }}>ObscuraAI</div>
        <div style={{ marginTop: 2 }}>IEEE TPAMI · ICCV · TIFS</div>
        <div style={{ marginTop: 2, fontSize: 10 }}>GRL · MLP · SHA-256</div>
        <div className="sidebar-footer-badge">
          <MdCircle style={{ fontSize: 7 }} /> API Live
        </div>
      </div>
    </aside>
  );
}
