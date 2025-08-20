import React from 'react';
import './Sidebar.css';

const Sidebar = ({ isOpen = true, currentView = 'analyze', onSelectView = () => {} }) => {
  return (
    <div className={`sidebar ${isOpen ? '' : 'collapsed'}`}>
      <div className="sidebar-header">
        <div className="logo">
          <div className="logo-icon">AI</div>
          <span className="logo-text">자기소개서 분석</span>
        </div>
      </div>
      
      <nav className="sidebar-nav">
        <ul className="nav-list">
          <li className={`nav-item ${currentView === 'analyze' ? 'active' : ''}`} onClick={() => onSelectView('analyze')}>
            <span className="icon">•</span>
            <span>자기소개서 분석</span>
          </li>
          <li className={`nav-item ${currentView === 'results' ? 'active' : ''}`} onClick={() => onSelectView('results')}>
            <span className="icon">•</span>
            <span>분석 결과</span>
          </li>
        </ul>
      </nav>
      
      <div className="sidebar-footer">
        <div className="version-info">
          <span>v1.0.0</span>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
