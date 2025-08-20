import React from 'react';
import './Header.css';

const Header = ({ onToggleSidebar }) => {
  return (
    <header className="header">
      <div className="header-left">
        <button className="toggle-sidebar-btn" onClick={onToggleSidebar} aria-label="Toggle sidebar">
          <span className="icon">≡</span>
        </button>
        <div className="notification-banner">
          <span className="icon">i</span>
          <span>Team-project-frontend</span>
        </div>
      </div>
      
      <div className="header-right">
        <div className="user-profile">
          <span className="icon">U</span>
          <div className="user-info">
            <span className="user-name">사용자</span>
            <span className="user-role">일반 사용자</span>
          </div>
        </div>        
        <div className="header-actions">
        </div>
      </div>
    </header>
  );
};

export default Header;
