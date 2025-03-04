def apply_application_style(app):
    """Apply custom blue and white styling to the application."""
    style = """
    /* Global text color */
    * {
        color: #333333;
    }
    
    QMainWindow {
        background-color: white;
    }
    
    QMenuBar {
        background-color: #f0f8ff;
        border-bottom: 1px solid #dae8fc;
    }
    
    QMenuBar::item {
        background-color: transparent;
        padding: 6px 10px;
    }
    
    QMenuBar::item:selected {
        background-color: #1976D2;
        color: white;
    }
    
    QMenu {
        background-color: white;
        border: 1px solid #dae8fc;
    }
    
    QMenu::item:selected {
        background-color: #e6f2ff;
        color: #1976D2;
    }
    
    QToolBar {
        background-color: #f0f8ff;
        border-bottom: 1px solid #dae8fc;
        spacing: 8px;
        padding: 2px;
    }
    
    QToolButton {
        border: none;
        border-radius: 4px;
        padding: 4px;
    }
    
    QToolButton:hover {
        background-color: #e6f2ff;
    }
    
    QStatusBar {
        background-color: #f0f8ff;
        color: #333333;
    }
    
    QSplitter::handle {
        background-color: #dae8fc;
    }
    
    /* Custom workspace styles */
    Workspace {
        background-color: white;
    }
    
    /* Sidebar styles */
    Sidebar {
        background-color: #f0f8ff;
        border-right: 1px solid #dae8fc;
    }
    
    CollapsibleSection {
        background-color: transparent;
    }
    
    CollapsibleSection > QPushButton {
        text-align: left;
        padding: 8px;
        padding-left: 12px;
        background-color: #f0f8ff;
        border: none;
        border-bottom: 1px solid #dae8fc;
        font-weight: bold;
        color: #1976D2;
    }
    
    CollapsibleSection > QPushButton:hover {
        background-color: #e6f2ff;
    }
    
    CollapsibleSection > QWidget {
        background-color: white;
    }
    
    SidebarItem {
        padding: 8px;
        padding-left: 20px;
        background-color: white;
        border: none;
        border-bottom: 1px solid #f0f8ff;
        text-align: left;
        color: #333333;
    }
    
    SidebarItem:hover {
        background-color: #e6f2ff;
        color: #1976D2;
    }
    
    /* Dialog styling */
    QDialog {
        background-color: white;
    }
    
    QLabel {
        color: #333333;
    }
    
    QPushButton {
        background-color: #f0f8ff;
        border: 1px solid #dae8fc;
        border-radius: 4px;
        padding: 6px 12px;
        color: #1976D2;
    }
    
    QPushButton:hover {
        background-color: #e6f2ff;
    }
    
    QPushButton:pressed {
        background-color: #dae8fc;
    }
    
    QSpinBox, QCheckBox {
        color: #333333;
    }
    """
    app.setStyleSheet(style)