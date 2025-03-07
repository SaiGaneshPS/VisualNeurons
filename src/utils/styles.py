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
    
    QSpinBox, QDoubleSpinBox {
        background-color: white;
        border: 1px solid #cccccc;
        padding: 2px 18px 2px 4px;  /* Space for buttons */
        min-width: 60px;
    }
    
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 16px;
        border-left: 1px solid #cccccc;
        border-bottom: 1px solid #cccccc;
        background: #f0f0f0;
    }
    
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 16px;
        border-left: 1px solid #cccccc;
        background: #f0f0f0;
    }
    
    QComboBox {
        border: 1px solid #cccccc;
        border-radius: 3px;
        padding: 1px 18px 1px 3px;
        min-width: 100px;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: url(resources/icons/dropdown.png);  /* You'll need to add this icon */
        width: 12px;
        height: 12px;
    }
    
    QComboBox QAbstractItemView {
        border: 1px solid #cccccc;
        selection-background-color: #e3f2fd;
    }
    """
    app.setStyleSheet(style)