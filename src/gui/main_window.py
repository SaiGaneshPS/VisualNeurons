from PyQt6.QtWidgets import (
    QMainWindow, QToolBar, QStatusBar, QFileDialog,
    QMessageBox, QSplitter, QHBoxLayout, QWidget,
    QDockWidget
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QAction

from .sidebar import Sidebar
from .workspace import Workspace

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ML Visual App")
        self.setMinimumSize(1200, 800)
        
        # Create central widget with horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create and add sidebar
        self.sidebar = Sidebar(self)
        # IMPORTANT: Connect the signal to handle adding nodes
        self.sidebar.add_node_requested.connect(self.add_node_to_workspace)
        
        # Create and add workspace
        self.workspace = Workspace(self)
        
        # Create splitter for resizable sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.workspace)
        splitter.setStretchFactor(0, 0)  # Sidebar doesn't stretch
        splitter.setStretchFactor(1, 1)  # Workspace stretches
        splitter.setSizes([250, 950])  # Initial sizes
        
        layout.addWidget(splitter)
        
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
    
    # IMPORTANT: Add this method to handle adding nodes to the workspace
    def add_node_to_workspace(self, node_type):
        """Add a new node to the workspace."""
        print(f"Adding node type: {node_type}")  # Debug print
        self.workspace.add_node(node_type)
        self.statusBar().showMessage(f"Added {node_type} node to workspace", 3000)

    def _create_menus(self):
        # Create menu bar
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        new_action = QAction("&New Project", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Project", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Project", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save Project &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        import_data_action = QAction("&Import Dataset", self)
        file_menu.addAction(import_data_action)
        
        export_results_action = QAction("&Export Results", self)
        file_menu.addAction(export_results_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        
        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        # Main toolbar
        main_toolbar = QToolBar("Main Toolbar")
        main_toolbar.setIconSize(QSize(24, 24))
        main_toolbar.setMovable(False)
        self.addToolBar(main_toolbar)
        
        # Add tools to toolbar
        new_project_action = QAction("New", self)
        new_project_action.setToolTip("New Project")
        main_toolbar.addAction(new_project_action)
        
        open_project_action = QAction("Open", self)
        open_project_action.setToolTip("Open Project")
        main_toolbar.addAction(open_project_action)
        
        save_project_action = QAction("Save", self)
        save_project_action.setToolTip("Save Project")
        main_toolbar.addAction(save_project_action)
        
        main_toolbar.addSeparator()
        
        import_data_action = QAction("Import", self)
        import_data_action.setToolTip("Import Dataset")
        main_toolbar.addAction(import_data_action)
        
        main_toolbar.addSeparator()
        
        run_action = QAction("Run", self)
        run_action.setToolTip("Run Pipeline")
        main_toolbar.addAction(run_action)
    
    def _create_statusbar(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("Ready")