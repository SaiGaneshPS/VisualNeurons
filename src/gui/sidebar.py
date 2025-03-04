from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QScrollArea,
    QFrame, QLabel
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal

class CollapsibleSection(QWidget):
    """A collapsible section for the sidebar."""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        
        self.content_area = QWidget()
        self.content_area.setVisible(False)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.toggle_button)
        self.layout.addWidget(self.content_area)
        
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        
        self.toggle_button.toggled.connect(self._on_toggle)
        
    def _on_toggle(self, checked):
        self.content_area.setVisible(checked)
        
    def add_item(self, item):
        self.content_layout.addWidget(item)

class SidebarItem(QPushButton):
    """A clickable item in the sidebar."""
    
    # Add the node_requested signal
    node_requested = pyqtSignal(str)
    
    def __init__(self, text, node_type="", parent=None):
        super().__init__(text, parent)
        self.node_type = node_type
        self.setFixedHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._on_clicked)
    
    def _on_clicked(self):
        # Emit the node_requested signal with the node type
        if self.node_type:
            self.node_requested.emit(self.node_type)

class Sidebar(QScrollArea):
    """The sidebar containing collapsible sections for different ML components."""
    
    # Add signal to notify when a node should be added
    add_node_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(250)
        self.setMaximumWidth(400)
        
        # Main container widget
        container = QWidget()
        self.setWidget(container)
        
        # Main layout
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create sections
        self._create_data_section(layout)
        self._create_preprocessing_section(layout)
        self._create_models_section(layout)
        self._create_evaluation_section(layout)
        self._create_deployment_section(layout)
        
        # Add stretch to push all content to the top
        layout.addStretch()
    
    def _create_data_section(self, layout):
        section = CollapsibleSection("Data Sources")
        
        # Add items with node types
        csv_item = SidebarItem("Load CSV/Excel", "csv_excel")
        csv_item.node_requested.connect(self._on_node_requested)
        section.add_item(csv_item)
        
        image_item = SidebarItem("Load Images", "load_image")
        image_item.node_requested.connect(self._on_node_requested)
        section.add_item(image_item)
        
        sample_item = SidebarItem("Sample Datasets", "sample_dataset")
        sample_item.node_requested.connect(self._on_node_requested)
        section.add_item(sample_item)
        
        layout.addWidget(section)
    
    def _create_preprocessing_section(self, layout):
        section = CollapsibleSection("Data Preprocessing")
        
        # Add items
        section.add_item(SidebarItem("Feature Selection"))
        section.add_item(SidebarItem("Missing Values"))
        section.add_item(SidebarItem("Normalization"))
        section.add_item(SidebarItem("Encoding"))
        section.add_item(SidebarItem("Train/Test Split"))
        
        layout.addWidget(section)
    
    def _create_models_section(self, layout):
        section = CollapsibleSection("Models")
        
        # Classification models
        section.add_item(SidebarItem("Linear Regression"))
        section.add_item(SidebarItem("Logistic Regression"))
        section.add_item(SidebarItem("Decision Tree"))
        section.add_item(SidebarItem("Random Forest"))
        section.add_item(SidebarItem("Support Vector Machine"))
        section.add_item(SidebarItem("Neural Network"))
        
        layout.addWidget(section)
    
    def _create_evaluation_section(self, layout):
        section = CollapsibleSection("Evaluation")
        
        section.add_item(SidebarItem("Metrics"))
        section.add_item(SidebarItem("Cross Validation"))
        section.add_item(SidebarItem("Confusion Matrix"))
        section.add_item(SidebarItem("ROC Curve"))
        
        layout.addWidget(section)
    
    def _create_deployment_section(self, layout):
        section = CollapsibleSection("Deployment")
        
        section.add_item(SidebarItem("Export Model"))
        section.add_item(SidebarItem("API Generation"))
        
        layout.addWidget(section)
    
    def _on_node_requested(self, node_type):
        """Forward the node request to the main window."""
        self.add_node_requested.emit(node_type)