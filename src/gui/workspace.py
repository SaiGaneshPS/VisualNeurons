from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QMenu, QDialog,
    QSpinBox, QLabel, QFormLayout, QPushButton, QColorDialog, QHBoxLayout,
    QCheckBox, QGraphicsItem, QGraphicsProxyWidget, QGraphicsRectItem, QMessageBox
)
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QLineF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QAction, QPalette

# Import data nodes
from .nodes.data_nodes import FileDataNode, SampleDatasetNode
from .nodes.preprocessing_nodes import FeatureSelectionNode, MissingValuesNode, NormalizationNode, EncodingNode
from .nodes.preprocessing_nodes import (
    FeatureSelectionNode, MissingValuesNode, NormalizationNode, EncodingNode,
    TrainTestSplitNode, DataTypeNode, DimensionalityReductionNode
)
from .nodes.model_nodes import (
    LogisticRegressionNode, DecisionTreeNode, RandomForestNode,
    SVMNode, NaiveBayesNode, KNNNode
)
from .nodes.evaluation_nodes import (
    MetricsNode, ConfusionMatrixNode, ROCCurveNode
)


class GridSettings:
    """Class to store grid settings."""
    def __init__(self):
        self.size = 20
        self.color = QColor("#E0E8FF")  # Light blue grid lines
        self.enabled = True

class GridGraphicsScene(QGraphicsScene):
    """A custom graphics scene that draws a grid in the background."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_settings = GridSettings()
    
    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        
        # Only draw grid if enabled
        if not self.grid_settings.enabled:
            return
        
        # Draw the grid
        left = int(rect.left()) - (int(rect.left()) % self.grid_settings.size)
        top = int(rect.top()) - (int(rect.top()) % self.grid_settings.size)
        
        # Create a pen for the grid lines
        painter.setPen(QPen(self.grid_settings.color, 1))
        
        # Draw vertical grid lines
        x = left
        while x <= rect.right():
            line = QLineF(x, rect.top(), x, rect.bottom())
            painter.drawLine(line)
            x += self.grid_settings.size
        
        # Draw horizontal grid lines
        y = top
        while y <= rect.bottom():
            line = QLineF(rect.left(), y, rect.right(), y)
            painter.drawLine(line)
            y += self.grid_settings.size

class GridSettingsDialog(QDialog):
    """Dialog for configuring grid settings."""
    
    def __init__(self, grid_settings, parent=None):
        super().__init__(parent)
        self.grid_settings = grid_settings
        
        self.setWindowTitle("Grid Settings")
        self.resize(250, 150)
        
        # Create layout
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # Grid enabled checkbox
        self.enabled_checkbox = QCheckBox("Grid Enabled")
        self.enabled_checkbox.setChecked(self.grid_settings.enabled)
        form_layout.addRow(self.enabled_checkbox)
        
        # Grid size spinner
        self.size_spinner = QSpinBox()
        self.size_spinner.setRange(5, 100)
        self.size_spinner.setValue(self.grid_settings.size)
        form_layout.addRow("Grid Size:", self.size_spinner)
        
        # Grid color button
        self.color_button = QPushButton()
        self.color_button.setAutoFillBackground(True)
        self._update_color_button()
        self.color_button.clicked.connect(self._select_color)
        form_layout.addRow("Grid Color:", self.color_button)
        
        layout.addLayout(form_layout)
        
        # Add dialog buttons
        button_box = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_box.addStretch()
        button_box.addWidget(apply_button)
        button_box.addWidget(cancel_button)
        
        layout.addStretch()
        layout.addLayout(button_box)
    
    def _update_color_button(self):
        """Update the color button appearance to show the selected color."""
        palette = self.color_button.palette()
        palette.setColor(QPalette.ColorRole.Button, self.grid_settings.color)
        self.color_button.setPalette(palette)
        self.color_button.update()
    
    def _select_color(self):
        """Open color dialog to select grid color."""
        color = QColorDialog.getColor(self.grid_settings.color, self)
        if color.isValid():
            self.grid_settings.color = color
            self._update_color_button()
    
    def accept(self):
        """Apply the settings when OK is clicked."""
        self.grid_settings.size = self.size_spinner.value()
        self.grid_settings.enabled = self.enabled_checkbox.isChecked()
        super().accept()

class Connection:
    """Represents a connection between two nodes."""
    def __init__(self, source_connector, target_connector, color=None):
        self.source_connector = source_connector
        self.target_connector = target_connector
        self.color = color or QColor("#1976D2")  # Default blue
        self.line = None
        self.update_position()  # Add initial position update

    def update_position(self):
        """Update the line position based on connector positions."""
        if self.line:
            source_pos = self.source_connector.scenePos() + self.source_connector.rect().center()
            target_pos = self.target_connector.scenePos() + self.target_connector.rect().center()
            self.line.setLine(source_pos.x(), source_pos.y(), target_pos.x(), target_pos.y())

class WorkspaceView(QGraphicsView):
    """A custom graphics view for the ML workflow workspace."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up the scene
        self.scene = GridGraphicsScene(self)
        self.setScene(self.scene)
        
        # Set rendering hints for better quality
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Set scroll bar policies
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Set scene rect (can be adjusted based on workflow size)
        self.scene.setSceneRect(-2000, -1500, 4000, 3000)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Enable rubberband selection ONLY when not creating connections
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        # Lists to keep track of nodes and connections
        self.nodes = []
        self.connections = []
        self.temp_connection = None
        self.source_connector = None
        
        # Connection colors
        self.connection_colors = [
            QColor("#1976D2"),  # Blue
            QColor("#388E3C"),  # Green
            QColor("#D32F2F"),  # Red
            QColor("#7B1FA2"),  # Purple
            QColor("#FFA000"),  # Orange
            QColor("#00796B"),  # Teal
        ]
        self.current_color_index = 0
        
        # Flag to track if we're creating a connection
        self.creating_connection = False
        self.node_counter = {}  # Add counter for node names
    
    def mousePressEvent(self, event):
        """Handle mouse press events for connection creation."""
        item = self.itemAt(event.pos())
        
        # Check if we clicked on a connector
        if isinstance(item, QGraphicsRectItem) and hasattr(item, 'is_connector'):
            # Check if trying to start from input connector
            if hasattr(item, 'is_input') and item.is_input:
                event.accept()
                return
                
            # Get the parent node when starting connection
            source_node = item.parentItem()
            
            if self.source_connector is None:
                # Start connection
                if hasattr(item, 'is_output') and item.is_output:
                    self.source_connector = item
                    self.source_node = source_node  # Store source node
                    color = self.connection_colors[self.current_color_index]
                    self.source_connector.setBrush(QBrush(color))
                    
                    # Create temporary line for visual feedback
                    start_pos = self.source_connector.scenePos() + self.source_connector.rect().center()
                    self.temp_connection = self.scene.addLine(
                        QLineF(start_pos, start_pos), 
                        QPen(color, 2)
                    )
                    
                    # Consume the event to prevent rubber band selection
                    event.accept()
                    return
            else:
                # Complete connection
                if hasattr(item, 'is_input') and item.is_input:
                    # Create permanent connection
                    color = self.connection_colors[self.current_color_index]
                    self.create_connection(self.source_connector, item, color)
                    self.current_color_index = (self.current_color_index + 1) % len(self.connection_colors)
                
                # Reset temporary connection
                if self.temp_connection:
                    self.scene.removeItem(self.temp_connection)
                    self.temp_connection = None
                
                self.source_connector = None
                
                # Consume the event to prevent rubber band selection
                event.accept()
                return
        
        # Re-enable rubber band selection for normal interactions
        if not self.creating_connection:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        # Pass event to parent for regular handling
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Update temporary connection line while dragging."""
        if self.temp_connection and self.source_connector:
            start_pos = self.source_connector.scenePos() + self.source_connector.rect().center()
            end_pos = self.mapToScene(event.pos())
            self.temp_connection.setLine(QLineF(start_pos, end_pos))
            event.accept()
            return
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Clean up temporary connection if needed."""
        if self.temp_connection and self.source_connector:
            # Find the nearest input connector
            target_connector = self.find_nearest_input_connector(event.pos())
            
            if target_connector:
                # Get parent nodes
                source_node = self.source_connector.parentItem()
                target_node = target_connector.parentItem()
                
                # Check if it's a self-connection
                if source_node == target_node:
                    QMessageBox.warning(None, "Invalid Connection", "Cannot connect a node to itself!")
                else:
                    color = self.connection_colors[self.current_color_index]
                    self.create_connection(self.source_connector, target_connector, color)
                    self.current_color_index = (self.current_color_index + 1) % len(self.connection_colors)
            else:
                # No valid target found, reset source connector color
                self.source_connector.setBrush(QBrush(QColor("#dae8fc")))
                print("No valid input connector found")
            
            # Clean up temporary connection
            self.scene.removeItem(self.temp_connection)
            self.temp_connection = None
            self.source_connector = None
            
            event.accept()
        
        # Reset connection creation flag and restore rubber band selection
        self.creating_connection = False
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        super().mouseReleaseEvent(event)

    def find_nearest_input_connector(self, pos, search_radius=20):
        """Find the nearest input connector within search radius."""
        scene_pos = self.mapToScene(pos)
        
        # Create a search area around the mouse position
        search_rect = QRectF(
            scene_pos.x() - search_radius,
            scene_pos.y() - search_radius,
            search_radius * 2,
            search_radius * 2
        )
        
        # Get all items in the search area
        items = self.scene.items(search_rect)
        
        # Filter for input connectors
        input_connectors = [
            item for item in items
            if isinstance(item, QGraphicsRectItem) and 
            hasattr(item, 'is_connector') and 
            hasattr(item, 'is_input') and 
            item.is_input
        ]
        
        if not input_connectors:
            return None
        
        # Find the closest one
        closest = min(input_connectors, key=lambda i: 
                    (i.scenePos().x() + i.rect().center().x() - scene_pos.x())**2 +
                    (i.scenePos().y() + i.rect().center().y() - scene_pos.y())**2)
        
        return closest
    
    def create_connection(self, source_connector, target_connector, color):
        """Create a permanent connection between two connectors."""
        # Create the connection line
        start_pos = source_connector.scenePos() + source_connector.rect().center()
        end_pos = target_connector.scenePos() + target_connector.rect().center()
        
        # Use path for better visual appearance
        line = self.scene.addLine(
            QLineF(start_pos, end_pos), 
            QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        )
        
        # Color the connectors
        source_connector.setBrush(QBrush(color))
        target_connector.setBrush(QBrush(color))
        
        # Store the connection
        connection = Connection(source_connector, target_connector, color)
        connection.line = line
        self.connections.append(connection)
        
        # Get the nodes
        source_node = source_connector.parentItem()
        target_node = target_connector.parentItem()
        
        # Find the input name for the target connector
        input_name = None
        for name, input_info in target_node.inputs.items():
            if input_info["connector"] == target_connector:
                input_name = name
                break
        
        # Find the output name for the source connector
        output_name = None
        for name, output_info in source_node.outputs.items():
            if output_info["connector"] == source_connector:
                output_name = name
                break
        
        # Connect the nodes - get output data and set as input
        if input_name and output_name:
            output_data = source_node.get_output_data(output_name)
            if output_data:
                target_node.set_input_data(input_name, output_data)
                print(f"Connected {source_node.name}.{output_name} to {target_node.name}.{input_name}")
            else:
                print(f"Warning: No output data available from {source_node.name}.{output_name}")
        else:
            print("Warning: Could not determine input/output names for connection")

    def add_node(self, node_type):
        """Add a new node to the workspace."""
        view_center = self.mapToScene(self.viewport().rect().center())
        node = None
        
        # Update counter for this node type
        self.node_counter[node_type] = self.node_counter.get(node_type, 0) + 1
        node_name = f"{node_type}_{self.node_counter[node_type]}"

        # Create the appropriate node type
        if node_type == "csv_excel":
            node = FileDataNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "sample_dataset":
            node = SampleDatasetNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "feature_selection":
            node = FeatureSelectionNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "missing_values":
            node = MissingValuesNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "normalization":
            node = NormalizationNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "encoding":
            node = EncodingNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "train_test_split":
            node = TrainTestSplitNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "data_type":
            node = DataTypeNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "dim_reduction":
            node = DimensionalityReductionNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "logistic_regression":
            node = LogisticRegressionNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "decision_tree":
            node = DecisionTreeNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "random_forest":
            node = RandomForestNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "svm":
            node = SVMNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "naive_bayes":
            node = NaiveBayesNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "knn":
            node = KNNNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "metrics":
            node = MetricsNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "confusion_matrix":
            node = ConfusionMatrixNode(view_center.x() - 125, view_center.y() - 100, node_name)
        elif node_type == "roc_curve":
            node = ROCCurveNode(view_center.x() - 125, view_center.y() - 100, node_name)

        if node:
            self.scene.addItem(node)
            self.nodes.append(node)
            
            # Connect the node's position change signal
            node.positionChanged.connect(self.update_connections)

    def update_connections(self):
        """Update all connection positions when nodes move."""
        for connection in self.connections:
            connection.update_position()

class Workspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = WorkspaceView(self)
        layout.addWidget(self.view)
        
        self.setStyleSheet("background-color: white;")
    
    def add_node(self, node_type):
        """Add a new node to the workspace."""
        self.view.add_node(node_type)