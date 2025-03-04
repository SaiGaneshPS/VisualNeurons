from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QMenu, QDialog,
    QSpinBox, QLabel, QFormLayout, QPushButton, QColorDialog, QHBoxLayout,
    QCheckBox, QGraphicsItem, QGraphicsProxyWidget
)
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QLineF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QAction, QPalette

# Import data nodes
from .nodes.data_nodes import FileDataNode, SampleDatasetNode

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
        
        # Enable rubberband selection
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        # List to keep track of nodes
        self.nodes = []
    
    def mousePressEvent(self, event):
        """Override mouse press event for improved interaction."""
        # Check if we clicked on a node or its widgets
        item = self.itemAt(event.pos())

        # For all cases, use default handling
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double clicks to activate buttons within nodes."""
        # Pass to parent handler
        super().mouseDoubleClickEvent(event)
    
    def add_node(self, node_type):
        """Add a new node to the workspace."""
        # Calculate position for the new node - center of current view
        view_center = self.mapToScene(self.viewport().rect().center())
        
        node = None
        
        # Create the appropriate node type
        if node_type == "csv_excel":
            node = FileDataNode(view_center.x() - 125, view_center.y() - 100)
        elif node_type == "sample_dataset":
            node = SampleDatasetNode(view_center.x() - 125, view_center.y() - 100)
        
        # Add node to scene if created
        if node:
            self.scene.addItem(node)
            self.nodes.append(node)
    
    def wheelEvent(self, event):
        """Override wheel event to implement zooming."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        # Save the scene position under the cursor
        old_pos = self.mapToScene(event.position().toPoint())
        
        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        
        self.scale(zoom_factor, zoom_factor)
        
        # Get the new position under the cursor
        new_pos = self.mapToScene(event.position().toPoint())
        
        # Move scene to old position
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
    
    def contextMenuEvent(self, event):
        """Show context menu on right click."""
        context_menu = QMenu(self)
        
        # Grid settings action
        grid_settings_action = QAction("Grid Settings...", self)
        grid_settings_action.triggered.connect(self.show_grid_settings)
        context_menu.addAction(grid_settings_action)
        
        # Show the context menu
        context_menu.exec(event.globalPosition().toPoint())
    
    def show_grid_settings(self):
        """Show dialog to configure grid settings."""
        dialog = GridSettingsDialog(self.scene.grid_settings, self)
        if dialog.exec():
            # If dialog is accepted, update the scene
            self.scene.update()

class Workspace(QWidget):
    """The main workspace area where users can build ML workflows."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = WorkspaceView(self)
        layout.addWidget(self.view)
        
        # Set background color for the workspace
        self.setStyleSheet("background-color: white;")
    
    def add_node(self, node_type):
        """Add a new node to the workspace."""
        self.view.add_node(node_type)