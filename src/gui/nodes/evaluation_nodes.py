from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFormLayout, QDialog, 
    QPushButton, QGraphicsRectItem, QTableWidget,
    QTableWidgetItem, QHBoxLayout, QDialogButtonBox, QGraphicsTextItem, QToolTip
)
from PyQt6.QtGui import QPen, QBrush, QColor, QFont
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .data_nodes import DataNode, DataNodeWidget
from ..components.combo_box import NavigableComboBox

class EvaluationNode(DataNode):
    """Base class for evaluation nodes."""
    
    def __init__(self, x=0, y=0, width=220, height=200, name=None):
        self.inputs = {
            "model": {
                "type": "DataFrame",
                "description": "Model predictions",
                "required": True
            },
            "data": {
                "type": "DataFrame",
                "description": "Data for evaluation (train/test/validation)",
                "required": True
            }
        }
        
        super().__init__(x, y, width, height, name)
        
        # Override styling for evaluation nodes
        self.setPen(QPen(QColor("#9C27B0"), 2))  # Purple border
        self.setBrush(QBrush(QColor("#F3E5F5")))  # Light purple background
        
        # Create input connectors
        self._setup_input_connectors()
    
    def _setup_input_connectors(self):
        """Set up input connectors for model predictions and data."""
        rect = self.rect()
        spacing = rect.width() / 3  # Divide width into 3 parts for even spacing
        
        # Model predictions connector (left)
        self.model_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.model_connector.setPos(spacing - 10, -10)
        self.model_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.model_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.model_connector.is_connector = True
        self.model_connector.is_input = True
        
        # Add permanent text for model connector
        self.model_text = QGraphicsTextItem(self)
        self.model_text.setPlainText("Input: model predictions")
        self.model_text.setDefaultTextColor(QColor("black"))
        self.model_text.setFont(QFont("Arial", 8))
        text_width = self.model_text.boundingRect().width()
        self.model_text.setPos(
            spacing - text_width/2,  # Center horizontally
            -30  # Above connector
        )
        
        # Data connector (right)
        self.data_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.data_connector.setPos(2 * spacing - 10, -10)
        self.data_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.data_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.data_connector.is_connector = True
        self.data_connector.is_input = True
        
        # Add permanent text for data connector
        self.data_text = QGraphicsTextItem(self)
        self.data_text.setPlainText("Input: ground truth data")
        self.data_text.setDefaultTextColor(QColor("black"))
        self.data_text.setFont(QFont("Arial", 8))
        text_width = self.data_text.boundingRect().width()
        self.data_text.setPos(
            2 * spacing - text_width/2,  # Center horizontally
            -30  # Above connector
        )
        
        # Add to input connectors dictionary
        self.input_connectors["model"] = self.model_connector
        self.input_connectors["data"] = self.data_connector
        
        # Add labels
        self._add_input_labels()
    
    def _add_input_labels(self):
        """Add labels for each input connector."""
        rect = self.rect()
        spacing = rect.width() / 3
        
        # Model label
        self.model_label = QGraphicsTextItem(self)
        self.model_label.setPlainText("model")
        self.model_label.setFont(QFont("Arial", 8))
        label_width = self.model_label.boundingRect().width()
        self.model_label.setPos(spacing - label_width/2, -25)
        
        # Data label
        self.data_label = QGraphicsTextItem(self)
        self.data_label.setPlainText("data")
        self.data_label.setFont(QFont("Arial", 8))
        label_width = self.data_label.boundingRect().width()
        self.data_label.setPos(2 * spacing - label_width/2, -25)

class EvaluationWidget(QWidget):
    """Base widget for evaluation nodes."""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFixedHeight(32)  # Fixed height for consistent title size
        self.title_label.setStyleSheet("""
            font-weight: bold;
            color: white;
            background-color: #9C27B0;
            padding: 4px;
            border-radius: 3px;
        """)
        self.layout.addWidget(self.title_label)
        
        # Set widget styling
        self.setStyleSheet("""
            QWidget {
                background-color: #F3E5F5;
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 9pt;
                background-color: transparent;
                padding: 2px;
            }
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:pressed {
                background-color: #6A1B9A;
            }
        """)

class MetricsWidget(EvaluationWidget):
    """Widget for displaying classification metrics."""
    
    # Add signal for average method changes
    average_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__("Classification Metrics", parent)
        
        # Main content layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        # Add averaging method selector
        self.avg_combo = NavigableComboBox(
            label="Average:",
            items=['macro', 'micro', 'weighted']
        )
        self.avg_combo.value_changed.connect(self._on_average_changed)
        content_layout.addWidget(self.avg_combo)
        
        # Add metrics display
        metrics_layout = QFormLayout()
        metrics_layout.setSpacing(6)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setHorizontalSpacing(10)
        
        self.accuracy_label = QLabel("-")
        self.precision_label = QLabel("-")
        self.recall_label = QLabel("-")
        self.f1_label = QLabel("-")
        
        # Style metric labels
        metric_style = "color: #333333; font-weight: bold; padding: 2px;"
        for label in [self.accuracy_label, self.precision_label, 
                     self.recall_label, self.f1_label]:
            label.setStyleSheet(metric_style)
            label.setMinimumWidth(60)
            label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        metrics_layout.addRow("Accuracy:", self.accuracy_label)
        metrics_layout.addRow("Precision:", self.precision_label)
        metrics_layout.addRow("Recall:", self.recall_label)
        metrics_layout.addRow("F1 Score:", self.f1_label)
        
        content_layout.addLayout(metrics_layout)
        self.layout.addLayout(content_layout)
        
        # Add stretch to push everything to the top
        self.layout.addStretch()
        
        # Set fixed height
        self.setFixedHeight(200)
        
        # Store data
        self.y_true = None
        self.y_pred = None
    
    def _on_average_changed(self, method):
        """Handle averaging method change."""
        self.average_changed.emit(method)  # Emit signal with new method
        if self.y_true is not None and self.y_pred is not None:
            self.update_metrics(self.y_true, self.y_pred)  # Update metrics directly
    
    def update_metrics(self, y_true, y_pred):
        """Update the displayed metrics."""
        if y_true is None or y_pred is None:
            return
        
        # Store the data
        self.y_true = y_true
        self.y_pred = y_pred
        
        average = self.avg_combo.currentText()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Update labels
        self.accuracy_label.setText(f"{accuracy:.3f}")
        self.precision_label.setText(f"{precision:.3f}")
        self.recall_label.setText(f"{recall:.3f}")
        self.f1_label.setText(f"{f1:.3f}")

class MetricsNode(EvaluationNode):
    """Node for displaying classification metrics."""
    
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 200, name)
        self.widget_content = MetricsWidget()
        self.widget_content.average_changed.connect(self._on_average_changed)
        self.setup_widget(self.widget_content)
        
        # Store predictions and ground truth
        self.y_pred = None
        self.y_true = None
    
    def _on_average_changed(self, method):
        """Handle average method changes."""
        if self.y_pred is not None and self.y_true is not None:
            self.widget_content.update_metrics(self.y_true, self.y_pred)

class ConfusionMatrixDialog(QDialog):
    """Dialog for displaying the confusion matrix."""
    
    def __init__(self, matrix, labels=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confusion Matrix")
        self.setMinimumSize(400, 400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create table
        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #CE93D8;
            }
            QTableWidget::item {
                padding: 5px;
                border: 1px solid #E1BEE7;
            }
            QHeaderView::section {
                background-color: #9C27B0;
                color: white;
                padding: 5px;
                border: 1px solid #7B1FA2;
            }
        """)
        
        # Set up table
        n_classes = len(matrix)
        self.table.setRowCount(n_classes)
        self.table.setColumnCount(n_classes)
        
        # Set headers
        if labels is None:
            labels = [f"Class {i}" for i in range(n_classes)]
        
        self.table.setHorizontalHeaderLabels(labels)
        self.table.setVerticalHeaderLabels(labels)
        
        # Fill table
        for i in range(n_classes):
            for j in range(n_classes):
                item = QTableWidgetItem(str(matrix[i, j]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if i == j:
                    item.setBackground(QBrush(QColor("#E1BEE7")))
                self.table.setItem(i, j, item)
        
        # Adjust column widths
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        
        layout.addWidget(self.table)
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

class ConfusionMatrixWidget(EvaluationWidget):
    """Widget for displaying confusion matrix."""
    
    def __init__(self, parent=None):
        super().__init__("Confusion Matrix", parent)
        
        # Info layout
        info_layout = QFormLayout()
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(8, 8, 8, 4)
        
        self.classes_label = QLabel("No data")
        self.accuracy_label = QLabel("-")
        
        info_style = "color: #333333; padding: 2px;"
        self.classes_label.setStyleSheet(info_style)
        self.accuracy_label.setStyleSheet(info_style)
        
        info_layout.addRow("Classes:", self.classes_label)
        info_layout.addRow("Accuracy:", self.accuracy_label)
        
        self.layout.addLayout(info_layout)
        
        # Add flexible space
        self.layout.addStretch()
        
        # Add show button
        self.show_button = QPushButton("Show Matrix")
        self.show_button.setFixedHeight(32)
        self.show_button.setEnabled(False)
        self.layout.addWidget(self.show_button)
        
        # Connect signals
        self.show_button.clicked.connect(self._show_matrix)
        
        # Store matrix data
        self.matrix = None
        self.labels = None
        
        # Set fixed size
        self.setFixedWidth(220)
        self.setFixedHeight(150)
    
    def update_matrix(self, y_true, y_pred):
        """Update the confusion matrix."""
        if y_true is None or y_pred is None:
            return
        
        self.matrix = confusion_matrix(y_true, y_pred)
        self.labels = sorted(list(set(y_true) | set(y_pred)))
        self.show_button.setEnabled(True)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Update info labels
        self.classes_label.setText(f"{len(self.labels)} classes")
        self.accuracy_label.setText(f"{accuracy:.3f}")
    
    def _show_matrix(self):
        """Show the confusion matrix dialog."""
        if self.matrix is not None:
            dialog = ConfusionMatrixDialog(self.matrix, self.labels, self)
            dialog.exec()

class ROCCurveDialog(QDialog):
    """Dialog for displaying ROC curves."""
    
    def __init__(self, y_true, y_score, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROC Curve")
        self.setMinimumSize(600, 400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate ROC curve and AUC for each class
        n_classes = y_score.shape[1]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        # Add plot to dialog
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

class ROCCurveWidget(EvaluationWidget):
    """Widget for displaying ROC curves."""
    
    def __init__(self, parent=None):
        super().__init__("ROC Curve", parent)
        
        # Info layout
        info_layout = QFormLayout()
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(8, 8, 8, 4)
        
        self.classes_label = QLabel("No data")
        self.auc_label = QLabel("-")
        
        info_style = "color: #333333; padding: 2px;"
        self.classes_label.setStyleSheet(info_style)
        self.auc_label.setStyleSheet(info_style)
        
        info_layout.addRow("Classes:", self.classes_label)
        info_layout.addRow("Mean AUC:", self.auc_label)
        
        self.layout.addLayout(info_layout)
        
        # Add flexible space
        self.layout.addStretch()
        
        # Add show button
        self.show_button = QPushButton("Show ROC Curve")
        self.show_button.setFixedHeight(32)
        self.show_button.setEnabled(False)
        self.layout.addWidget(self.show_button)
        
        # Connect signals
        self.show_button.clicked.connect(self._show_curve)
        
        # Store data
        self.y_true = None
        self.y_score = None
        
        # Set fixed size
        self.setFixedWidth(220)
        self.setFixedHeight(150)
    
    def update_data(self, y_true, y_score):
        """Update the ROC curve data."""
        if y_true is None or y_score is None:
            return
        
        self.y_true = y_true
        self.y_score = y_score
        self.show_button.setEnabled(True)
        
        # Calculate mean AUC
        n_classes = y_score.shape[1] if len(y_score.shape) > 1 else 1
        mean_auc = 0
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_score[:, i])
            mean_auc += auc(fpr, tpr)
        mean_auc /= n_classes
        
        # Update info labels
        self.classes_label.setText(f"{n_classes} classes")
        self.auc_label.setText(f"{mean_auc:.3f}")
    
    def _show_curve(self):
        """Show the ROC curve dialog."""
        if self.y_true is not None and self.y_score is not None:
            dialog = ROCCurveDialog(self.y_true, self.y_score, self)
            dialog.exec()

class ConfusionMatrixNode(EvaluationNode):
    """Node for displaying confusion matrix."""
    
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 150, name)
        self.widget_content = ConfusionMatrixWidget()
        self.setup_widget(self.widget_content)
        
        # Store predictions and ground truth
        self.y_pred = None
        self.y_true = None
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        if input_name == "model":
            self.y_pred = data_package["data"]
        elif input_name == "data":
            self.y_true = data_package["data"]
        
        if self.y_pred is not None and self.y_true is not None:
            self.widget_content.update_matrix(self.y_true, self.y_pred)

class ROCCurveNode(EvaluationNode):
    """Node for displaying ROC curves."""
    
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 150, name)
        self.widget_content = ROCCurveWidget()
        self.setup_widget(self.widget_content)
        
        # Store predictions and ground truth
        self.y_true = None
        self.y_score = None
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        if input_name == "model":
            self.y_score = data_package["data"]
        elif input_name == "data":
            self.y_true = data_package["data"]
        
        if self.y_true is not None and self.y_score is not None:
            self.widget_content.update_data(self.y_true, self.y_score)
