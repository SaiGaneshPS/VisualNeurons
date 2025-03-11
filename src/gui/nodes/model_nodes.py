from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFormLayout, QSpinBox, QDoubleSpinBox, QScrollArea, 
    QGraphicsRectItem, QPushButton, QGraphicsTextItem, QToolTip, QProgressBar
)
from PyQt6.QtGui import QPen, QBrush, QColor, QFont
from PyQt6.QtCore import Qt, pyqtSignal
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .data_nodes import DataNode, DataNodeWidget
from ..components.combo_box import NavigableComboBox

class ModelNode(DataNode):
    """Base class for model nodes."""
    
    def __init__(self, x=0, y=0, width=220, height=200, name=None):
        # Define inputs before parent initialization
        self.inputs = {
            "train": {
                "type": "DataFrame",
                "description": "Training dataset",
                "required": True,
                "connector": None
            },
            "validation": {
                "type": "DataFrame",
                "description": "Validation dataset",
                "required": False,
                "connector": None
            }
        }
        
        # Define outputs
        self.outputs = {
            "predictions": {
                "type": "DataFrame",
                "description": "Model predictions",
                "connector": None
            }
        }
        
        super().__init__(x, y, width, height, name)
        
        # Override styling for model nodes
        self.setPen(QPen(QColor("#E57373"), 2))  # Light red border
        self.setBrush(QBrush(QColor("#FFEBEE")))  # Very light red background
        
        # Create input connectors
        self._setup_input_connectors()
        
        # Create output connector
        self._setup_output_connectors()
        
        # Initialize model
        self.model = None
        self.trained = False
    
    def _setup_input_connectors(self):
        """Set up input connectors for train and validation data."""
        rect = self.rect()
        spacing = rect.width() / 3  # Divide width into 3 parts for even spacing
        
        # Training data connector (left)
        self.train_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.train_connector.setPos(spacing - 10, -10)
        self.train_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.train_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.train_connector.is_connector = True
        self.train_connector.is_input = True
        self.train_connector.setToolTip("Input: training data")
        
        # Add permanent text for training connector
        self.train_text = self._create_connector_text("Input: train")
        text_width = self.train_text.boundingRect().width()
        self.train_text.setPos(
            spacing - text_width/2,  # Center horizontally
            -35  # Above connector, increased spacing
        )
        
        # Validation data connector (right)
        self.val_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.val_connector.setPos(2 * spacing - 10, -10)
        self.val_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.val_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.val_connector.is_connector = True
        self.val_connector.is_input = True
        self.val_connector.setToolTip("Input: validation data")
        
        # Add permanent text for validation connector
        self.val_text = self._create_connector_text("Input: valid")
        text_width = self.val_text.boundingRect().width()
        self.val_text.setPos(
            2 * spacing - text_width/2,  # Center horizontally
            -35  # Above connector, increased spacing
        )
        
        # Store connectors in inputs dictionary
        self.inputs["train"]["connector"] = self.train_connector
        self.inputs["validation"]["connector"] = self.val_connector
        
        # Store in input_connectors for compatibility
        self.input_connectors["train"] = self.train_connector
        self.input_connectors["validation"] = self.val_connector
    
    def _setup_output_connectors(self):
        """Set up output connector for predictions."""
        rect = self.rect()
        
        # Create predictions output connector at bottom center
        self.predictions_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.predictions_connector.setPos(rect.width()/2 - 10, rect.height())
        self.predictions_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.predictions_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.predictions_connector.is_connector = True
        self.predictions_connector.is_input = False
        self.predictions_connector.is_output = True
        self.predictions_connector.setToolTip("Output: predictions")
        
        # Add text for predictions connector
        self.predictions_text = self._create_connector_text("Output: predictions")
        text_width = self.predictions_text.boundingRect().width()
        self.predictions_text.setPos(
            rect.width()/2 - text_width/2,
            rect.height() + 15
        )
        
        # Store in outputs dictionary
        self.outputs["predictions"]["connector"] = self.predictions_connector
        
        # Store in output_connector for compatibility
        self.output_connector = self.predictions_connector
    
    def get_output_data(self, output_name="predictions"):
        """Get the output data for the specified output port."""
        if output_name != "predictions" or not self.trained or not self.model:
            return None
        
        # If we have validation data, use that for predictions
        if self.val_data is not None:
            data = self.val_data["data"]
            target_column = self.val_data["target_column"]
        # Otherwise use training data
        elif self.train_data is not None:
            data = self.train_data["data"]
            target_column = self.train_data["target_column"]
        else:
            return None
        
        try:
            X = data.drop(columns=[target_column])
            predictions = self.model.predict(X)
            
            return {
                "data": predictions,
                "target_column": target_column,
                "metadata": {
                    "type": "predictions",
                    "model_type": self.__class__.__name__,
                    "samples": len(predictions)
                }
            }
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            return None

class ModelWidget(DataNodeWidget):
    """Base widget for model nodes."""
    
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        
        # Update styling while keeping the red theme
        self.setStyleSheet("""
            QWidget {
                background-color: #FFEBEE;
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 9pt;
                background-color: transparent;
                padding: 2px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #FFCDD2;
                padding: 2px 5px;
                border-radius: 2px;
                min-height: 20px;
                color: #333333;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                border: none;
                width: 16px;
                border-left: 1px solid #FFCDD2;
                background-color: #FFEBEE;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #FFCDD2;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url(resources/icons/up_arrow.png);
                width: 10px;
                height: 10px;
                border: 1px solid #E57373;
                background: #FFFFFF;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url(resources/icons/down_arrow.png);
                width: 10px;
                height: 10px;
                border: 1px solid #E57373;
                background: #FFFFFF;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #FFCDD2;
                padding: 2px 5px;
                border-radius: 2px;
                min-height: 20px;
                color: #333333;
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
                border-left: 1px solid #FFCDD2;
                background-color: #FFEBEE;
            }
            QComboBox::drop-down:hover {
                background-color: #FFCDD2;
            }
            QComboBox::down-arrow {
                image: url(resources/icons/down_arrow.png);
                width: 10px;
                height: 10px;
                border: 1px solid #E57373;
                background: #FFFFFF;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                selection-background-color: #FFEBEE;
                selection-color: #333333;
                border: 1px solid #FFCDD2;
            }
            QFormLayout {
                spacing: 10px;
            }
        """)
        
        # Update title styling
        self.title_label.setStyleSheet("""
            font-weight: bold;
            color: white;
            background-color: #E57373;
            padding: 8px;
            border-radius: 3px;
            margin-bottom: 8px;
        """)

class ClassificationModelWidget(ModelWidget):
    """Base widget for classification models."""
    
    params_changed = pyqtSignal(dict)
    train_requested = pyqtSignal()
    
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        
        # Remove file-related info from parent
        if hasattr(self, 'file_name_label'):
            self.file_name_label.setParent(None)
        if hasattr(self, 'rows_label'):
            self.rows_label.setParent(None)
        if hasattr(self, 'columns_label'):
            self.columns_label.setParent(None)
        if hasattr(self, 'target_combo'):
            self.target_combo.setParent(None)
        if hasattr(self, 'details_layout'):
            self.details_layout.setParent(None)
        
        # Model info section with improved layout
        self.info_layout = QFormLayout()
        self.info_layout.setSpacing(8)
        self.info_layout.setContentsMargins(8, 8, 8, 8)
        
        # Data status
        self.data_status = QLabel("No data loaded")
        self.accuracy_label = QLabel("-")
        self.samples_label = QLabel("-")
        
        # Style the info labels
        info_style = "color: #333333; font-weight: bold;"
        self.data_status.setStyleSheet(info_style)
        self.accuracy_label.setStyleSheet(info_style)
        self.samples_label.setStyleSheet(info_style)
        
        self.info_layout.addRow("Status:", self.data_status)
        self.info_layout.addRow("Accuracy:", self.accuracy_label)
        self.info_layout.addRow("Samples:", self.samples_label)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Training: %p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E57373;
                border-radius: 3px;
                text-align: center;
                background-color: white;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #E57373;
            }
        """)
        self.progress_bar.hide()  # Initially hidden
        
        # Add train button with improved styling
        self.train_button = QPushButton("Train Model")
        self.train_button.setFixedHeight(30)
        self.train_button.setStyleSheet("""
            QPushButton {
                background-color: #E57373;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #EF5350;
            }
            QPushButton:pressed {
                background-color: #F44336;
            }
            QPushButton:disabled {
                background-color: #FFCDD2;
            }
        """)
        self.train_button.clicked.connect(self._on_train_clicked)
        
        # Add layouts to main layout
        self.layout.addLayout(self.info_layout)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.train_button)
        
        # Set fixed height for consistent sizing
        self.setFixedHeight(400)
    
    def _on_train_clicked(self):
        """Handle train button click."""
        self.data_status.setText("Training...")
        self.train_button.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.train_requested.emit()
    
    def update_training_info(self, info: dict):
        """Update the training information display."""
        if 'status' in info:
            self.data_status.setText(info['status'])
        if 'accuracy' in info:
            self.accuracy_label.setText(f"{info['accuracy']:.2%}")
        if 'samples' in info:
            self.samples_label.setText(str(info['samples']))
        if 'progress' in info:
            self.progress_bar.setValue(int(info['progress'] * 100))
            if info['progress'] >= 1.0:
                self.progress_bar.hide()
                self.train_button.setEnabled(True)

class LogisticRegressionWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Logistic Regression", parent)
        
        # Add parameters form layout
        params_layout = QFormLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(8, 4, 8, 8)
        
        # C value (inverse regularization)
        self.c_value = QDoubleSpinBox()
        self.c_value.setRange(0.1, 100.0)
        self.c_value.setValue(1.0)
        self.c_value.setSingleStep(0.1)
        self.c_value.setFixedWidth(80)
        self.c_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("C:", self.c_value)
        
        # Max iterations
        self.max_iter = QSpinBox()
        self.max_iter.setRange(100, 1000)
        self.max_iter.setValue(100)
        self.max_iter.setSingleStep(50)
        self.max_iter.setFixedWidth(80)
        self.max_iter.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Max Iterations:", self.max_iter)
        
        # Solver selection
        self.solver = NavigableComboBox(
            label="Solver:",
            items=['lbfgs', 'liblinear', 'newton-cg', 'sag']
        )
        params_layout.addRow("", self.solver)
        
        # Insert parameters layout after description
        self.layout.insertLayout(2, params_layout)
        
        # Connect signals
        self.c_value.valueChanged.connect(self._on_params_changed)
        self.max_iter.valueChanged.connect(self._on_params_changed)
        self.solver.value_changed.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'C': self.c_value.value(),
            'max_iter': self.max_iter.value(),
            'solver': self.solver.currentText()
        })

class DecisionTreeWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Decision Tree", parent)
        
        # Add parameters form layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        params_layout = QFormLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Max depth
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 50)
        self.max_depth.setValue(5)
        self.max_depth.setFixedWidth(80)
        self.max_depth.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Max Depth:", self.max_depth)
        
        # Min samples split
        self.min_samples_split = QSpinBox()
        self.min_samples_split.setRange(2, 20)
        self.min_samples_split.setValue(2)
        self.min_samples_split.setFixedWidth(80)
        self.min_samples_split.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Min Samples:", self.min_samples_split)
        
        content_layout.addLayout(params_layout)
        
        # Criterion
        self.criterion = NavigableComboBox(
            label="Criterion:",
            items=['gini', 'entropy']
        )
        content_layout.addWidget(self.criterion)
        
        # Add content layout to main layout
        self.layout.insertLayout(1, content_layout)
        
        # Connect signals
        self.max_depth.valueChanged.connect(self._on_params_changed)
        self.min_samples_split.valueChanged.connect(self._on_params_changed)
        self.criterion.value_changed.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'max_depth': self.max_depth.value(),
            'min_samples_split': self.min_samples_split.value(),
            'criterion': self.criterion.currentText()
        })

class RandomForestWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Random Forest", parent)
        
        # Add parameters form layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        params_layout = QFormLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Number of trees
        self.n_estimators = QSpinBox()
        self.n_estimators.setRange(10, 1000)
        self.n_estimators.setValue(100)
        self.n_estimators.setFixedWidth(80)
        self.n_estimators.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Trees:", self.n_estimators)
        
        # Max depth
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 50)
        self.max_depth.setValue(5)
        self.max_depth.setFixedWidth(80)
        self.max_depth.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Max Depth:", self.max_depth)
        
        content_layout.addLayout(params_layout)
        
        # Criterion
        self.criterion = NavigableComboBox(
            label="Criterion:",
            items=['gini', 'entropy']
        )
        content_layout.addWidget(self.criterion)
        
        # Add content layout to main layout
        self.layout.insertLayout(1, content_layout)
        
        # Connect signals
        self.n_estimators.valueChanged.connect(self._on_params_changed)
        self.max_depth.valueChanged.connect(self._on_params_changed)
        self.criterion.value_changed.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'n_estimators': self.n_estimators.value(),
            'max_depth': self.max_depth.value(),
            'criterion': self.criterion.currentText()
        })

class SVMWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Support Vector Machine", parent)
        
        # Add parameters form layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        # Kernel selection
        self.kernel = NavigableComboBox(
            label="Kernel:",
            items=['rbf', 'linear', 'poly', 'sigmoid']
        )
        content_layout.addWidget(self.kernel)
        
        params_layout = QFormLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(0, 0, 0, 0)
        
        # C value
        self.c_value = QDoubleSpinBox()
        self.c_value.setRange(0.1, 100.0)
        self.c_value.setValue(1.0)
        self.c_value.setSingleStep(0.1)
        self.c_value.setFixedWidth(80)
        self.c_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("C:", self.c_value)
        
        content_layout.addLayout(params_layout)
        
        # Gamma selection
        self.gamma = NavigableComboBox(
            label="Gamma:",
            items=['scale', 'auto']
        )
        content_layout.addWidget(self.gamma)
        
        # Add content layout to main layout
        self.layout.insertLayout(1, content_layout)
        
        # Connect signals
        self.kernel.value_changed.connect(self._on_params_changed)
        self.c_value.valueChanged.connect(self._on_params_changed)
        self.gamma.value_changed.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'kernel': self.kernel.currentText(),
            'C': self.c_value.value(),
            'gamma': self.gamma.currentText()
        })

class NaiveBayesWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Naive Bayes", parent)
        
        # Add parameters form layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        # Type selection
        self.nb_type = NavigableComboBox(
            label="Type:",
            items=['Gaussian', 'Multinomial']
        )
        content_layout.addWidget(self.nb_type)
        
        # Add content layout to main layout
        self.layout.insertLayout(1, content_layout)
        
        # Connect signals
        self.nb_type.value_changed.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'type': self.nb_type.currentText()
        })

class KNNWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("K-Nearest Neighbors", parent)
        
        # Add parameters form layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        params_layout = QFormLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Number of neighbors
        self.n_neighbors = QSpinBox()
        self.n_neighbors.setRange(1, 50)
        self.n_neighbors.setValue(5)
        self.n_neighbors.setFixedWidth(80)
        self.n_neighbors.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Neighbors:", self.n_neighbors)
        
        content_layout.addLayout(params_layout)
        
        # Weights
        self.weights = NavigableComboBox(
            label="Weights:",
            items=['uniform', 'distance']
        )
        content_layout.addWidget(self.weights)
        
        # Metric
        self.metric = NavigableComboBox(
            label="Metric:",
            items=['minkowski', 'euclidean', 'manhattan']
        )
        content_layout.addWidget(self.metric)
        
        # Add content layout to main layout
        self.layout.insertLayout(1, content_layout)
        
        # Connect signals
        self.n_neighbors.valueChanged.connect(self._on_params_changed)
        self.weights.value_changed.connect(self._on_params_changed)
        self.metric.value_changed.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'n_neighbors': self.n_neighbors.value(),
            'weights': self.weights.currentText(),
            'metric': self.metric.currentText()
        })

# Now add corresponding Node classes for each widget
class LogisticRegressionNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 400, name)  # Increased height
        self.widget_content = LogisticRegressionWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.widget_content.train_requested.connect(self._on_train_requested)
        self.setup_widget(self.widget_content)
        self.model = LogisticRegression()
        
        # Store data
        self.train_data = None
        self.val_data = None
    
    def _on_params_changed(self, params):
        self.model = LogisticRegression(**params)
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        if input_name == "train":
            self.train_data = data_package
            self.widget_content.update_training_info({
                'status': 'Training data loaded',
                'samples': len(data_package["data"])
            })
        elif input_name == "validation":
            self.val_data = data_package
            self.widget_content.update_training_info({
                'status': 'Validation data loaded',
                'samples': len(data_package["data"])
            })
    
    def _on_train_requested(self):
        """Handle training request."""
        if self.train_data is None:
            self.widget_content.update_training_info({
                'status': 'No training data',
                'progress': 1.0
            })
            return
        
        try:
            # Get training data
            X_train = self.train_data["data"].drop(columns=[self.train_data["target_column"]])
            y_train = self.train_data["data"][self.train_data["target_column"]]
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Preparing data...',
                'progress': 0.2
            })
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Training...',
                'progress': 0.6
            })
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_train, y_train)
            
            # If validation data exists, calculate validation accuracy
            val_accuracy = None
            if self.val_data is not None:
                X_val = self.val_data["data"].drop(columns=[self.val_data["target_column"]])
                y_val = self.val_data["data"][self.val_data["target_column"]]
                val_accuracy = self.model.score(X_val, y_val)
            
            # Update final status
            self.widget_content.update_training_info({
                'status': 'Training complete',
                'accuracy': val_accuracy if val_accuracy is not None else train_accuracy,
                'samples': len(self.train_data["data"]),
                'progress': 1.0
            })
            
        except Exception as e:
            self.widget_content.update_training_info({
                'status': f'Error: {str(e)}',
                'progress': 1.0
            })

class DecisionTreeNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 400, name)  # Increased height
        self.widget_content = DecisionTreeWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.widget_content.train_requested.connect(self._on_train_requested)
        self.setup_widget(self.widget_content)
        self.model = DecisionTreeClassifier()
        
        # Store data
        self.train_data = None
        self.val_data = None
    
    def _on_params_changed(self, params):
        self.model = DecisionTreeClassifier(**params)
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        if input_name == "train":
            self.train_data = data_package
            self.widget_content.update_training_info({
                'status': 'Training data loaded',
                'samples': len(data_package["data"])
            })
        elif input_name == "validation":
            self.val_data = data_package
            self.widget_content.update_training_info({
                'status': 'Validation data loaded',
                'samples': len(data_package["data"])
            })
    
    def _on_train_requested(self):
        """Handle training request."""
        if self.train_data is None:
            self.widget_content.update_training_info({
                'status': 'No training data',
                'progress': 1.0
            })
            return
        
        try:
            # Get training data
            X_train = self.train_data["data"].drop(columns=[self.train_data["target_column"]])
            y_train = self.train_data["data"][self.train_data["target_column"]]
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Preparing data...',
                'progress': 0.2
            })
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Training...',
                'progress': 0.6
            })
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_train, y_train)
            
            # If validation data exists, calculate validation accuracy
            val_accuracy = None
            if self.val_data is not None:
                X_val = self.val_data["data"].drop(columns=[self.val_data["target_column"]])
                y_val = self.val_data["data"][self.val_data["target_column"]]
                val_accuracy = self.model.score(X_val, y_val)
            
            # Update final status
            self.widget_content.update_training_info({
                'status': 'Training complete',
                'accuracy': val_accuracy if val_accuracy is not None else train_accuracy,
                'samples': len(self.train_data["data"]),
                'progress': 1.0
            })
            
        except Exception as e:
            self.widget_content.update_training_info({
                'status': f'Error: {str(e)}',
                'progress': 1.0
            })

class RandomForestNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 400, name)  # Increased height
        self.widget_content = RandomForestWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.widget_content.train_requested.connect(self._on_train_requested)
        self.setup_widget(self.widget_content)
        self.model = RandomForestClassifier()
        
        # Store data
        self.train_data = None
        self.val_data = None
    
    def _on_params_changed(self, params):
        self.model = RandomForestClassifier(**params)
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        if input_name == "train":
            self.train_data = data_package
            self.widget_content.update_training_info({
                'status': 'Training data loaded',
                'samples': len(data_package["data"])
            })
        elif input_name == "validation":
            self.val_data = data_package
            self.widget_content.update_training_info({
                'status': 'Validation data loaded',
                'samples': len(data_package["data"])
            })
    
    def _on_train_requested(self):
        """Handle training request."""
        if self.train_data is None:
            self.widget_content.update_training_info({
                'status': 'No training data',
                'progress': 1.0
            })
            return
        
        try:
            # Get training data
            X_train = self.train_data["data"].drop(columns=[self.train_data["target_column"]])
            y_train = self.train_data["data"][self.train_data["target_column"]]
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Preparing data...',
                'progress': 0.2
            })
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Training...',
                'progress': 0.6
            })
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_train, y_train)
            
            # If validation data exists, calculate validation accuracy
            val_accuracy = None
            if self.val_data is not None:
                X_val = self.val_data["data"].drop(columns=[self.val_data["target_column"]])
                y_val = self.val_data["data"][self.val_data["target_column"]]
                val_accuracy = self.model.score(X_val, y_val)
            
            # Update final status
            self.widget_content.update_training_info({
                'status': 'Training complete',
                'accuracy': val_accuracy if val_accuracy is not None else train_accuracy,
                'samples': len(self.train_data["data"]),
                'progress': 1.0
            })
            
        except Exception as e:
            self.widget_content.update_training_info({
                'status': f'Error: {str(e)}',
                'progress': 1.0
            })

class SVMNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 400, name)  # Increased height
        self.widget_content = SVMWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.widget_content.train_requested.connect(self._on_train_requested)
        self.setup_widget(self.widget_content)
        self.model = SVC()
        
        # Store data
        self.train_data = None
        self.val_data = None
    
    def _on_params_changed(self, params):
        self.model = SVC(**params)
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        if input_name == "train":
            self.train_data = data_package
            self.widget_content.update_training_info({
                'status': 'Training data loaded',
                'samples': len(data_package["data"])
            })
        elif input_name == "validation":
            self.val_data = data_package
            self.widget_content.update_training_info({
                'status': 'Validation data loaded',
                'samples': len(data_package["data"])
            })
    
    def _on_train_requested(self):
        """Handle training request."""
        if self.train_data is None:
            self.widget_content.update_training_info({
                'status': 'No training data',
                'progress': 1.0
            })
            return
        
        try:
            # Get training data
            X_train = self.train_data["data"].drop(columns=[self.train_data["target_column"]])
            y_train = self.train_data["data"][self.train_data["target_column"]]
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Preparing data...',
                'progress': 0.2
            })
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Training...',
                'progress': 0.6
            })
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_train, y_train)
            
            # If validation data exists, calculate validation accuracy
            val_accuracy = None
            if self.val_data is not None:
                X_val = self.val_data["data"].drop(columns=[self.val_data["target_column"]])
                y_val = self.val_data["data"][self.val_data["target_column"]]
                val_accuracy = self.model.score(X_val, y_val)
            
            # Update final status
            self.widget_content.update_training_info({
                'status': 'Training complete',
                'accuracy': val_accuracy if val_accuracy is not None else train_accuracy,
                'samples': len(self.train_data["data"]),
                'progress': 1.0
            })
            
        except Exception as e:
            self.widget_content.update_training_info({
                'status': f'Error: {str(e)}',
                'progress': 1.0
            })

class NaiveBayesNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 400, name)  # Increased height
        self.widget_content = NaiveBayesWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.widget_content.train_requested.connect(self._on_train_requested)
        self.setup_widget(self.widget_content)
        self.model = GaussianNB()
        
        # Store data
        self.train_data = None
        self.val_data = None
    
    def _on_params_changed(self, params):
        if params['type'] == 'Gaussian':
            self.model = GaussianNB()
        else:
            self.model = MultinomialNB()
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        if input_name == "train":
            self.train_data = data_package
            self.widget_content.update_training_info({
                'status': 'Training data loaded',
                'samples': len(data_package["data"])
            })
        elif input_name == "validation":
            self.val_data = data_package
            self.widget_content.update_training_info({
                'status': 'Validation data loaded',
                'samples': len(data_package["data"])
            })
    
    def _on_train_requested(self):
        """Handle training request."""
        if self.train_data is None:
            self.widget_content.update_training_info({
                'status': 'No training data',
                'progress': 1.0
            })
            return
        
        try:
            # Get training data
            X_train = self.train_data["data"].drop(columns=[self.train_data["target_column"]])
            y_train = self.train_data["data"][self.train_data["target_column"]]
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Preparing data...',
                'progress': 0.2
            })
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Training...',
                'progress': 0.6
            })
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_train, y_train)
            
            # If validation data exists, calculate validation accuracy
            val_accuracy = None
            if self.val_data is not None:
                X_val = self.val_data["data"].drop(columns=[self.val_data["target_column"]])
                y_val = self.val_data["data"][self.val_data["target_column"]]
                val_accuracy = self.model.score(X_val, y_val)
            
            # Update final status
            self.widget_content.update_training_info({
                'status': 'Training complete',
                'accuracy': val_accuracy if val_accuracy is not None else train_accuracy,
                'samples': len(self.train_data["data"]),
                'progress': 1.0
            })
            
        except Exception as e:
            self.widget_content.update_training_info({
                'status': f'Error: {str(e)}',
                'progress': 1.0
            })

class KNNNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 400, name)  # Increased height
        self.widget_content = KNNWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.widget_content.train_requested.connect(self._on_train_requested)
        self.setup_widget(self.widget_content)
        self.model = KNeighborsClassifier()
        
        # Store data
        self.train_data = None
        self.val_data = None
    
    def _on_params_changed(self, params):
        self.model = KNeighborsClassifier(**params)
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        if input_name == "train":
            self.train_data = data_package
            self.widget_content.update_training_info({
                'status': 'Training data loaded',
                'samples': len(data_package["data"])
            })
        elif input_name == "validation":
            self.val_data = data_package
            self.widget_content.update_training_info({
                'status': 'Validation data loaded',
                'samples': len(data_package["data"])
            })
    
    def _on_train_requested(self):
        """Handle training request."""
        if self.train_data is None:
            self.widget_content.update_training_info({
                'status': 'No training data',
                'progress': 1.0
            })
            return
        
        try:
            # Get training data
            X_train = self.train_data["data"].drop(columns=[self.train_data["target_column"]])
            y_train = self.train_data["data"][self.train_data["target_column"]]
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Preparing data...',
                'progress': 0.2
            })
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Update progress
            self.widget_content.update_training_info({
                'status': 'Training...',
                'progress': 0.6
            })
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_train, y_train)
            
            # If validation data exists, calculate validation accuracy
            val_accuracy = None
            if self.val_data is not None:
                X_val = self.val_data["data"].drop(columns=[self.val_data["target_column"]])
                y_val = self.val_data["data"][self.val_data["target_column"]]
                val_accuracy = self.model.score(X_val, y_val)
            
            # Update final status
            self.widget_content.update_training_info({
                'status': 'Training complete',
                'accuracy': val_accuracy if val_accuracy is not None else train_accuracy,
                'samples': len(self.train_data["data"]),
                'progress': 1.0
            })
            
        except Exception as e:
            self.widget_content.update_training_info({
                'status': f'Error: {str(e)}',
                'progress': 1.0
            })
