from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFormLayout, 
    QComboBox, QSpinBox, QDoubleSpinBox, QScrollArea, QGraphicsRectItem, QPushButton
)
from PyQt6.QtGui import QPen, QBrush, QColor
from PyQt6.QtCore import Qt, pyqtSignal
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .data_nodes import DataNode, DataNodeWidget

class ModelNode(DataNode):
    """Base class for model nodes."""
    
    def __init__(self, x=0, y=0, width=220, height=200, name=None):
        self.inputs = {
            "train": {
                "type": "DataFrame",
                "description": "Training dataset",
                "required": True
            },
            "validation": {
                "type": "DataFrame",
                "description": "Validation dataset",
                "required": False
            }
        }
        
        super().__init__(x, y, width, height, name)
        
        # Override styling for model nodes
        self.setPen(QPen(QColor("#E57373"), 2))  # Light red border
        self.setBrush(QBrush(QColor("#FFEBEE")))  # Very light red background
        
        # Create input connectors
        self._setup_input_connectors()
        
        # Initialize model
        self.model = None
        self.trained = False
    
    def _setup_input_connectors(self):
        """Set up input connectors for train and validation data."""
        spacing = self.rect().width() / 3
        
        # Training data connector
        self.train_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.train_connector.setPos(spacing - 10, -10)
        self.train_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.train_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.train_connector.is_connector = True
        self.train_connector.is_input = True
        
        # Validation data connector
        self.val_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.val_connector.setPos(2 * spacing - 10, -10)
        self.val_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.val_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.val_connector.is_connector = True
        self.val_connector.is_input = True
        
        # Add to input connectors dictionary
        self.input_connectors["train"] = self.train_connector
        self.input_connectors["validation"] = self.val_connector

class ModelWidget(DataNodeWidget):
    """Base widget for model nodes."""
    
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QWidget {
                background-color: #FFEBEE;
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 9pt;
                background-color: transparent;
            }
            QPushButton {
                background-color: #E57373;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #EF5350;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #FFCDD2;
                padding: 2px;
                border-radius: 2px;
            }
        """)
        
        # Update title styling
        self.title_label.setStyleSheet("""
            font-weight: bold;
            color: white;
            background-color: #E57373;
            padding: 5px;
            border-radius: 3px;
        """)

class ClassificationModelWidget(ModelWidget):
    """Base widget for classification models."""
    
    params_changed = pyqtSignal(dict)
    
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        
        # Add train/evaluate button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self._on_train_clicked)
        self.layout.addWidget(self.train_button)
        
        # Model info section
        self.info_layout = QFormLayout()
        self.training_status = QLabel("Not trained")
        self.accuracy_label = QLabel("-")
        self.samples_label = QLabel("-")
        
        self.info_layout.addRow("Status:", self.training_status)
        self.info_layout.addRow("Accuracy:", self.accuracy_label)
        self.info_layout.addRow("Samples:", self.samples_label)
        
        self.layout.addLayout(self.info_layout)
    
    def _on_train_clicked(self):
        """Handle train button click."""
        self.training_status.setText("Training...")
        self.train_button.setEnabled(False)
        self.train_requested.emit()
    
    def update_training_info(self, info: dict):
        """Update the training information display."""
        self.training_status.setText(info.get('status', 'Unknown'))
        self.accuracy_label.setText(f"{info.get('accuracy', 0):.2%}")
        self.samples_label.setText(str(info.get('samples', 0)))
        self.train_button.setEnabled(True)

class LogisticRegressionWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Logistic Regression", parent)
        
        params_layout = QFormLayout()
        
        # Add hyperparameters
        self.c_value = QDoubleSpinBox()
        self.c_value.setRange(0.1, 100.0)
        self.c_value.setValue(1.0)
        self.c_value.setSingleStep(0.1)
        params_layout.addRow("C:", self.c_value)
        
        self.max_iter = QSpinBox()
        self.max_iter.setRange(100, 1000)
        self.max_iter.setValue(100)
        self.max_iter.setSingleStep(50)
        params_layout.addRow("Max Iterations:", self.max_iter)
        
        # Add solver selection
        self.solver = QComboBox()
        self.solver.addItems(['lbfgs', 'liblinear', 'newton-cg', 'sag'])
        params_layout.addRow("Solver:", self.solver)
        
        self.layout.insertLayout(1, params_layout)
        
        # Connect signals
        for widget in [self.c_value, self.max_iter, self.solver]:
            widget.valueChanged.connect(self._on_params_changed)
            widget.currentTextChanged.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'C': self.c_value.value(),
            'max_iter': self.max_iter.value(),
            'solver': self.solver.currentText()
        })

class DecisionTreeWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Decision Tree", parent)
        
        params_layout = QFormLayout()
        
        # Add hyperparameters
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 50)
        self.max_depth.setValue(5)
        params_layout.addRow("Max Depth:", self.max_depth)
        
        self.min_samples_split = QSpinBox()
        self.min_samples_split.setRange(2, 20)
        self.min_samples_split.setValue(2)
        params_layout.addRow("Min Samples Split:", self.min_samples_split)
        
        self.criterion = QComboBox()
        self.criterion.addItems(['gini', 'entropy'])
        params_layout.addRow("Criterion:", self.criterion)
        
        self.layout.insertLayout(1, params_layout)
        
        # Connect signals
        for widget in [self.max_depth, self.min_samples_split, self.criterion]:
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._on_params_changed)
            if hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'max_depth': self.max_depth.value(),
            'min_samples_split': self.min_samples_split.value(),
            'criterion': self.criterion.currentText()
        })

class RandomForestWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Random Forest", parent)
        
        params_layout = QFormLayout()
        
        # Add hyperparameters
        self.n_estimators = QSpinBox()
        self.n_estimators.setRange(10, 1000)
        self.n_estimators.setValue(100)
        params_layout.addRow("Number of Trees:", self.n_estimators)
        
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 50)
        self.max_depth.setValue(5)
        params_layout.addRow("Max Depth:", self.max_depth)
        
        self.criterion = QComboBox()
        self.criterion.addItems(['gini', 'entropy'])
        params_layout.addRow("Criterion:", self.criterion)
        
        self.layout.insertLayout(1, params_layout)
        
        # Connect signals
        for widget in [self.n_estimators, self.max_depth, self.criterion]:
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._on_params_changed)
            if hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'n_estimators': self.n_estimators.value(),
            'max_depth': self.max_depth.value(),
            'criterion': self.criterion.currentText()
        })

class SVMWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Support Vector Machine", parent)
        
        params_layout = QFormLayout()
        
        # Add hyperparameters
        self.kernel = QComboBox()
        self.kernel.addItems(['rbf', 'linear', 'poly', 'sigmoid'])
        params_layout.addRow("Kernel:", self.kernel)
        
        self.c_value = QDoubleSpinBox()
        self.c_value.setRange(0.1, 100.0)
        self.c_value.setValue(1.0)
        self.c_value.setSingleStep(0.1)
        params_layout.addRow("C:", self.c_value)
        
        self.gamma = QComboBox()
        self.gamma.addItems(['scale', 'auto'])
        params_layout.addRow("Gamma:", self.gamma)
        
        self.layout.insertLayout(1, params_layout)
        
        # Connect signals
        for widget in [self.kernel, self.c_value, self.gamma]:
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._on_params_changed)
            if hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'kernel': self.kernel.currentText(),
            'C': self.c_value.value(),
            'gamma': self.gamma.currentText()
        })

class NaiveBayesWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("Naive Bayes", parent)
        
        params_layout = QFormLayout()
        
        # Add type selection
        self.nb_type = QComboBox()
        self.nb_type.addItems(['Gaussian', 'Multinomial'])
        params_layout.addRow("Type:", self.nb_type)
        
        self.layout.insertLayout(1, params_layout)
        
        # Connect signals
        self.nb_type.currentTextChanged.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'type': self.nb_type.currentText()
        })

class KNNWidget(ClassificationModelWidget):
    def __init__(self, parent=None):
        super().__init__("K-Nearest Neighbors", parent)
        
        params_layout = QFormLayout()
        
        # Add hyperparameters
        self.n_neighbors = QSpinBox()
        self.n_neighbors.setRange(1, 50)
        self.n_neighbors.setValue(5)
        params_layout.addRow("Number of Neighbors:", self.n_neighbors)
        
        self.weights = QComboBox()
        self.weights.addItems(['uniform', 'distance'])
        params_layout.addRow("Weights:", self.weights)
        
        self.metric = QComboBox()
        self.metric.addItems(['minkowski', 'euclidean', 'manhattan'])
        params_layout.addRow("Metric:", self.metric)
        
        self.layout.insertLayout(1, params_layout)
        
        # Connect signals
        for widget in [self.n_neighbors, self.weights, self.metric]:
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._on_params_changed)
            if hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'n_neighbors': self.n_neighbors.value(),
            'weights': self.weights.currentText(),
            'metric': self.metric.currentText()
        })

# Now add corresponding Node classes for each widget
class LogisticRegressionNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 300, name)
        self.widget_content = LogisticRegressionWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.setup_widget(self.widget_content)
        self.model = LogisticRegression()
    
    def _on_params_changed(self, params):
        self.model = LogisticRegression(**params)

class DecisionTreeNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 300, name)
        self.widget_content = DecisionTreeWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.setup_widget(self.widget_content)
        self.model = DecisionTreeClassifier()
    
    def _on_params_changed(self, params):
        self.model = DecisionTreeClassifier(**params)

class RandomForestNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 300, name)
        self.widget_content = RandomForestWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.setup_widget(self.widget_content)
        self.model = RandomForestClassifier()
    
    def _on_params_changed(self, params):
        self.model = RandomForestClassifier(**params)

class SVMNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 300, name)
        self.widget_content = SVMWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.setup_widget(self.widget_content)
        self.model = SVC()
    
    def _on_params_changed(self, params):
        self.model = SVC(**params)

class NaiveBayesNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 300, name)
        self.widget_content = NaiveBayesWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.setup_widget(self.widget_content)
        self.model = GaussianNB()
    
    def _on_params_changed(self, params):
        if params['type'] == 'Gaussian':
            self.model = GaussianNB()
        else:
            self.model = MultinomialNB()

class KNNNode(ModelNode):
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 300, name)
        self.widget_content = KNNWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        self.setup_widget(self.widget_content)
        self.model = KNeighborsClassifier()
    
    def _on_params_changed(self, params):
        self.model = KNeighborsClassifier(**params)
