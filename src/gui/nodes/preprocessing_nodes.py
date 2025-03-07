import pandas as pd
import numpy as np
from typing import List, Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, 
    QCheckBox, QFrame, QComboBox, QGridLayout,
    QFormLayout, QHBoxLayout, QGraphicsRectItem, QSpinBox, QPushButton,
    QDialog, QDialogButtonBox, QStyleFactory, QTableWidget, QTableWidgetItem,
    QHeaderView, QVBoxLayout, QApplication
)

import PyQt6.QtCore as QtCore
from PyQt6.QtGui import QPen, QBrush, QColor
from PyQt6.QtCore import Qt, pyqtSignal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from .data_nodes import DataNode, DataNodeWidget
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class PreprocessingNode(DataNode):
    """Base class for preprocessing nodes with input connectors."""
    
    def __init__(self, x=0, y=0, width=220, height=200, name=None):
        # Define inputs before calling super().__init__
        self.inputs = {
            "data": {
                "type": "DataFrame",
                "description": "Input dataset",
                "required": True
            }
        }
        
        super().__init__(x, y, width, height, name)
        
        # Override the styling for preprocessing nodes
        self.setPen(QPen(QColor("#FB8C00"), 2))  # Orange border
        self.setBrush(QBrush(QColor("#fff3e0")))  # Light orange background
        
        # Create and set up the input connector
        self.input_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.input_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.input_connector.setBrush(QBrush(QColor("#dae8fc")))
        
        # Explicitly set connector flags as direct properties
        # These flags are critical for the connection logic
        self.input_connector.is_connector = True  
        self.input_connector.is_input = True
        self.input_connector.is_output = False
        
        # Position it at the top center
        rect = self.rect()
        self.input_connector.setPos(
            rect.width()/2 - 10,  # Center horizontally
            -10                   # Top of node
        )
        
        # Make sure this is added to the input_connectors dictionary
        # This is critical for the connection logic
        self.input_connectors["data"] = self.input_connector
        
        # Debug visualization - add a label to make the connector more visible
        from PyQt6.QtWidgets import QGraphicsTextItem
        from PyQt6.QtGui import QFont
        text = QGraphicsTextItem(self)
        text.setPlainText("INPUT")
        text.setFont(QFont("Arial", 7))
        text.setPos(rect.width()/2 - 20, -25)
        
        print(f"PreprocessingNode: Created input connector with flags: is_connector={self.input_connector.is_connector}, is_input={self.input_connector.is_input}")
    
    def update_input_connector_position(self):
        """Update the position of the input connector."""
        if hasattr(self, 'input_connector'):
            rect = self.rect()
            self.input_connector.setPos(
                rect.width()/2 - 10,  # Center horizontally
                -10                   # Top of node
            )
    
    def setup_widget(self, widget):
        """Override to also update input connector position."""
        super().setup_widget(widget)
        self.update_input_connector_position()
        
class PreprocessingNodeWidget(DataNodeWidget):
    """Base widget for preprocessing nodes."""
    
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        
        # Remove target column selection as it's inherited from input
        self.target_layout.removeWidget(self.target_label)
        self.target_layout.removeWidget(self.target_combo)
        self.target_label.deleteLater()
        self.target_combo.deleteLater()
        self.layout.removeItem(self.target_layout)
        
        # Update styling for preprocessing nodes
        self.setStyleSheet("""
            QWidget {
                background-color: #fff3e0;  /* Light orange background */
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 9pt;
                background-color: transparent;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #cccccc;
                padding: 2px;
                border-radius: 2px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QCheckBox {
                background-color: transparent;
                padding: 2px;
            }
            QCheckBox:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)
        
        # Update title styling
        self.title_label.setStyleSheet("""
            font-weight: bold;
            color: white;
            background-color: #FB8C00;  /* Orange */
            padding: 5px;
            border-radius: 3px;
        """)
        
class FeatureSelectionEditDialog(QDialog):
    """Dialog for selecting features."""
    
    def __init__(self, features, selected_features, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Features")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        # Set proper window flags
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        # Center the dialog on the screen
        if parent:
            parent_center = parent.mapToGlobal(parent.rect().center())
            self.move(
                parent_center.x() - self.width() // 2,
                parent_center.y() - self.height() // 2
            )
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Add select/deselect all buttons
        buttons_layout = QHBoxLayout()
        select_all = QPushButton("Select All")
        deselect_all = QPushButton("Deselect All")
        select_all.clicked.connect(self._select_all)
        deselect_all.clicked.connect(self._deselect_all)
        buttons_layout.addWidget(select_all)
        buttons_layout.addWidget(deselect_all)
        layout.addLayout(buttons_layout)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Feature", "Selected"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.setAlternatingRowColors(True)
        
        # Populate table
        self.table.setRowCount(len(features))
        self.checkboxes = {}
        
        for i, feature in enumerate(features):
            # Feature name
            self.table.setItem(i, 0, QTableWidgetItem(feature))
            # Checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(feature in selected_features)
            self.table.setCellWidget(i, 1, checkbox)
            self.checkboxes[feature] = checkbox
        
        layout.addWidget(self.table)
        
        # Add buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QTableWidget {
                gridline-color: #d0d0d0;
                selection-background-color: #e0e0e0;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: none;
                border-bottom: 1px solid #d0d0d0;
            }
        """)
    
    def _select_all(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
    
    def _deselect_all(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_features(self):
        """Get list of selected features."""
        return [
            feature for feature, checkbox in self.checkboxes.items()
            if checkbox.isChecked()
        ]

class FeatureSelectionWidget(PreprocessingNodeWidget):
    """Widget for feature selection."""
    
    features_changed = pyqtSignal(list)  # Signal emitted when feature selection changes
    
    def __init__(self, parent=None):
        super().__init__("Feature Selection", parent)
        
        # Add selection info
        info_layout = QHBoxLayout()
        self.selection_label = QLabel("Selected: 0/0 features")
        self.selection_label.setStyleSheet("color: #666666;")
        info_layout.addWidget(self.selection_label)
        info_layout.addStretch()
        self.layout.addLayout(info_layout)
        
        # Add edit button
        self.edit_button = QPushButton("Edit Features")
        self.edit_button.clicked.connect(self.show_edit_dialog)
        self.layout.addWidget(self.edit_button)
        
        # Store data
        self.features = []
        self.selected_features = []
    
    def update_features(self, features: List[str], target_column: str):
        """Store feature information."""
        self.features = [f for f in features if f != target_column]
        self.selected_features = self.features.copy()
        self.update_selection_label()
    
    def update_selection_label(self):
        """Update the selection info label."""
        self.selection_label.setText(
            f"Selected: {len(self.selected_features)}/{len(self.features)} features"
        )
    
    def show_edit_dialog(self):
        """Show the edit dialog."""
        dialog = FeatureSelectionEditDialog(
            self.features,
            self.selected_features,
            self.window()  # Pass the main window as parent
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.selected_features = dialog.get_selected_features()
            self.update_selection_label()
            self.features_changed.emit(self.selected_features)

class FeatureSelectionNode(PreprocessingNode):
    """Node for selecting features to use."""
    
    def __init__(self, x=0, y=0, name=None):
                
        self.outputs = {
            "data": {
                "type": "DataFrame",
                "description": "Dataset with selected features",
                "required_columns": []
            }
        }
        
        super().__init__(x, y, 220, 300, name)
        
        # Create widget
        self.widget_content = FeatureSelectionWidget()
        self.widget_content.features_changed.connect(self._on_features_changed)
        
        # Setup widget
        self.setup_widget(self.widget_content)
        
        # Initialize selected features
        self.selected_features = []
        self.processed_data = None
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Process input data."""
        if input_name == "data" and data_package:
            self.data = data_package["data"]
            self.target_column = data_package["target_column"]
            self.metadata = data_package["metadata"]
            
            # Update feature list in widget
            self.widget_content.update_features(
                list(self.data.columns),
                self.target_column
            )
            
            # Initially select all features
            self.selected_features = [
                col for col in self.data.columns 
                if col != self.target_column
            ]
            self._process_data()
    
    def _on_features_changed(self, selected_features: List[str]):
        """Handle feature selection changes."""
        self.selected_features = selected_features
        self._process_data()
    
    def _process_data(self):
        """Process the data with selected features."""
        if self.data is None or not self.selected_features:
            return
        
        try:
            # Create new dataframe with selected features + target
            self.processed_data = self.data[
                self.selected_features + [self.target_column]
            ]
            
            # Update widget info
            self.widget_content.update_data_info(
                self.processed_data,
                {
                    'name': 'Selected Features',
                    'rows': len(self.processed_data),
                    'columns': len(self.processed_data.columns)
                }
            )
        
        except Exception as e:
            print(f"Error in feature selection: {str(e)}")
    
    def get_output_data(self, output_name="data"):
        """Get the output data."""
        if output_name == "data" and self.processed_data is not None:
            return {
                "data": self.processed_data,
                "target_column": self.target_column,
                "metadata": {
                    "type": "feature_selected",
                    "selected_features": self.selected_features,
                    "rows": len(self.processed_data),
                    "columns": len(self.processed_data.columns)
                }
            }
        return None

class MissingValuesWidget(PreprocessingNodeWidget):
    """Widget for handling missing values."""
    
    methods_changed = pyqtSignal(dict)  # Signal emitted when methods change
    
    def __init__(self, parent=None):
        super().__init__("Missing Values Handler", parent)
        
        # Add description
        self.desc_label = QLabel(
            "Handle missing values in your dataset using different methods."
        )
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #666666; margin-bottom: 10px;")
        self.layout.addWidget(self.desc_label)
        
        # Add method selection for all columns
        global_method_layout = QFormLayout()
        self.global_method = QComboBox()
        self.global_method.setFixedWidth(120)
        self.global_method.addItems(['mean', 'median', 'mode', 'constant'])
        self.global_method.currentTextChanged.connect(self._on_global_method_changed)
        global_method_layout.addRow("Apply to all:", self.global_method)
        self.layout.addLayout(global_method_layout)
        
        # Create table-like widget for features
        table_widget = QWidget()
        table_layout = QGridLayout(table_widget)
        table_layout.setSpacing(10)
        
        # Headers
        headers = ["Feature", "Missing", "Method"]
        for i, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            table_layout.addWidget(label, 0, i)
        
        # Add scroll area
        scroll = QScrollArea()
        scroll.setWidget(table_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(150)
        self.layout.addWidget(scroll)
        
        self.table_layout = table_layout
        self.method_selectors = {}
    
    def update_features(self, features: List[str], missing_counts: Dict[str, int]):
        """Update the feature list with missing value counts."""
        # Clear existing widgets
        for i in reversed(range(self.table_layout.count())): 
            self.table_layout.itemAt(i).widget().deleteLater()
        self.method_selectors.clear()
        
        # Add row for each feature
        for i, feature in enumerate(features, 1):
            # Feature name
            self.table_layout.addWidget(QLabel(feature), i, 0)
            
            # Missing count
            count = missing_counts.get(feature, 0)
            self.table_layout.addWidget(QLabel(str(count)), i, 1)
            
            # Method selector
            method_combo = QComboBox()
            method_combo.addItems(['mean', 'median', 'mode', 'constant'])
            method_combo.currentTextChanged.connect(
                lambda method, f=feature: self._on_method_changed(f, method)
            )
            self.table_layout.addWidget(method_combo, i, 2)
            self.method_selectors[feature] = method_combo
    
    def _on_method_changed(self, feature: str, method: str):
        """Emit signal with current methods for all features."""
        methods = {
            f: selector.currentText()
            for f, selector in self.method_selectors.items()
        }
        self.methods_changed.emit(methods)
    
    def _on_global_method_changed(self, method):
        """Apply selected method to all features."""
        for combo in self.method_selectors.values():
            combo.setCurrentText(method)

class MissingValuesNode(PreprocessingNode):
    """Node for handling missing values."""
    
    def __init__(self, x=0, y=0, name=None):
        
        self.outputs = {
            "data": {
                "type": "DataFrame",
                "description": "Dataset with handled missing values",
                "required_columns": []
            }
        }
        
        super().__init__(x, y, 220, 300, name)
        
        # Create widget
        self.widget_content = MissingValuesWidget()
        self.widget_content.methods_changed.connect(self._on_methods_changed)
        
        # Setup widget
        self.setup_widget(self.widget_content)
        
        # Initialize
        self.imputation_methods = {}
        self.processed_data = None
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Process input data."""
        if input_name == "data" and data_package:
            self.data = data_package["data"]
            self.target_column = data_package["target_column"]
            self.metadata = data_package["metadata"]
            
            # Get missing value counts for each feature
            features = [col for col in self.data.columns if col != self.target_column]
            missing_counts = self.data[features].isnull().sum().to_dict()
            
            # Update widget
            self.widget_content.update_features(features, missing_counts)
            
            # Initialize methods
            self.imputation_methods = {f: 'mean' for f in features}
            self._process_data()
    
    def _on_methods_changed(self, methods: Dict[str, str]):
        """Handle imputation method changes."""
        self.imputation_methods = methods
        self._process_data()
    
    def _process_data(self):
        """Process the data with selected imputation methods."""
        if self.data is None:
            return
        
        try:
            # Create copy of data
            self.processed_data = self.data.copy()
            
            # Apply imputation methods
            for feature, method in self.imputation_methods.items():
                if self.processed_data[feature].isnull().any():
                    if method == 'mean':
                        value = self.processed_data[feature].mean()
                    elif method == 'median':
                        value = self.processed_data[feature].median()
                    elif method == 'mode':
                        value = self.processed_data[feature].mode()[0]
                    else:  # constant
                        value = 0
                    
                    self.processed_data[feature].fillna(value, inplace=True)
            
            # Update widget info
            self.widget_content.update_data_info(
                self.processed_data,
                {
                    'name': 'Imputed Data',
                    'rows': len(self.processed_data),
                    'columns': len(self.processed_data.columns)
                }
            )
        
        except Exception as e:
            print(f"Error in missing value imputation: {str(e)}")
    
    def get_output_data(self, output_name="data"):
        """Get the output data."""
        if output_name == "data" and self.processed_data is not None:
            return {
                "data": self.processed_data,
                "target_column": self.target_column,
                "metadata": {
                    "type": "imputed",
                    "imputation_methods": self.imputation_methods,
                    "rows": len(self.processed_data),
                    "columns": len(self.processed_data.columns)
                }
            }
        return None

class NormalizationWidget(PreprocessingNodeWidget):
    """Widget for data normalization."""
    
    method_changed = pyqtSignal(str)  # Signal emitted when normalization method changes
    
    def __init__(self, parent=None):
        super().__init__("Normalization", parent)
        
        # Add description label
        self.desc_label = QLabel(
            "Select a normalization method to scale your features:"
        )
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #666666; margin-bottom: 10px;")
        self.layout.insertWidget(1, self.desc_label)
        
        # Normalization method selector
        form_layout = QFormLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            'StandardScaler',
            'MinMaxScaler',
            'RobustScaler',
            'Normalizer'
        ])
        
        # Add tooltips for each method
        self.method_combo.setItemData(
            0,  # StandardScaler
            "Standardize features by removing the mean and scaling to unit variance",
            Qt.ItemDataRole.ToolTipRole
        )
        self.method_combo.setItemData(
            1,  # MinMaxScaler
            "Scale features to a given range (default [0, 1])",
            Qt.ItemDataRole.ToolTipRole
        )
        self.method_combo.setItemData(
            2,  # RobustScaler
            "Scale features using statistics that are robust to outliers",
            Qt.ItemDataRole.ToolTipRole
        )
        self.method_combo.setItemData(
            3,  # Normalizer
            "Scale samples individually to unit norm",
            Qt.ItemDataRole.ToolTipRole
        )
        
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        form_layout.addRow("Method:", self.method_combo)
        
        # Add method description
        self.method_desc = QLabel()
        self.method_desc.setWordWrap(True)
        self.method_desc.setStyleSheet("color: #666666; font-style: italic;")
        form_layout.addRow("", self.method_desc)
        
        self.layout.insertLayout(2, form_layout)
        
        # Initialize description
        self._update_method_description(self.method_combo.currentText())
    
    def _on_method_changed(self, method: str):
        """Handle method change and update description."""
        self._update_method_description(method)
        self.method_changed.emit(method)
    
    def _update_method_description(self, method: str):
        """Update the description text based on selected method."""
        descriptions = {
            'StandardScaler': """
                Standardizes features by removing the mean and scaling to unit variance.
                The standard score of a sample x is calculated as: z = (x - u) / s
                where u is the mean and s is the standard deviation.
            """,
            'MinMaxScaler': """
                Transforms features by scaling each feature to a given range (default [0, 1]).
                The transformation is given by: X_scaled = (X - X_min) / (X_max - X_min)
            """,
            'RobustScaler': """
                Scales features using statistics that are robust to outliers.
                Uses the interquartile range to scale the data, making it robust to outliers.
            """,
            'Normalizer': """
                Scales samples individually to unit norm (vector length).
                Each sample is scaled independently of other samples.
            """
        }
        
        # Clean up the description text
        desc = descriptions[method].strip().replace('\n', ' ').replace('    ', '')
        self.method_desc.setText(desc)

class NormalizationNode(PreprocessingNode):
    """Node for data normalization."""
    
    def __init__(self, x=0, y=0, name=None):
        
        self.outputs = {
            "data": {
                "type": "DataFrame",
                "description": "Normalized dataset",
                "required_columns": []
            }
        }
        
        super().__init__(x, y, 220, 250, name)  # Adjusted height to fit content
        
        # Create widget
        self.widget_content = NormalizationWidget()
        self.widget_content.method_changed.connect(self._on_method_changed)
        
        # Setup widget
        self.setup_widget(self.widget_content)
        
        # Initialize
        self.normalization_method = 'StandardScaler'
        self.processed_data = None
        self._setup_scaler()
    
    def _setup_scaler(self):
        """Setup the scaler based on selected method."""
        if self.normalization_method == 'StandardScaler':
            self.scaler = StandardScaler()
        elif self.normalization_method == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif self.normalization_method == 'RobustScaler':
            self.scaler = RobustScaler()
        else:  # Normalizer
            self.scaler = Normalizer()
    
    def _on_method_changed(self, method: str):
        """Handle normalization method changes."""
        self.normalization_method = method
        self._setup_scaler()
        self._process_data()
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Process input data."""
        if input_name == "data" and data_package:
            self.data = data_package["data"]
            self.target_column = data_package["target_column"]
            self.metadata = data_package["metadata"]
            self._process_data()
    
    def _process_data(self):
        """Process the data with selected normalization method."""
        if self.data is None:
            return
        
        try:
            # Separate features and target
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            
            # Normalize features
            X_normalized = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Combine with target
            self.processed_data = X_normalized.copy()
            self.processed_data[self.target_column] = y
            
            # Update widget info
            self.widget_content.update_data_info(
                self.processed_data,
                {
                    'name': f'Normalized Data ({self.normalization_method})',
                    'rows': len(self.processed_data),
                    'columns': len(self.processed_data.columns),
                    'normalization_method': self.normalization_method
                }
            )
            
        except Exception as e:
            print(f"Error in normalization: {str(e)}")
    
    def get_output_data(self, output_name="data"):
        """Get the output data."""
        if output_name == "data" and self.processed_data is not None:
            return {
                "data": self.processed_data,
                "target_column": self.target_column,
                "metadata": {
                    "type": "normalized",
                    "normalization_method": self.normalization_method,
                    "rows": len(self.processed_data),
                    "columns": len(self.processed_data.columns)
                }
            }
        return None

class EncodingWidget(PreprocessingNodeWidget):
    """Widget for encoding categorical variables."""
    
    encoding_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__("Categorical Encoding", parent)
        
        # Add description
        self.desc_label = QLabel(
            "Select an encoding method to apply to all categorical columns."
        )
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #666666; margin-bottom: 10px;")
        self.layout.addWidget(self.desc_label)
        
        # Single encoding method for all columns
        form_layout = QFormLayout()
        self.encoding_method = QComboBox()
        self.encoding_method.setFixedWidth(120)
        self.encoding_method.addItems(['Label', 'One-Hot', 'Ordinal'])
        form_layout.addRow("Encoding Method:", self.encoding_method)
        self.layout.addLayout(form_layout)
        
        # Display columns to be encoded
        self.features_list = QLabel()
        self.features_list.setWordWrap(True)
        self.layout.addWidget(self.features_list)
        
        self.encoding_method.currentTextChanged.connect(self._on_encoding_changed)
    
    def update_features(self, features: List[str]):
        """Update the list of features to be encoded."""
        self.features = features
        self.features_list.setText(f"Features to encode: {', '.join(features)}")
        self._on_encoding_changed(self.encoding_method.currentText())
    
    def _on_encoding_changed(self, method: str):
        """Emit the same encoding method for all features."""
        self.encoding_changed.emit({f: method for f in self.features})

class EncodingNode(PreprocessingNode):
    """Node for encoding categorical variables."""
    
    def __init__(self, x=0, y=0, name=None):
        self.outputs = {
            "data": {
                "type": "DataFrame",
                "description": "Dataset with encoded features",
                "required_columns": []
            }
        }
        
        super().__init__(x, y, 220, 300, name)
        
        # Create widget
        self.widget_content = EncodingWidget()
        self.widget_content.encoding_changed.connect(self._on_encoding_changed)
        
        # Setup widget
        self.setup_widget(self.widget_content)
        
        # Initialize
        self.encoding_methods = {}
        self.processed_data = None
        self.encoders = {}
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Process input data."""
        if input_name == "data" and data_package:
            self.data = data_package["data"]
            self.target_column = data_package["target_column"]
            self.metadata = data_package["metadata"]
            
            # Update feature list in widget
            self.widget_content.data = self.data  # Set data for dtype access
            features = [col for col in self.data.columns if col != self.target_column]
            self.widget_content.update_features(features)
            
            self._process_data()
    
    def _on_encoding_changed(self, methods: Dict[str, str]):
        """Handle encoding method changes."""
        self.encoding_methods = methods
        self._process_data()
    
    def _process_data(self):
        """Process the data with selected encoding methods."""
        if self.data is None:
            return
        
        try:
            from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
            
            # Create copy of data
            self.processed_data = self.data.copy()
            
            # Apply encoding methods
            for feature, method in self.encoding_methods.items():
                if method == 'Label':
                    if feature not in self.encoders:
                        self.encoders[feature] = LabelEncoder()
                    self.processed_data[feature] = self.encoders[feature].fit_transform(
                        self.processed_data[feature]
                    )
                elif method == 'One-Hot':
                    # Create dummy variables
                    dummies = pd.get_dummies(self.processed_data[feature], prefix=feature)
                    self.processed_data = pd.concat([self.processed_data, dummies], axis=1)
                    self.processed_data.drop(columns=[feature], inplace=True)
                elif method == 'Ordinal':
                    if feature not in self.encoders:
                        self.encoders[feature] = OrdinalEncoder()
                    self.processed_data[feature] = self.encoders[feature].fit_transform(
                        self.processed_data[[feature]]
                    )
            
            # Update widget info
            self.widget_content.update_data_info(
                self.processed_data,
                {
                    'name': 'Encoded Data',
                    'rows': len(self.processed_data),
                    'columns': len(self.processed_data.columns)
                }
            )
        
        except Exception as e:
            print(f"Error in encoding: {str(e)}")
    
    def get_output_data(self, output_name="data"):
        """Get the output data."""
        if output_name == "data" and self.processed_data is not None:
            return {
                "data": self.processed_data,
                "target_column": self.target_column,
                "metadata": {
                    "type": "encoded",
                    "encoding_methods": self.encoding_methods,
                    "rows": len(self.processed_data),
                    "columns": len(self.processed_data.columns)
                }
            }
        return None

class TrainTestSplitWidget(PreprocessingNodeWidget):
    # Add signal definition at class level
    split_changed = pyqtSignal(dict)  # Add this line
    
    def __init__(self, parent=None):
        super().__init__("Train/Test Split", parent)
        
        form_layout = QFormLayout()
        form_layout.setContentsMargins(8, 8, 8, 8)
        form_layout.setSpacing(10)
        
        # Train size spinner with better layout
        train_layout = QHBoxLayout()
        self.train_spin = QSpinBox()
        self.train_spin.setRange(1, 98)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix("%")
        self.train_spin.setFixedWidth(80)
        self.train_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.train_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        train_layout.addWidget(self.train_spin)
        train_layout.addStretch()
        
        # Test size slider/spinbox
        test_layout = QHBoxLayout()
        self.test_spin = QSpinBox()
        self.test_spin.setRange(1, 98)
        self.test_spin.setValue(20)
        self.test_spin.setSuffix("%")
        self.test_spin.setFixedWidth(80)
        self.test_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.test_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        test_layout.addWidget(self.test_spin)
        test_layout.addStretch()
        
        # Validation size slider/spinbox
        val_layout = QHBoxLayout()
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 98)
        self.val_spin.setValue(10)
        self.val_spin.setSuffix("%")
        self.val_spin.setFixedWidth(80)
        self.val_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.val_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        val_layout.addWidget(self.val_spin)
        val_layout.addStretch()
        
        # Add spinboxes to form
        form_layout.addRow("Train Size:", train_layout)
        form_layout.addRow("Test Size:", test_layout)
        form_layout.addRow("Validation Size:", val_layout)
        
        # Add info labels
        self.train_count = QLabel("-")
        self.test_count = QLabel("-")
        self.val_count = QLabel("-")
        
        form_layout.addRow("Train Samples:", self.train_count)
        form_layout.addRow("Test Samples:", self.test_count)
        form_layout.addRow("Validation Samples:", self.val_count)
        
        # Random seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)
        form_layout.addRow("Random Seed:", self.seed_spin)
        
        self.layout.addLayout(form_layout)
        
        # Connect signals
        self.train_spin.valueChanged.connect(self._on_split_changed)
        self.test_spin.valueChanged.connect(self._on_split_changed)
        self.val_spin.valueChanged.connect(self._on_split_changed)
        self.seed_spin.valueChanged.connect(self._on_split_changed)
    
    def _on_split_changed(self):
        """Handle split ratio changes."""
        with QtCore.QSignalBlocker(self.val_spin):  # Prevent signal feedback loops
            total = self.train_spin.value() + self.test_spin.value()
            self.val_spin.setValue(100 - total)
        
        self.split_changed.emit({
            'train_size': self.train_spin.value() / 100,
            'test_size': self.test_spin.value() / 100,
            'val_size': self.val_spin.value() / 100,
            'random_state': self.seed_spin.value()
        })
    
    def update_counts(self, total_samples: int, split_ratios: dict):
        """Update the sample count labels."""
        train_count = int(total_samples * split_ratios['train_size'])
        test_count = int(total_samples * split_ratios['test_size'])
        val_count = int(total_samples * split_ratios['val_size'])
        
        self.train_count.setText(str(train_count))
        self.test_count.setText(str(test_count))
        self.val_count.setText(str(val_count))

class TrainTestSplitNode(PreprocessingNode):
    """Node for splitting data into train/test/validation sets."""
    
    def __init__(self, x=0, y=0, name=None):
        self.outputs = {
            "train": {
                "type": "DataFrame",
                "description": "Training dataset"
            },
            "test": {
                "type": "DataFrame",
                "description": "Testing dataset"
            },
            "validation": {
                "type": "DataFrame",
                "description": "Validation dataset"
            }
        }
        
        super().__init__(x, y, 220, 300, name)
        
        # Create widget
        self.widget_content = TrainTestSplitWidget()
        self.widget_content.split_changed.connect(self._on_split_changed)
        
        # Setup widget
        self.setup_widget(self.widget_content)
        
        # Initialize
        self.split_ratios = {
            'train_size': 0.7,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        }
    
    def _on_split_changed(self, ratios: dict):
        """Handle split ratio changes."""
        self.split_ratios = ratios
        self._process_data()
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Process input data."""
        if input_name == "data" and data_package:
            self.data = data_package["data"]
            self.target_column = data_package["target_column"]
            self.metadata = data_package["metadata"]
            
            # Update sample counts in widget
            self.widget_content.update_counts(len(self.data), self.split_ratios)
            
            self._process_data()
    
    def _process_data(self):
        """Split the data into train/test/validation sets."""
        if self.data is None:
            return
        
        try:
            from sklearn.model_selection import train_test_split
            
            # First split: separate validation set
            train_test_size = 1 - self.split_ratios['val_size']
            if train_test_size < 1:
                temp_data, self.val_data = train_test_split(
                    self.data,
                    train_size=train_test_size,
                    random_state=self.split_ratios['random_state']
                )
            else:
                temp_data = self.data
                self.val_data = pd.DataFrame(columns=self.data.columns)
            
            # Second split: separate train and test
            train_ratio = self.split_ratios['train_size'] / train_test_size
            self.train_data, self.test_data = train_test_split(
                temp_data,
                train_size=train_ratio,
                random_state=self.split_ratios['random_state']
            )
            
            # Update widget info
            self.widget_content.update_data_info(
                self.data,
                {
                    'name': 'Split Data',
                    'train_samples': len(self.train_data),
                    'test_samples': len(self.test_data),
                    'val_samples': len(self.val_data)
                }
            )
            
        except Exception as e:
            print(f"Error in train/test split: {str(e)}")
    
    def get_output_data(self, output_name="train"):
        """Get the output data for the specified split."""
        if output_name == "train" and hasattr(self, 'train_data'):
            return {
                "data": self.train_data,
                "target_column": self.target_column,
                "metadata": {**self.metadata, "split": "train"}
            }
        elif output_name == "test" and hasattr(self, 'test_data'):
            return {
                "data": self.test_data,
                "target_column": self.target_column,
                "metadata": {**self.metadata, "split": "test"}
            }
        elif output_name == "validation" and hasattr(self, 'val_data'):
            return {
                "data": self.val_data,
                "target_column": self.target_column,
                "metadata": {**self.metadata, "split": "validation"}
            }
        return None

class DataTypeEditDialog(QDialog):
    """Dialog for editing data types."""
    
    def __init__(self, features, current_types, available_types, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Data Types")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        # Set proper window flags
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        # Center the dialog on the screen
        if parent:
            parent_center = parent.mapToGlobal(parent.rect().center())
            self.move(
                parent_center.x() - self.width() // 2,
                parent_center.y() - self.height() // 2
            )
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Feature", "Current Type", "New Type"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setAlternatingRowColors(True)
        
        # Populate table
        self.table.setRowCount(len(features))
        self.type_combos = {}
        
        for i, feature in enumerate(features):
            # Feature name
            self.table.setItem(i, 0, QTableWidgetItem(feature))
            # Current type
            self.table.setItem(i, 1, QTableWidgetItem(str(current_types.get(feature, 'unknown'))))
            # New type selector
            combo = QComboBox()
            combo.addItems(available_types)
            combo.setCurrentText(str(current_types.get(feature, 'unknown')))
            self.table.setCellWidget(i, 2, combo)
            self.type_combos[feature] = combo
        
        layout.addWidget(self.table)
        
        # Add buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QTableWidget {
                gridline-color: #d0d0d0;
                selection-background-color: #e0e0e0;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: none;
                border-bottom: 1px solid #d0d0d0;
            }
        """)
    
    def get_type_mapping(self):
        """Get the mapping of features to their new types."""
        return {
            feature: combo.currentText()
            for feature, combo in self.type_combos.items()
        }

class DataTypeWidget(PreprocessingNodeWidget):
    """Widget for correcting data types."""
    
    types_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__("Data Type Correction", parent)
        
        # Add description and status
        info_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Unchanged")
        self.status_label.setStyleSheet("color: #666666;")
        info_layout.addWidget(self.status_label)
        info_layout.addStretch()
        self.layout.addLayout(info_layout)
        
        # Add edit button
        self.edit_button = QPushButton("Edit Data Types")
        self.edit_button.clicked.connect(self.show_edit_dialog)
        self.layout.addWidget(self.edit_button)
        
        # Store data
        self.features = []
        self.current_types = {}
        self.available_types = ['int64', 'float64', 'string', 'bool', 'category']
    
    def update_features(self, features: List[str], dtypes: Dict[str, str]):
        """Store feature information."""
        self.features = features
        self.current_types = dtypes
    
    def show_edit_dialog(self):
        """Show the edit dialog."""
        dialog = DataTypeEditDialog(
            self.features,
            self.current_types,
            self.available_types,
            self.window()  # Pass the main window as parent
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_types = dialog.get_type_mapping()
            if new_types != self.current_types:
                self.status_label.setText("Status: Changed")
                self.status_label.setStyleSheet("color: #FB8C00;")
            self.types_changed.emit(new_types)

class DataTypeNode(PreprocessingNode):
    """Node for correcting data types."""
    
    def __init__(self, x=0, y=0, name=None):
        self.outputs = {
            "data": {
                "type": "DataFrame",
                "description": "Dataset with corrected types",
                "required_columns": []
            }
        }
        
        super().__init__(x, y, 220, 300, name)
        
        # Create widget
        self.widget_content = DataTypeWidget()
        self.widget_content.types_changed.connect(self._on_types_changed)
        
        # Setup widget
        self.setup_widget(self.widget_content)
        
        # Initialize
        self.type_mapping = {}
        self.processed_data = None
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Process input data."""
        if input_name == "data" and data_package:
            self.data = data_package["data"]
            self.target_column = data_package["target_column"]
            self.metadata = data_package["metadata"]
            
            # Update feature list in widget
            features = [col for col in self.data.columns if col != self.target_column]
            dtypes = {col: str(self.data[col].dtype) for col in features}
            self.widget_content.update_features(features, dtypes)
            
            self._process_data()
    
    def _on_types_changed(self, type_mapping: Dict[str, str]):
        """Handle type changes."""
        self.type_mapping = type_mapping
        self._process_data()
    
    def _process_data(self):
        """Process the data with new types."""
        if self.data is None:
            return
        
        try:
            # Create copy of data
            self.processed_data = self.data.copy()
            
            # Apply type conversions
            for feature, new_type in self.type_mapping.items():
                try:
                    if new_type == 'string':
                        self.processed_data[feature] = self.processed_data[feature].astype(str)
                    elif new_type == 'category':
                        self.processed_data[feature] = self.processed_data[feature].astype('category')
                    else:
                        self.processed_data[feature] = self.processed_data[feature].astype(new_type)
                except Exception as e:
                    print(f"Error converting {feature} to {new_type}: {str(e)}")
            
            # Update widget info
            self.widget_content.update_data_info(
                self.processed_data,
                {
                    'name': 'Type Corrected Data',
                    'rows': len(self.processed_data),
                    'columns': len(self.processed_data.columns)
                }
            )
        
        except Exception as e:
            print(f"Error in type correction: {str(e)}")
    
    def get_output_data(self, output_name="data"):
        """Get the output data."""
        if output_name == "data" and self.processed_data is not None:
            return {
                "data": self.processed_data,
                "target_column": self.target_column,
                "metadata": {
                    "type": "type_corrected",
                    "type_mapping": self.type_mapping,
                    "rows": len(self.processed_data),
                    "columns": len(self.processed_data.columns)
                }
            }
        return None

class DimensionalityReductionWidget(PreprocessingNodeWidget):
    """Widget for dimensionality reduction."""
    
    params_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__("Dimensionality Reduction", parent)
        
        # Add description
        self.desc_label = QLabel(
            "Reduce the number of features while preserving important patterns."
        )
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #666666; margin-bottom: 10px;")
        self.layout.addWidget(self.desc_label)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        
        # Method selector
        self.method_combo = QComboBox()
        self.method_combo.addItems(['PCA', 't-SNE'])
        self.method_combo.setFixedWidth(120)
        form_layout.addRow("Method:", self.method_combo)
        
        # Components selector
        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(2, 100)
        self.n_components_spin.setValue(2)
        self.n_components_spin.setFixedWidth(80)
        form_layout.addRow("Components:", self.n_components_spin)
        
        # t-SNE specific parameters
        self.perplexity_spin = QSpinBox()
        self.perplexity_spin.setRange(5, 50)
        self.perplexity_spin.setValue(30)
        self.perplexity_spin.setFixedWidth(80)
        form_layout.addRow("Perplexity:", self.perplexity_spin)
        
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(250, 1000)
        self.iterations_spin.setValue(300)
        self.iterations_spin.setFixedWidth(80)
        form_layout.addRow("Iterations:", self.iterations_spin)
        
        # Add group box for better organization
        for i in range(form_layout.rowCount()):
            form_layout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().setFixedWidth(120)
        
        self.layout.addLayout(form_layout)
        
        # Explained variance (for PCA)
        self.variance_label = QLabel("-")
        form_layout.addRow("Explained Variance:", self.variance_label)
        
        self.layout.addLayout(form_layout)
        
        # Connect signals
        self.method_combo.currentTextChanged.connect(self._on_params_changed)
        self.n_components_spin.valueChanged.connect(self._on_params_changed)
        self.perplexity_spin.valueChanged.connect(self._on_params_changed)
        self.iterations_spin.valueChanged.connect(self._on_params_changed)
        
        # Initial params update
        self._on_params_changed()
    
    def _on_params_changed(self):
        """Handle parameter changes."""
        method = self.method_combo.currentText()
        params = {
            'method': method,
            'n_components': self.n_components_spin.value()
        }
        
        if method == 't-SNE':
            params.update({
                'perplexity': self.perplexity_spin.value(),
                'n_iter': self.iterations_spin.value()
            })
        
        self.params_changed.emit(params)
    
    def update_variance(self, variance: float):
        """Update the explained variance label."""
        self.variance_label.setText(f"{variance:.2%}")

class DimensionalityReductionNode(PreprocessingNode):
    """Node for dimensionality reduction."""
    
    def __init__(self, x=0, y=0, name=None):
        self.outputs = {
            "data": {
                "type": "DataFrame",
                "description": "Reduced dimension dataset",
                "required_columns": []
            }
        }
        
        super().__init__(x, y, 220, 300, name)
        
        # Create widget
        self.widget_content = DimensionalityReductionWidget()
        self.widget_content.params_changed.connect(self._on_params_changed)
        
        # Setup widget
        self.setup_widget(self.widget_content)
        
        # Initialize
        self.params = {
            'method': 'PCA',
            'n_components': 2
        }
        self.processed_data = None
    
    def _on_params_changed(self, params: dict):
        """Handle parameter changes."""
        self.params = params
        self._process_data()
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Process input data."""
        if input_name == "data" and data_package:
            self.data = data_package["data"]
            self.target_column = data_package["target_column"]
            self.metadata = data_package["metadata"]
            self._process_data()
    
    def _process_data(self):
        """Process the data with dimensionality reduction."""
        if self.data is None:
            return
        
        try:
            # Separate features and target
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            
            # Apply dimensionality reduction
            if self.params['method'] == 'PCA':
                reducer = PCA(n_components=self.params['n_components'])
                X_reduced = reducer.fit_transform(X)
                # Update explained variance
                variance = sum(reducer.explained_variance_ratio_)
                self.widget_content.update_variance(variance)
            else:  # t-SNE
                reducer = TSNE(
                    n_components=self.params['n_components'],
                    perplexity=self.params['perplexity'],
                    n_iter=self.params['n_iter']
                )
                X_reduced = reducer.fit_transform(X)
            
            # Create new dataframe
            feature_names = [f"component_{i+1}" for i in range(self.params['n_components'])]
            self.processed_data = pd.DataFrame(X_reduced, columns=feature_names, index=X.index)
            self.processed_data[self.target_column] = y
            
            # Update widget info
            self.widget_content.update_data_info(
                self.processed_data,
                {
                    'name': f'Reduced Data ({self.params["method"]})',
                    'rows': len(self.processed_data),
                    'columns': len(self.processed_data.columns)
                }
            )
            
        except Exception as e:
            print(f"Error in dimensionality reduction: {str(e)}")
    
    def get_output_data(self, output_name="data"):
        """Get the output data."""
        if output_name == "data" and self.processed_data is not None:
            return {
                "data": self.processed_data,
                "target_column": self.target_column,
                "metadata": {
                    "type": "dimension_reduced",
                    "method": self.params['method'],
                    "n_components": self.params['n_components'],
                    "rows": len(self.processed_data),
                    "columns": len(self.processed_data.columns)
                }
            }
        return None