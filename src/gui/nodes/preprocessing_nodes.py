import pandas as pd
import numpy as np
from typing import List, Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, 
    QCheckBox, QFrame, QGridLayout,
    QFormLayout, QHBoxLayout, QGraphicsRectItem, QSpinBox, QPushButton,
    QDialog, QDialogButtonBox, QStyleFactory, QTableWidget, QTableWidgetItem,
    QHeaderView, QVBoxLayout, QApplication, QStyle, QGraphicsTextItem
)

import PyQt6.QtCore as QtCore
from PyQt6.QtGui import QPen, QBrush, QColor, QFont
from PyQt6.QtCore import Qt, pyqtSignal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from .data_nodes import DataNode, DataNodeWidget
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ..components.combo_box import NavigableComboBox

class PreprocessingNode(DataNode):
    """Base class for preprocessing nodes."""
    
    def __init__(self, x=0, y=0, width=220, height=200, name=None):
        # Define inputs before calling super().__init__
        if not hasattr(self, 'inputs'):
            self.inputs = {
                "data": {
                    "type": "DataFrame",
                    "description": "Input dataset",
                    "required": True,
                    "connector": None
                }
            }
        
        # Define default outputs if not already defined by child class
        if not hasattr(self, 'outputs'):
            self.outputs = {
                "data": {
                    "type": "DataFrame",
                    "description": "Processed dataset",
                    "connector": None
                }
            }
        
        super().__init__(x, y, width, height, name)
        
        # Set node style
        self.setPen(QPen(QColor("#1976D2"), 2))
        self.setBrush(QBrush(QColor("#f0f8ff")))
        
        # Create input and output connectors
        self._setup_input_connectors()
        self._setup_output_connectors()
        
        # Debug visualization - add a label to make the connector more visible
        text = QGraphicsTextItem(self)
        text.setPlainText("INPUT")
        text.setFont(QFont("Arial", 7))
        text.setPos(self.rect().width()/2 - 20, -25)
        
        print(f"PreprocessingNode: Created input connector with flags: is_connector={self.input_connector.is_connector}, is_input={self.input_connector.is_input}")
    
    def _setup_input_connectors(self):
        """Set up input connector."""
        # Create input connector at the top center
        self.input_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.input_connector.setPos(self.rect().width()/2 - 10, -10)
        self.input_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.input_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.input_connector.is_connector = True
        self.input_connector.is_input = True
        self.input_connector.setAcceptHoverEvents(True)
        self.input_connector.setToolTip("Input: data")
        
        # Add to input connectors dictionary
        self.input_connectors["data"] = self.input_connector
        self.inputs["data"]["connector"] = self.input_connector
    
    def _setup_output_connectors(self):
        """Set up output connectors. Can be overridden by child classes."""
        # Only set up default output connector if we're using the default outputs
        if list(self.outputs.keys()) == ["data"]:
            # Update the existing output connector from parent class
            self.output_connector.setPos(self.rect().width()/2 - 10, self.rect().height())
            self.output_connector.setAcceptHoverEvents(True)
            self.output_connector.setToolTip("Output: processed data")
            self.outputs["data"]["connector"] = self.output_connector
            
            # Add labels
            self._add_output_labels()
    
    def _add_output_labels(self):
        """Add labels for each output connector."""
        rect = self.rect()
        spacing = rect.width() / 4
        
        # Output label
        self.output_label = QGraphicsTextItem(self)
        self.output_label.setPlainText("output")
        self.output_label.setFont(QFont("Arial", 8))
        label_width = self.output_label.boundingRect().width()
        self.output_label.setPos(spacing - label_width/2, rect.height() + 10)
    
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
        
        # Remove target selection from preprocessing nodes
        if hasattr(self, 'target_combo'):
            self.target_combo.deleteLater()
        
        # Update labels
        if hasattr(self, 'file_name_label'):
            self.details_layout.removeRow(0)  # Remove "File:" row
            self.status_label = QLabel("No data loaded")
            self.details_layout.insertRow(0, "Status:", self.status_label)
        
        # Update styling for preprocessing nodes
        self.setStyleSheet("""
            QWidget {
                background-color: #fff3e0;
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 9pt;
                background-color: transparent;
                padding: 2px;
            }
            QPushButton {
                background-color: #FB8C00;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #EF6C00;
            }
            QPushButton:disabled {
                background-color: #FFE0B2;
            }
            QCheckBox {
                color: #333333;
                padding: 2px;
            }
            QCheckBox:hover {
                background-color: #ffe0b2;
                border-radius: 3px;
            }
        """)
        
        # Update title styling
        self.title_label.setStyleSheet("""
            font-weight: bold;
            color: white;
            background-color: #FB8C00;
            padding: 8px;
            border-radius: 3px;
            margin-bottom: 8px;
        """)
    
    def update_data_info(self, data, metadata=None):
        """Update the data information display."""
        if data is not None:
            if metadata and 'name' in metadata:
                self.status_label.setText(metadata['name'])
            else:
                self.status_label.setText("Data loaded")
            self.rows_label.setText(str(len(data)))
            self.columns_label.setText(str(len(data.columns)))
        else:
            self.status_label.setText("No data loaded")
            self.rows_label.setText("-")
            self.columns_label.setText("-")

class FeatureSelectionEditDialog(QDialog):
    """Dialog for selecting features."""
    
    def __init__(self, features, selected_features, parent=None):
        super().__init__(None)
        self.setWindowTitle("Select Features")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        # Center on screen
        self.setGeometry(
            QStyle.alignedRect(
                Qt.LayoutDirection.LeftToRight,
                Qt.AlignmentFlag.AlignCenter,
                self.size(),
                QApplication.primaryScreen().availableGeometry(),
            )
        )
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Add select/deselect all buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        select_all = QPushButton("Select All")
        deselect_all = QPushButton("Deselect All")
        select_all.clicked.connect(self._select_all)
        deselect_all.clicked.connect(self._deselect_all)
        buttons_layout.addWidget(select_all)
        buttons_layout.addWidget(deselect_all)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Style the buttons
        for button in [select_all, deselect_all]:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #1976D2;
                    color: white;
                    border: none;
                    padding: 6px 20px;
                    border-radius: 3px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #1565C0;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
            """)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Feature", "Selected"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.setAlternatingRowColors(True)
        
        # Style the table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dae8fc;
                gridline-color: #dae8fc;
            }
            QTableWidget::item {
                padding: 8px;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #1976D2;
                color: white;
                padding: 8px;
                border: none;
            }
            QTableWidget::item:alternate {
                background-color: #f5f5f5;
            }
            QCheckBox {
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #1976D2;
                background-color: white;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #1976D2;
                background-color: #1976D2;
                border-radius: 3px;
            }
        """)
        
        # Populate table
        self.table.setRowCount(len(features))
        self.checkboxes = {}
        
        for i, feature in enumerate(features):
            # Feature name
            feature_item = QTableWidgetItem(feature)
            feature_item.setFlags(feature_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, feature_item)
            
            # Checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(feature in selected_features)
            self.table.setCellWidget(i, 1, checkbox)
            self.checkboxes[feature] = checkbox
        
        # Adjust row heights and column widths
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.table.setColumnWidth(1, 100)  # Fixed width for checkbox column
        
        layout.addWidget(self.table)
        
        # Add dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 6px 20px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(buttons)
    
    def _select_all(self):
        """Select all features."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
    
    def _deselect_all(self):
        """Deselect all features."""
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
        """Show the feature selection edit dialog."""
        dialog = FeatureSelectionEditDialog(
            self.features,
            self.selected_features,
            None  # Set parent to None to make it an independent window
        )
        
        # Make sure dialog is modal
        dialog.setModal(True)
        
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

class MissingValuesEditDialog(QDialog):
    """Dialog for configuring missing value handling methods."""
    
    def __init__(self, features, dtypes, missing_counts, current_methods, parent=None):
        super().__init__(None)
        self.setWindowTitle("Configure Missing Values")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        # Center on screen
        self.setGeometry(
            QStyle.alignedRect(
                Qt.LayoutDirection.LeftToRight,
                Qt.AlignmentFlag.AlignCenter,
                self.size(),
                QApplication.primaryScreen().availableGeometry(),
            )
        )
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Add select/deselect all buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        
        # Method selection for all
        self.global_method = NavigableComboBox(
            label="Global Method:",
            items=['mean', 'median', 'most_frequent', 'constant']
        )
        self.global_method.value_changed.connect(self._apply_global_method)
        buttons_layout.addWidget(QLabel("Apply to all:"))
        buttons_layout.addWidget(self.global_method)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Feature", "Data Type", "Missing Values", "Method"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setAlternatingRowColors(True)
        
        # Style the table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dae8fc;
                gridline-color: #dae8fc;
            }
            QTableWidget::item {
                padding: 8px;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #1976D2;
                color: white;
                padding: 8px;
                border: none;
            }
            QTableWidget::item:alternate {
                background-color: #f5f5f5;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #dae8fc;
                border-radius: 3px;
                padding: 4px;
                min-width: 120px;
            }
        """)
        
        # Populate table
        self.table.setRowCount(len(features))
        self.method_combos = {}
        
        for i, feature in enumerate(features):
            # Feature name
            feature_item = QTableWidgetItem(feature)
            feature_item.setFlags(feature_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, feature_item)
            
            # Data type
            dtype_item = QTableWidgetItem(str(dtypes[feature]))
            dtype_item.setFlags(dtype_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 1, dtype_item)
            
            # Missing values count
            missing_item = QTableWidgetItem(str(missing_counts[feature]))
            missing_item.setFlags(missing_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            missing_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 2, missing_item)
            
            # Method selector
            method_combo = NavigableComboBox(items=['mean', 'median', 'most_frequent', 'constant'])
            if feature in current_methods:
                method_combo.setCurrentText(current_methods[feature])
            self.method_combos[feature] = method_combo
            self.table.setCellWidget(i, 3, method_combo)
        
        # Adjust column widths
        self.table.setColumnWidth(1, 100)  # Data type column
        self.table.setColumnWidth(2, 100)  # Missing values column
        self.table.setColumnWidth(3, 150)  # Method column
        
        layout.addWidget(self.table)
        
        # Add dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 6px 20px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(buttons)
    
    def _apply_global_method(self, method):
        """Apply the selected method to all features."""
        for combo in self.method_combos.values():
            combo.setCurrentText(method)
    
    def get_methods(self):
        """Get the mapping of features to their selected methods."""
        return {
            feature: combo.currentText()
            for feature, combo in self.method_combos.items()
        }

class MissingValuesWidget(PreprocessingNodeWidget):
    """Widget for handling missing values."""
    
    methods_changed = pyqtSignal(dict)  # Signal emitted when methods change
    
    def __init__(self, parent=None):
        super().__init__("Missing Values", parent)
        
        # Add edit button
        self.edit_button = QPushButton("Edit Methods")
        self.edit_button.clicked.connect(self.show_edit_dialog)
        self.layout.addWidget(self.edit_button)
        
        # Store data
        self.features = []
        self.dtypes = {}
        self.missing_counts = {}
        self.current_methods = {}
    
    def update_data_info(self, data, metadata=None):
        """Update the data information display."""
        if data is not None:
            # Update missing value counts
            self.missing_counts = data.isnull().sum().to_dict()
            self.dtypes = {col: str(dtype) for col, dtype in data.dtypes.items()}
            
            # Update status label
            total_missing = sum(self.missing_counts.values())
            if total_missing > 0:
                self.status_label.setText(f"{total_missing} missing values")
            else:
                self.status_label.setText("No missing values")
            
            # Update rows and columns
            self.rows_label.setText(str(len(data)))
            self.columns_label.setText(str(len(data.columns)))
            
            # Enable edit button if there are missing values
            self.edit_button.setEnabled(total_missing > 0)
        else:
            self.status_label.setText("No data loaded")
            self.rows_label.setText("-")
            self.columns_label.setText("-")
            self.edit_button.setEnabled(False)
    
    def show_edit_dialog(self):
        """Show the missing values edit dialog."""
        dialog = MissingValuesEditDialog(
            self.features,
            self.dtypes,
            self.missing_counts,
            self.current_methods,
            None  # Set parent to None to make it an independent window
        )
        
        # Make sure dialog is modal
        dialog.setModal(True)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.current_methods = dialog.get_methods()
            self.methods_changed.emit(self.current_methods)
            
            # Update status label with method count
            method_count = len(set(self.current_methods.values()))
            self.status_label.setText(f"Using {method_count} different method(s)")
    
    def update_features(self, features: List[str], data: pd.DataFrame):
        """Update the feature list and their information."""
        self.features = features
        self.missing_counts = data[features].isnull().sum().to_dict()
        self.dtypes = {col: str(data[col].dtype) for col in features}
        
        # Initialize methods if not set
        for feature in features:
            if feature not in self.current_methods:
                self.current_methods[feature] = 'mean'
        
        # Update status
        total_missing = sum(self.missing_counts.values())
        if total_missing > 0:
            self.status_label.setText(f"{total_missing} missing values")
        else:
            self.status_label.setText("No missing values")
        self.edit_button.setEnabled(total_missing > 0)

class MissingValuesNode(PreprocessingNode):
    """Node for handling missing values."""
    
    def __init__(self, x=0, y=0, name=None):
        super().__init__(x, y, 220, 200, name)
        
        # Create widget
        self.widget_content = MissingValuesWidget()
        self.widget_content.methods_changed.connect(self._on_methods_changed)
        self.setup_widget(self.widget_content)
        
        # Initialize methods
        self.imputation_methods = {}
        self.processed_data = None
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Process input data."""
        if input_name == "data" and data_package:
            self.data = data_package["data"]
            self.target_column = data_package["target_column"]
            self.metadata = data_package["metadata"]
            
            # Update widget with data info
            features = [col for col in self.data.columns if col != self.target_column]
            self.widget_content.update_features(features, self.data)
            
            # Process data with current methods
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
                    elif method == 'most_frequent':
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
        
        # Create content layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        # Method selector
        self.method_combo = NavigableComboBox(
            label="Method:",
            items=['standard', 'minmax', 'robust', 'normalizer']
        )
        content_layout.addWidget(self.method_combo)
        
        # Add data info section
        info_layout = QFormLayout()
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(8, 8, 8, 4)
        
        self.file_label = QLabel("No file loaded")
        self.rows_label = QLabel("-")
        self.columns_label = QLabel("-")
        
        info_layout.addRow("File:", self.file_label)
        info_layout.addRow("Rows:", self.rows_label)
        info_layout.addRow("Columns:", self.columns_label)
        
        content_layout.addLayout(info_layout)
        
        # Add content layout to main layout
        self.layout.insertLayout(1, content_layout)
        
        # Connect signals
        self.method_combo.value_changed.connect(self.method_changed.emit)
        
        # Set fixed height
        self.setFixedHeight(200)

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
        super().__init__("Encoding", parent)
        
        # Create content layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        # Encoding method selector
        self.encoding_method = NavigableComboBox(
            label="Method:",
            items=['label', 'onehot', 'ordinal']
        )
        content_layout.addWidget(self.encoding_method)
        
        # Feature methods layout
        self.feature_methods_layout = QVBoxLayout()
        self.feature_methods_layout.setSpacing(4)
        self.feature_methods = {}
        
        # Add layouts to main layout
        content_layout.addLayout(self.feature_methods_layout)
        self.layout.insertLayout(1, content_layout)
        
        # Connect signals
        self.encoding_method.value_changed.connect(self.encoding_changed.emit)
        
        # Set fixed height
        self.setFixedHeight(200)

    def update_features(self, features: List[str], data: pd.DataFrame = None):
        """Update the feature list."""
        pass  # We don't need to implement this for now, but it needs to exist

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
    """Widget for train/test/validation split."""
    
    split_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__("Train/Test Split", parent)
        
        # Main form layout
        form_layout = QFormLayout()
        form_layout.setContentsMargins(8, 8, 8, 8)
        form_layout.setSpacing(8)
        
        # Train size spinner
        train_layout = QHBoxLayout()
        self.train_spin = QSpinBox()
        self.train_spin.setRange(1, 98)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix("%")
        self.train_spin.setFixedWidth(70)
        self.train_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.train_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        train_layout.addWidget(self.train_spin)
        
        # Test size spinner
        test_layout = QHBoxLayout()
        self.test_spin = QSpinBox()
        self.test_spin.setRange(1, 98)
        self.test_spin.setValue(20)
        self.test_spin.setSuffix("%")
        self.test_spin.setFixedWidth(70)
        self.test_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.test_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        test_layout.addWidget(self.test_spin)
        
        # Validation size spinner
        val_layout = QHBoxLayout()
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 98)
        self.val_spin.setValue(10)
        self.val_spin.setSuffix("%")
        self.val_spin.setFixedWidth(70)
        self.val_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.val_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        val_layout.addWidget(self.val_spin)
        
        # Add spinboxes to form
        form_layout.addRow("Train:", train_layout)
        form_layout.addRow("Test:", test_layout)
        form_layout.addRow("Validation:", val_layout)
        
        # Add info labels
        self.train_count = QLabel("-")
        self.test_count = QLabel("-")
        self.val_count = QLabel("-")
        
        # Style the labels
        for label in [self.train_count, self.test_count, self.val_count]:
            label.setStyleSheet("color: #FB8C00; font-weight: bold;")
        
        form_layout.addRow("Train Samples:", self.train_count)
        form_layout.addRow("Test Samples:", self.test_count)
        form_layout.addRow("Val. Samples:", self.val_count)
        
        # Random seed
        seed_layout = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)
        self.seed_spin.setFixedWidth(70)
        self.seed_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        seed_layout.addWidget(self.seed_spin)
        form_layout.addRow("Seed:", seed_layout)
        
        # Add form layout to main layout
        self.layout.addLayout(form_layout)
        
        # Set fixed height
        self.setFixedHeight(350)
        
        # Style the widget
        self.setStyleSheet("""
            QWidget {
                background-color: #fff3e0;
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 9pt;
                background-color: transparent;
            }
            QSpinBox {
                background-color: white;
                border: 1px solid #FB8C00;
                padding: 2px;
                color: #333333;
                min-height: 20px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                border: none;
                background: #FFE0B2;
                width: 16px;
                border-left: 1px solid #FB8C00;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #FFCC80;
            }
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
                background: #FB8C00;
            }
            QSpinBox::up-arrow {
                image: url(resources/icons/up_arrow.png);
                width: 10px;
                height: 10px;
            }
            QSpinBox::down-arrow {
                image: url(resources/icons/down_arrow.png);
                width: 10px;
                height: 10px;
            }
            QSpinBox[readOnly="true"] {
                background-color: #F5F5F5;
                border: 1px solid #BDBDBD;
            }
        """)
        
        # Style the spinboxes individually for better visibility
        for spin in [self.train_spin, self.test_spin, self.val_spin, self.seed_spin]:
            spin.setStyleSheet("""
                QSpinBox {
                    background-color: white;
                    border: 2px solid #FB8C00;
                    border-radius: 4px;
                    padding: 2px 4px;
                    color: #333333;
                    font-weight: bold;
                }
            """)
        
        # Connect signals
        self.train_spin.valueChanged.connect(self._on_split_changed)
        self.test_spin.valueChanged.connect(self._on_split_changed)
        self.val_spin.valueChanged.connect(self._on_split_changed)
        self.seed_spin.valueChanged.connect(self._on_split_changed)
    
    def _on_split_changed(self):
        """Handle split ratio changes."""
        with QtCore.QSignalBlocker(self.val_spin):
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
    """Node for displaying train/test/validation split."""
    
    def __init__(self, x=0, y=0, name=None):
        # Define outputs before parent initialization
        self.outputs = {
            "train": {
                "type": "DataFrame",
                "description": "Training dataset",
                "connector": None
            },
            "validation": {
                "type": "DataFrame",
                "description": "Validation dataset",
                "connector": None
            },
            "test": {
                "type": "DataFrame",
                "description": "Test dataset",
                "connector": None
            }
        }
        
        super().__init__(x=x, y=y, width=220, height=350, name=name)
        
        # Create widget
        self.widget_content = TrainTestSplitWidget()
        self.widget_content.split_changed.connect(self._on_split_changed)
        self.setup_widget(self.widget_content)
        
        # Initialize split ratios
        self.split_ratios = {
            'train_size': 0.7,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        }
        
        # Create output connectors
        self._setup_output_connectors()
    
    def _setup_output_connectors(self):
        """Set up output connectors for train, validation, and test data."""
        rect = self.rect()
        spacing = rect.width() / 4  # Divide width into 4 parts for even spacing
        
        # Training data connector (left)
        self.train_out = QGraphicsRectItem(0, 0, 20, 10, self)
        self.train_out.setPos(spacing - 10, rect.height())
        self.train_out.setPen(QPen(QColor("#1976D2"), 2))
        self.train_out.setBrush(QBrush(QColor("#dae8fc")))
        self.train_out.is_connector = True
        self.train_out.is_output = True
        
        # Add text for train output
        self.train_text = self._create_connector_text("Output: train")
        text_width = self.train_text.boundingRect().width()
        self.train_text.setPos(
            spacing - text_width/2,
            rect.height() + 15
        )
        
        # Validation data connector (middle)
        self.val_out = QGraphicsRectItem(0, 0, 20, 10, self)
        self.val_out.setPos(2 * spacing - 10, rect.height())
        self.val_out.setPen(QPen(QColor("#1976D2"), 2))
        self.val_out.setBrush(QBrush(QColor("#dae8fc")))
        self.val_out.is_connector = True
        self.val_out.is_output = True
        
        # Add text for validation output
        self.val_text = self._create_connector_text("Output: valid")
        text_width = self.val_text.boundingRect().width()
        self.val_text.setPos(
            2 * spacing - text_width/2,
            rect.height() + 15
        )
        
        # Test data connector (right)
        self.test_out = QGraphicsRectItem(0, 0, 20, 10, self)
        self.test_out.setPos(3 * spacing - 10, rect.height())
        self.test_out.setPen(QPen(QColor("#1976D2"), 2))
        self.test_out.setBrush(QBrush(QColor("#dae8fc")))
        self.test_out.is_connector = True
        self.test_out.is_output = True
        
        # Add text for test output
        self.test_text = self._create_connector_text("Output: test")
        text_width = self.test_text.boundingRect().width()
        self.test_text.setPos(
            3 * spacing - text_width/2,
            rect.height() + 15
        )
        
        # Add to outputs dictionary
        self.outputs["train"]["connector"] = self.train_out
        self.outputs["validation"]["connector"] = self.val_out
        self.outputs["test"]["connector"] = self.test_out
    
    def _on_split_changed(self, ratios: dict):
        """Handle changes to split ratios."""
        self.split_ratios = ratios
        self._process_data()
    
    def set_input_data(self, input_name: str, data_package: dict):
        """Handle input data."""
        if not data_package or "data" not in data_package:
            return
        
        self.data = data_package["data"]
        self.target_column = data_package.get("target_column")
        self.metadata = data_package.get("metadata", {})
        
        # Update widget with total samples
        if self.data is not None:
            self.widget_content.update_counts(len(self.data), self.split_ratios)
        
        self._process_data()
    
    def _process_data(self):
        """Process the input data and create train/validation/test splits."""
        if self.data is None:
            return
        
        try:
            from sklearn.model_selection import train_test_split
            import numpy as np
            
            # Calculate absolute sizes
            total_samples = len(self.data)
            train_size = int(total_samples * self.split_ratios['train_size'])
            val_size = int(total_samples * self.split_ratios['val_size'])
            test_size = total_samples - train_size - val_size
            
            # First split: separate test set
            train_val_data, test_data = train_test_split(
                self.data,
                test_size=test_size,
                random_state=self.split_ratios['random_state']
            )
            
            # Second split: separate validation set from training set
            if val_size > 0:
                val_fraction = val_size / (train_size + val_size)
                train_data, val_data = train_test_split(
                    train_val_data,
                    test_size=val_fraction,
                    random_state=self.split_ratios['random_state']
                )
            else:
                train_data = train_val_data
                val_data = None
            
            # Update widget with actual counts
            actual_splits = {
                'train_size': len(train_data) / total_samples,
                'val_size': len(val_data) / total_samples if val_data is not None else 0,
                'test_size': len(test_data) / total_samples
            }
            self.widget_content.update_counts(total_samples, actual_splits)
            
            # Store splits
            self.train_data = train_data
            self.val_data = val_data
            self.test_data = test_data
            
        except Exception as e:
            print(f"Error in data splitting: {str(e)}")
    
    def get_output_data(self, output_name="train"):
        """Get the output data for the specified split."""
        if output_name == "train" and hasattr(self, 'train_data'):
            return {
                "data": self.train_data,
                "target_column": self.target_column,
                "metadata": {
                    **self.metadata,
                    "split": "train",
                    "split_ratios": self.split_ratios
                }
            }
        elif output_name == "validation" and hasattr(self, 'val_data'):
            return {
                "data": self.val_data,
                "target_column": self.target_column,
                "metadata": {
                    **self.metadata,
                    "split": "validation",
                    "split_ratios": self.split_ratios
                }
            }
        elif output_name == "test" and hasattr(self, 'test_data'):
            return {
                "data": self.test_data,
                "target_column": self.target_column,
                "metadata": {
                    **self.metadata,
                    "split": "test",
                    "split_ratios": self.split_ratios
                }
            }
        return None

class DataTypeEditDialog(QDialog):
    """Dialog for editing data types."""
    
    def __init__(self, features, current_types, available_types, parent=None):
        super().__init__(None)
        self.setWindowTitle("Edit Data Types")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Feature", "Type"])
        self.table.horizontalHeader().setStretchLastSection(True)
        
        # Style the table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dae8fc;
            }
            QTableWidget::item {
                padding: 8px;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #1976D2;
                color: white;
                padding: 8px;
                border: none;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #dae8fc;
                border-radius: 3px;
                padding: 4px;
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(resources/icons/dropdown.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #dae8fc;
                selection-background-color: #e3f2fd;
                selection-color: #333333;
            }
        """)
        
        # Add features and type selectors
        self.table.setRowCount(len(features))
        self.type_combos = {}
        
        for i, feature in enumerate(features):
            # Feature name
            feature_item = QTableWidgetItem(feature)
            feature_item.setFlags(feature_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, feature_item)
            
            # Type selector
            type_combo = NavigableComboBox(items=available_types)
            if feature in current_types:
                type_combo.setCurrentText(current_types[feature])
            self.type_combos[feature] = type_combo
            self.table.setCellWidget(i, 1, type_combo)
        
        # Adjust row heights and column widths
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        
        # Add table to layout
        layout.addWidget(self.table)
        
        # Add dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 6px 20px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(button_box)
    
    def get_type_mapping(self):
        """Get the mapping of features to their selected types."""
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
        
        # Create content layout with proper spacing
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(8, 4, 8, 8)
        content_layout.setSpacing(8)
        
        # Method selector
        self.method_combo = NavigableComboBox(
            label="Method:",
            items=['PCA', 'TSNE']
        )
        content_layout.addWidget(self.method_combo)
        
        # Parameters form
        params_layout = QFormLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(8, 4, 8, 8)
        
        # Number of components
        self.n_components = QSpinBox()
        self.n_components.setRange(2, 100)
        self.n_components.setValue(2)
        self.n_components.setFixedWidth(80)
        self.n_components.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Components:", self.n_components)
        
        content_layout.addLayout(params_layout)
        
        # Add data info section
        info_layout = QFormLayout()
        info_layout.setSpacing(4)
        info_layout.setContentsMargins(8, 8, 8, 4)
        
        self.file_label = QLabel("No file loaded")
        self.rows_label = QLabel("-")
        self.columns_label = QLabel("-")
        
        info_layout.addRow("File:", self.file_label)
        info_layout.addRow("Rows:", self.rows_label)
        info_layout.addRow("Columns:", self.columns_label)
        
        content_layout.addLayout(info_layout)
        
        # Add content layout to main layout
        self.layout.insertLayout(1, content_layout)
        
        # Connect signals
        self.method_combo.value_changed.connect(self._emit_params)
        self.n_components.valueChanged.connect(self._emit_params)
        
        # Set fixed height
        self.setFixedHeight(250)
    
    def _emit_params(self):
        """Emit parameters when they change."""
        self.params_changed.emit({
            'method': self.method_combo.currentText(),
            'n_components': self.n_components.value()
        })

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