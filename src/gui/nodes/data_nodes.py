import os
import pandas as pd
from typing import Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QFormLayout, QComboBox, QHBoxLayout, QMessageBox,
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsProxyWidget,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QColor, QPen, QBrush, QFont

class DataNodeWidget(QWidget):
    """Widget for data source node."""
    
    # Signal emitted when target column changes
    target_column_changed = pyqtSignal(str)
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.data = None
        self.metadata = {}
        self.target_column = None
        
        # Configure node appearance
        self.setMinimumWidth(220)
        self.setFixedWidth(220)
        self.setMinimumHeight(180)
        self.setMaximumHeight(250)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f8ff;
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
                font-size: 9pt;
                background-color: transparent;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0d47a1;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #cccccc;
                padding: 2px;
                border-radius: 2px;
            }
        """)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(6)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            font-weight: bold;
            color: white;
            background-color: #1976D2;
            padding: 5px;
            border-radius: 3px;
        """)
        self.layout.addWidget(self.title_label)
        
        # File details section
        self.details_layout = QFormLayout()
        self.details_layout.setHorizontalSpacing(10)
        self.details_layout.setVerticalSpacing(5)
        
        self.file_name_label = QLabel("No file loaded")
        self.file_name_label.setWordWrap(True)
        
        self.rows_label = QLabel("-")
        self.columns_label = QLabel("-")
        
        self.details_layout.addRow("File:", self.file_name_label)
        self.details_layout.addRow("Rows:", self.rows_label)
        self.details_layout.addRow("Columns:", self.columns_label)
        
        self.layout.addLayout(self.details_layout)
        
        # Target column section
        self.target_layout = QHBoxLayout()
        self.target_label = QLabel("Target:")
        self.target_combo = QComboBox()
        self.target_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.target_combo.installEventFilter(self)
        self.target_combo.setEnabled(False)  # Disabled until data is loaded
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        
        self.target_layout.addWidget(self.target_label)
        self.target_layout.addWidget(self.target_combo)
        
        self.layout.addLayout(self.target_layout)
        
        # Add some stretch
        self.layout.addStretch(1)
    
    def eventFilter(self, obj, event):
        """Custom event filter to help with combo box interaction."""
        if obj == self.target_combo:
            if event.type() == event.Type.MouseButtonPress:
                # Force the combo box to show its popup when clicked
                self.target_combo.showPopup()
                return True
        return super().eventFilter(obj, event)
    
    def _on_target_changed(self, column_name):
        """Handle target column selection change."""
        if column_name:  # Prevent empty selection
            self.target_column = column_name
            self.target_column_changed.emit(column_name)
    
    def update_data_info(self, data, metadata):
        """Update the displayed data information."""
        self.data = data
        self.metadata = metadata
        
        # Update file information
        if 'file_name' in metadata:
            # Truncate long filenames
            filename = metadata['file_name']
            if len(filename) > 20:
                filename = filename[:17] + "..."
            self.file_name_label.setText(filename)
            # Set tooltip to show full name on hover
            self.file_name_label.setToolTip(metadata['file_name'])
        elif 'name' in metadata:
            self.file_name_label.setText(metadata['name'])
            self.file_name_label.setToolTip(metadata['name'])
        else:
            self.file_name_label.setText("Unnamed dataset")
            self.file_name_label.setToolTip("")
        
        # Update data dimensions
        if data is not None:
            self.rows_label.setText(str(len(data)))
            self.columns_label.setText(str(len(data.columns)))
            
            # Update target column combo box
            self.target_combo.blockSignals(True)  # Prevent unwanted signals
            self.target_combo.clear()
            
            for column in data.columns:
                self.target_combo.addItem(str(column))
            
            # Enable the combo box now that we have data
            self.target_combo.setEnabled(True)
            
            # Set default target column (last column)
            if len(data.columns) > 0:
                self.target_combo.setCurrentIndex(len(data.columns) - 1)
                self.target_column = str(data.columns[-1])
            
            self.target_combo.blockSignals(False)  # Re-enable signals
        else:
            self.rows_label.setText("-")
            self.columns_label.setText("-")
            self.target_combo.clear()
            self.target_combo.setEnabled(False)


class FileNodeWidget(DataNodeWidget):
    """Widget for CSV/Excel data node."""
    
    def __init__(self, parent=None):
        super().__init__("CSV/Excel Data", parent)
        
        # Add load button
        self.load_button = QPushButton("Select File")
        self.load_button.setFixedHeight(30)
        
        # Insert load button after the title
        self.layout.insertWidget(1, self.load_button)


class SampleDatasetNodeWidget(DataNodeWidget):
    """Widget for sample dataset node."""
    
    # Available sample datasets
    SAMPLE_DATASETS = {
        'iris': 'Iris Flower Classification',
        'wine': 'Wine Classification',
        'breast_cancer': 'Breast Cancer Classification',
        'diabetes': 'Diabetes Regression',
        'digits': 'Handwritten Digits',
    }
    
    def __init__(self, parent=None):
        # Initialize dataset_combo before calling parent's __init__
        self.dataset_combo = None
        super().__init__("Sample Dataset", parent)
        
        # Dataset selector layout
        dataset_layout = QHBoxLayout()
        dataset_layout.setContentsMargins(0, 0, 0, 5)
        
        self.dataset_label = QLabel("Dataset:")
        
        # Now create the actual combo box
        self.dataset_combo = QComboBox()
        self.dataset_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.dataset_combo.installEventFilter(self)
        
        # Add sample datasets to combo box
        for name, desc in self.SAMPLE_DATASETS.items():
            self.dataset_combo.addItem(f"{name.capitalize()}", name)
        
        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(self.dataset_combo, 1)
        
        # Load button
        self.load_button = QPushButton("Load Dataset")
        self.load_button.setFixedHeight(30)
        
        # Insert elements into layout after title
        self.layout.insertLayout(1, dataset_layout)
        self.layout.insertWidget(2, self.load_button)
    
    def eventFilter(self, obj, event):
        """Custom event filter to help with combo box interaction."""
        if hasattr(self, 'dataset_combo') and obj == self.dataset_combo:
            if event.type() == event.Type.MouseButtonPress:
                # Force the combo box to show its popup when clicked
                self.dataset_combo.showPopup()
                return True
        return super().eventFilter(obj, event)

class DataNode(QGraphicsRectItem):
    """Base graphical node for data sources."""
    
    def __init__(self, x=0, y=0, width=220, height=200):
        super().__init__(0, 0, width, height)  # Create rect at origin
        self.setPos(x, y)  # Set position separately
        
        # Node properties
        self.data = None
        self.metadata = {}
        self.target_column = None
        
        # Input/output definitions
        self.inputs = {}   # Dictionary of input ports and their data types
        self.outputs = {}  # Dictionary of output ports and their data types
        
        # Set node style
        self.setPen(QPen(QColor("#1976D2"), 2))
        self.setBrush(QBrush(QColor("#f0f8ff")))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # Output connector (positioned at bottom)
        self.output_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self._update_connector_position()
        
        # Output label
        self.output_label = QGraphicsTextItem(self)
        self.output_label.setPlainText("target")
        self.output_label.setFont(QFont("Arial", 8))
        self._update_label_position()
        
        # Input connectors (will be created as needed)
        self.input_connectors = {}
        
        # Widget and proxy
        self.widget = None
        self.proxy = None
        
        # Initialize connectors
        self.setup_connectors()
    
    def setup_connectors(self):
        """Setup input and output connectors based on defined interfaces."""
        # Setup default output connector (already created)
        self.outputs["data"] = {
            "type": "DataFrame",
            "description": "Output dataset with target column",
            "connector": self.output_connector
        }
        
        # Create input connectors as needed - for source nodes, this will be empty
        for input_name, input_spec in self.inputs.items():
            input_connector = QGraphicsRectItem(0, 0, 20, 10, self)
            input_connector.setPen(QPen(QColor("#1976D2"), 2))
            input_connector.setBrush(QBrush(QColor("#dae8fc")))
            
            # Position will depend on number of inputs
            self.input_connectors[input_name] = input_connector
            input_spec["connector"] = input_connector
        
        # Position input connectors
        self._update_input_connectors_position()
    
    def _update_input_connectors_position(self):
        """Update the position of input connectors."""
        rect = self.rect()
        num_inputs = len(self.inputs)
        
        if num_inputs == 0:
            return
        
        # Calculate spacing
        if num_inputs == 1:
            # Single input, center at top
            name = list(self.inputs.keys())[0]
            connector = self.input_connectors[name]
            connector.setRect(0, 0, 20, 10)
            connector.setPos(
                rect.width()/2 - 10,  # Center horizontally
                -10  # Top of node
            )
        else:
            # Multiple inputs, distribute evenly across top
            width = rect.width() - 40  # Leave margins
            step = width / (num_inputs - 1) if num_inputs > 1 else 0
            
            for i, name in enumerate(self.inputs.keys()):
                connector = self.input_connectors[name]
                connector.setRect(0, 0, 20, 10)
                connector.setPos(
                    20 + i * step - 10,  # Distribute horizontally with margins
                    -10  # Top of node
                )
    
    def _update_connector_position(self):
        """Update the output connector to center bottom of the node."""
        rect = self.rect()
        self.output_connector.setRect(0, 0, 20, 10)
        self.output_connector.setPos(
            rect.width()/2 - 10,  # Center horizontally
            rect.height()  # Bottom of node
        )
        self.output_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.output_connector.setBrush(QBrush(QColor("#dae8fc")))
    
    def _update_label_position(self):
        """Update the output label position to be centered under the connector."""
        rect = self.rect()
        label_width = self.output_label.boundingRect().width()
        self.output_label.setPos(
            rect.width()/2 - label_width/2,
            rect.height() + 10
        )
    
    def update_target_column(self, column_name):
        """Update the target column and label."""
        self.target_column = column_name
        self.output_label.setPlainText(str(column_name))
        self._update_label_position()
    
    def setup_widget(self, widget):
        """Set up the content widget within the node."""
        # Create proxy widget if it doesn't exist
        if self.proxy is None:
            self.proxy = QGraphicsProxyWidget(self)
        
        self.proxy.setWidget(widget)
        self.proxy.setPos(0, 0)  # Position at the origin of this item
        
        # Enable proxy widget to receive events
        self.proxy.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        
        # Adjust rect to fit the widget
        widget_size = widget.size()
        self.setRect(0, 0, widget_size.width(), widget_size.height())
        
        # Update connector and label positions
        self._update_connector_position()
        self._update_label_position()
        self._update_input_connectors_position()
        
        self.widget = widget
    
    def get_output_data(self, output_name="data"):
        """Get the output data for the specified output port."""
        if output_name != "data" or self.data is None:
            return None
        
        return {
            "data": self.data,
            "target_column": self.target_column,
            "metadata": self.metadata
        }
    
    def set_input_data(self, input_name, data_package):
        """Set input data for the specified input port.
        This should be overridden by subclasses that accept input.
        """
        pass
    
    def mousePressEvent(self, event):
        # Standard handling for moving the node
        super().mousePressEvent(event)
    
    def boundingRect(self):
        """Override boundingRect to include the output connector and label."""
        base_rect = super().boundingRect()
        # Add extra space at the bottom for the connector and label
        return QRectF(
            base_rect.left(),
            base_rect.top(),
            base_rect.width(),
            base_rect.height() + 25  # Add space for connector + label
        )


class FileDataNode(DataNode):
    """Node for loading data from CSV/Excel files."""
    
    def __init__(self, x=0, y=0):
        # Initialize with no inputs - this is a source node
        self.inputs = {}
        
        # Define outputs before parent initialization
        self.outputs = {
            "data": {
                "type": "DataFrame",
                "description": "DataFrame loaded from file with metadata",
                "required_columns": []  # No required columns
            }
        }
        
        super().__init__(x, y, 220, 200)
        
        # Create the widget
        self.widget_content = FileNodeWidget()
        
        # Connect signals before adding to the scene
        self.widget_content.load_button.clicked.connect(self.select_file)
        self.widget_content.target_column_changed.connect(self.update_target_column)
        
        # Setup widget
        self.setup_widget(self.widget_content)
    
    def select_file(self):
        """Open file dialog to select a CSV or Excel file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Data File",
            "",
            "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load data from selected file."""
        try:
            # Detect file type from extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.csv':
                data = pd.read_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Store data
            self.data = data
            
            # Create metadata
            self.metadata = {
                'source_type': 'file',
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'rows': len(data),
                'columns': len(data.columns),
                'column_names': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
            
            # Update widget with the data
            self.widget_content.update_data_info(data, self.metadata)
            
            # Set default target column (last column)
            if len(data.columns) > 0:
                last_col = str(data.columns[-1])
                self.update_target_column(last_col)
            
            QMessageBox.information(None, "Success", f"Loaded {os.path.basename(file_path)} successfully")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load data: {str(e)}")


class SampleDatasetNode(DataNode):
    """Node for loading sample datasets."""
    
    def __init__(self, x=0, y=0):
        # Initialize with no inputs - this is a source node
        self.inputs = {}
        
        # Define outputs before parent initialization
        self.outputs = {
            "data": {
                "type": "DataFrame",
                "description": "Sample dataset with target column",
                "required_columns": []  # No required columns
            }
        }
        
        super().__init__(x, y, 220, 220)
        
        # Create the widget
        self.widget_content = SampleDatasetNodeWidget()
        
        # Connect signals
        self.widget_content.load_button.clicked.connect(self.load_dataset)
        self.widget_content.target_column_changed.connect(self.update_target_column)
        
        # Setup widget
        self.setup_widget(self.widget_content)
    
    def load_dataset(self):
        """Load the selected sample dataset."""
        try:
            # Get selected dataset name (stored as user data in the combo box)
            current_index = self.widget_content.dataset_combo.currentIndex()
            dataset_name = self.widget_content.dataset_combo.itemData(current_index)
            
            # Import here to avoid circular imports
            from sklearn import datasets
            
            # Load the dataset based on selection
            if dataset_name == 'iris':
                dataset = datasets.load_iris()
            elif dataset_name == 'digits':
                dataset = datasets.load_digits()
            elif dataset_name == 'wine':
                dataset = datasets.load_wine()
            elif dataset_name == 'breast_cancer':
                dataset = datasets.load_breast_cancer()
            elif dataset_name == 'diabetes':
                dataset = datasets.load_diabetes()
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Convert to pandas DataFrame
            if hasattr(dataset, 'feature_names'):
                feature_names = dataset.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(dataset.data.shape[1])]
            
            data = pd.DataFrame(dataset.data, columns=feature_names)
            
            # Add target column
            if hasattr(dataset, 'target'):
                target_name = 'target'
                data[target_name] = dataset.target
            
            # Store data
            self.data = data
            
            # Create metadata
            self.metadata = {
                'source_type': 'sample',
                'name': dataset_name,
                'rows': len(data),
                'columns': len(data.columns),
                'column_names': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
            
            # Update widget
            self.widget_content.update_data_info(data, self.metadata)
            
            # Set target column
            self.target_column = 'target'
            self.update_target_column('target')
            
            # Show success message
            QMessageBox.information(None, "Success", f"Loaded {dataset_name} dataset successfully")
            
            # Save to disk
            self.save_dataset_to_disk(dataset_name, data)
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load dataset: {str(e)}")
    
    def save_dataset_to_disk(self, dataset_name, data):
        """Save the sample dataset to disk."""
        try:
            # Create sample_datasets directory if it doesn't exist
            sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'sample_datasets')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save dataset to CSV
            file_path = os.path.join(sample_dir, f"{dataset_name}.csv")
            data.to_csv(file_path, index=False)
            
            # Update metadata with file path
            self.metadata['file_path'] = file_path
            self.metadata['file_name'] = f"{dataset_name}.csv"
            
        except Exception as e:
            print(f"Warning: Could not save dataset to disk: {str(e)}")


class SampleDatasetNode(DataNode):
    """Node for loading sample datasets."""
    
    def __init__(self, x=0, y=0):
        super().__init__(x, y, 220, 220)
        
        # Create the widget
        self.widget_content = SampleDatasetNodeWidget()
        
        # Connect signals
        self.widget_content.load_button.clicked.connect(self.load_dataset)
        self.widget_content.target_column_changed.connect(self.update_target_column)
        
        # Setup widget
        self.setup_widget(self.widget_content)
    
    def load_dataset(self):
        """Load the selected sample dataset."""
        try:
            # Get selected dataset name (stored as user data in the combo box)
            current_index = self.widget_content.dataset_combo.currentIndex()
            dataset_name = self.widget_content.dataset_combo.itemData(current_index)
            
            # Import here to avoid circular imports
            from sklearn import datasets
            
            # Load the dataset based on selection
            if dataset_name == 'iris':
                dataset = datasets.load_iris()
            elif dataset_name == 'digits':
                dataset = datasets.load_digits()
            elif dataset_name == 'wine':
                dataset = datasets.load_wine()
            elif dataset_name == 'breast_cancer':
                dataset = datasets.load_breast_cancer()
            elif dataset_name == 'diabetes':
                dataset = datasets.load_diabetes()
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
            # Convert to pandas DataFrame
            if hasattr(dataset, 'feature_names'):
                feature_names = dataset.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(dataset.data.shape[1])]
                
            data = pd.DataFrame(dataset.data, columns=feature_names)
            
            # Add target column
            if hasattr(dataset, 'target'):
                target_name = 'target'
                data[target_name] = dataset.target
                
            # Store data
            self.data = data
            
            # Create metadata
            self.metadata = {
                'source_type': 'sample',
                'name': dataset_name,
                'rows': len(data),
                'columns': len(data.columns),
                'column_names': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
            
            # Update widget
            self.widget_content.update_data_info(data, self.metadata)
            
            # Set target column
            self.target_column = 'target'
            self.update_target_column('target')
            
            # Show success message
            QMessageBox.information(None, "Success", f"Loaded {dataset_name} dataset successfully")
            
            # Save to disk (optional)
            # self.save_dataset_to_disk(dataset_name, data)
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load dataset: {str(e)}")
    
    def save_dataset_to_disk(self, dataset_name, data):
        """Save the sample dataset to disk."""
        try:
            # Create sample_datasets directory if it doesn't exist
            sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'sample_datasets')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save dataset to CSV
            file_path = os.path.join(sample_dir, f"{dataset_name}.csv")
            data.to_csv(file_path, index=False)
            
            # Update metadata with file path
            self.metadata['file_path'] = file_path
            self.metadata['file_name'] = f"{dataset_name}.csv"
        except Exception as e:
            print(f"Warning: Could not save dataset to disk: {str(e)}")