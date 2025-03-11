import os
import pandas as pd
from typing import Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QFormLayout, QHBoxLayout, QMessageBox,
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsProxyWidget,
    QSizePolicy, QSpinBox, QDoubleSpinBox, QScrollArea, QToolTip
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QObject
from PyQt6.QtGui import QColor, QPen, QBrush, QFont
from ..components.combo_box import NavigableComboBox

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
        self.target_combo = NavigableComboBox(
            label="Target:",
            items=[]
        )
        self.target_combo.setEnabled(False)  # Disabled until data is loaded
        self.target_combo.value_changed.connect(self._on_target_changed)
        self.layout.addWidget(self.target_combo)
        
        # Add some stretch
        self.layout.addStretch(1)
    
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
            self.target_combo.clear()
            self.target_combo.addItems([str(col) for col in data.columns])
            
            # Enable the combo box now that we have data
            self.target_combo.setEnabled(True)
            
            # Set default target column (last column)
            if len(data.columns) > 0:
                self.target_combo.setCurrentText(str(data.columns[-1]))
                self.target_column = str(data.columns[-1])
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
        super().__init__("Sample Dataset", parent)
        
        # Dataset selector layout
        dataset_layout = QHBoxLayout()
        dataset_layout.setContentsMargins(0, 0, 0, 5)
        
        # Create dataset combo box
        self.dataset_combo = NavigableComboBox(
            label="Dataset:",
            items=[f"{name.capitalize()}" for name, desc in self.SAMPLE_DATASETS.items()]
        )
        
        # Store dataset keys for later use
        self.dataset_keys = list(self.SAMPLE_DATASETS.keys())
        
        # Load button
        self.load_button = QPushButton("Load Dataset")
        self.load_button.setFixedHeight(30)
        
        # Insert elements into layout after title
        self.layout.insertWidget(1, self.dataset_combo)
        self.layout.insertWidget(2, self.load_button)
    
    def get_selected_dataset(self):
        """Get the key of the currently selected dataset."""
        current_text = self.dataset_combo.currentText()
        index = self.dataset_combo.currentIndex()
        return self.dataset_keys[index]

class NodeSignals(QObject):
    positionChanged = pyqtSignal()

class DataNode(QGraphicsRectItem):
    """Base graphical node for data sources."""
    
    def __init__(self, x=0, y=0, width=220, height=200, name=None):
        super().__init__(0, 0, width, height)
        self.setPos(x, y)
        
        # Add node name
        self.name = name or "unnamed_node"
        
        # Create signals object
        self.signals = NodeSignals()
        self.positionChanged = self.signals.positionChanged
        
        # Node properties
        self.data = None
        self.metadata = {}
        self.target_column = None
        
        # Input/output definitions
        if not hasattr(self, 'inputs'):
            self.inputs = {}
        if not hasattr(self, 'outputs'):
            self.outputs = {
                "data": {
                    "type": "DataFrame",
                    "description": "Output dataset",
                    "connector": None
                }
            }
        
        # Set node style
        self.setPen(QPen(QColor("#1976D2"), 2))
        self.setBrush(QBrush(QColor("#f0f8ff")))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # Initialize storage for connectors
        self.input_connectors = {}
        
        # Create output connector
        self.output_connector = QGraphicsRectItem(0, 0, 20, 10, self)
        self.output_connector.setPen(QPen(QColor("#1976D2"), 2))
        self.output_connector.setBrush(QBrush(QColor("#dae8fc")))
        self.output_connector.is_connector = True
        self.output_connector.is_input = False
        self.output_connector.is_output = True
        
        # Create permanent output text
        self.output_text = QGraphicsTextItem(self)
        self.output_text.setPlainText("Output: data")
        self.output_text.setDefaultTextColor(QColor("black"))
        self.output_text.setFont(QFont("Arial", 8))
        
        # Add output connector to outputs dictionary
        if "data" in self.outputs:
            self.outputs["data"]["connector"] = self.output_connector
        
        # Create output label
        self.output_label = QGraphicsTextItem(self)
        self.output_label.setPlainText("output")
        self.output_label.setFont(QFont("Arial", 8))
        
        # Widget and proxy
        self.widget = None
        self.proxy = None
    
    def _create_connector_text(self, text):
        """Create a styled text item for connectors."""
        text_item = QGraphicsTextItem(self)
        text_item.setPlainText(text)
        text_item.setDefaultTextColor(QColor("black"))
        font = QFont("Arial", 8)
        font.setBold(True)
        text_item.setFont(font)
        
        # Add white background for better readability
        text_item.setHtml(f'<div style="background-color: white; padding: 2px 4px; border-radius: 2px;">{text}</div>')
        return text_item
    
    def _setup_input_connectors(self):
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
            
            # Add permanent text for the connector
            text = self._create_connector_text(f"Input: {name}")
            text_width = text.boundingRect().width()
            text.setPos(
                rect.width()/2 - text_width/2,  # Center horizontally
                -35  # Above connector, increased spacing
            )
            
        else:
            # Multiple inputs, distribute evenly across top
            width = rect.width() - 40  # Leave margins
            step = width / (num_inputs - 1) if num_inputs > 1 else 0
            
            for i, name in enumerate(self.inputs.keys()):
                connector = self.input_connectors[name]
                connector.setRect(0, 0, 20, 10)
                x_pos = 20 + i * step - 10
                connector.setPos(x_pos, -10)  # Top of node
                
                # Add permanent text for each connector
                text = self._create_connector_text(f"Input: {name}")
                text_width = text.boundingRect().width()
                text.setPos(
                    x_pos + 10 - text_width/2,  # Center under connector
                    -35  # Above connector, increased spacing
                )
    
    def _update_connector_position(self):
        """Update the output connector to center bottom of the node."""
        rect = self.rect()
        self.output_connector.setRect(0, 0, 20, 10)
        self.output_connector.setPos(
            rect.width()/2 - 10,  # Center horizontally
            rect.height()  # Bottom of node
        )
        
        # Update output text position with new style
        self.output_text = self._create_connector_text("Output: data")
        text_width = self.output_text.boundingRect().width()
        self.output_text.setPos(
            rect.width()/2 - text_width/2,  # Center horizontally
            rect.height() + 15  # Below connector
        )
    
    def update_target_column(self, column_name):
        """Update the target column and label."""
        self.target_column = column_name
        self.output_label.setPlainText(str(column_name))
    
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
    
    def mouseMoveEvent(self, event):
        """Handle node movement."""
        super().mouseMoveEvent(event)
        # Emit position changed signal
        self.positionChanged.emit()
    
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
    
    def __init__(self, x=0, y=0, name=None):
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
        
        super().__init__(x, y, 220, 200, name)
        
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
    
    def __init__(self, x=0, y=0, name=None):
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
        
        super().__init__(x, y, 220, 220, name)
        
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
            dataset_name = self.widget_content.get_selected_dataset()
            
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

class DataLoaderWidget(DataNodeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title.setText("Data Loader")
        
        # Add load button
        self.load_button = QPushButton("Load Data")
        self.load_button.setFixedHeight(32)
        self.layout.addWidget(self.load_button)
        
        # Add target column selection
        self.target_combo = NavigableComboBox(
            label="Target:",
            items=[]
        )
        self.target_combo.setEnabled(False)
        self.layout.addWidget(self.target_combo)
        
        # Connect signals
        self.target_combo.value_changed.connect(self._on_target_changed)
    
    def _on_target_changed(self):
        self.target_changed.emit(self.target_combo.currentText())
    
    def update_columns(self, columns):
        self.target_combo.clear()
        self.target_combo.addItems(columns)
        self.target_combo.setEnabled(True)

class DataSplitterWidget(DataNodeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title.setText("Data Splitter")
        
        # Add parameters form layout
        params_layout = QFormLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(8, 4, 8, 8)
        
        # Test size
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.5)
        self.test_size.setValue(0.2)
        self.test_size.setSingleStep(0.1)
        self.test_size.setFixedWidth(80)
        self.test_size.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Test Size:", self.test_size)
        
        # Random state
        self.random_state = QSpinBox()
        self.random_state.setRange(0, 100)
        self.random_state.setValue(42)
        self.random_state.setFixedWidth(80)
        self.random_state.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addRow("Random State:", self.random_state)
        
        # Insert parameters layout at the beginning
        self.layout.insertLayout(1, params_layout)
        
        # Connect signals
        self.test_size.valueChanged.connect(self._on_params_changed)
        self.random_state.valueChanged.connect(self._on_params_changed)
    
    def _on_params_changed(self):
        self.params_changed.emit({
            'test_size': self.test_size.value(),
            'random_state': self.random_state.value()
        })
