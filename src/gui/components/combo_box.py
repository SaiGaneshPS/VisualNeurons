from PyQt6.QtWidgets import QWidget, QHBoxLayout, QComboBox, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal

class NavigableComboBox(QWidget):
    """A combo box with navigation buttons."""
    
    value_changed = pyqtSignal(str)  # Signal emitted when value changes
    
    def __init__(self, label="", items=None, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Add label if provided
        if label:
            self.label = QLabel(label)
            layout.addWidget(self.label)
        
        # Create navigation buttons
        prev_button = QPushButton("<")
        next_button = QPushButton(">")
        for btn in [prev_button, next_button]:
            btn.setFixedWidth(20)
            btn.setFixedHeight(20)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #9C27B0;
                    color: white;
                    border: none;
                    border-radius: 2px;
                    font-weight: bold;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: #7B1FA2;
                }
                QPushButton:pressed {
                    background-color: #6A1B9A;
                }
            """)
        
        # Create combo box
        self.combo = QComboBox()
        if items:
            self.combo.addItems(items)
        self.combo.setFixedWidth(100)
        
        # Add widgets to layout
        layout.addWidget(prev_button)
        layout.addWidget(self.combo)
        layout.addWidget(next_button)
        layout.addStretch()
        
        # Connect signals
        prev_button.clicked.connect(self._previous_item)
        next_button.clicked.connect(self._next_item)
        self.combo.currentTextChanged.connect(self._on_value_changed)
    
    def _previous_item(self):
        """Select previous item."""
        current_idx = self.combo.currentIndex()
        count = self.combo.count()
        if count > 0:  # Only proceed if there are items
            # Wrap around to the end if at the beginning
            new_idx = count - 1 if current_idx <= 0 else current_idx - 1
            self.combo.setCurrentIndex(new_idx)
    
    def _next_item(self):
        """Select next item."""
        current_idx = self.combo.currentIndex()
        count = self.combo.count()
        if count > 0:  # Only proceed if there are items
            # Wrap around to the beginning if at the end
            new_idx = 0 if current_idx >= count - 1 else current_idx + 1
            self.combo.setCurrentIndex(new_idx)
    
    def _on_value_changed(self, value):
        """Handle value change."""
        self.value_changed.emit(value)
    
    def currentText(self):
        """Get current text."""
        return self.combo.currentText()
    
    def currentIndex(self):
        """Get current index."""
        return self.combo.currentIndex()
    
    def setCurrentText(self, text):
        """Set current text."""
        self.combo.setCurrentText(text)
    
    def addItems(self, items):
        """Add items to combo box."""
        self.combo.addItems(items)
    
    def clear(self):
        """Clear all items."""
        self.combo.clear() 