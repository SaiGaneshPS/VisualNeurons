import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.styles import apply_application_style

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ML Visual App")
    
    # Apply custom styling
    apply_application_style(app)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()