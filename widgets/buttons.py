from PySide6.QtWidgets import QPushButton, QPushButton
from PySide6.QtGui import QFont

class ButtonMain(QPushButton):
    def __init__(self, text="", parent=None, font_size=14, bold=True):
        super().__init__(text, parent)

        font = QFont("Noto Color Emoji") 
        font.setPointSize(font_size)
        font.setBold(bold)
        self.setFont(font)

        self.setStyleSheet("""
            QPushButton {
                background-color: #448aff;
                color: white;
                border: 2px solid #4f5b62;
                border-radius: 15px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #83b9ff;
            }
            QPushButton:pressed {
                background-color: #83b9ff;
            }
        """)