import os
from PySide6.QtWidgets import (
    QDialog, QLabel, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QSizePolicy
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PIL import Image

class GalleryWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Galeria de Imagens")
        self.resize(800, 600)

        self.image_folder = "data/raw"
        self.selected_image_path = None

        main_layout = QHBoxLayout(self)

        # Lista de imagens
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(200)
        self.list_widget.itemClicked.connect(self.show_image)
        main_layout.addWidget(self.list_widget)

        # Preview da imagem
        self.preview_label = QLabel("Selecione uma imagem")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.preview_label)

        self.load_images()

    def load_images(self):
        self.list_widget.clear()
        files = sorted(os.listdir(self.image_folder))
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in image_files:
            item = QListWidgetItem(img_file)
            self.list_widget.addItem(item)

    def show_image(self, item):
        filename = item.text()
        path = os.path.join(self.image_folder, filename)
        self.selected_image_path = path

        pil_image = Image.open(path)
        w = self.preview_label.width()
        h = self.preview_label.height()
        pil_image = pil_image.resize((w, h), Image.Resampling.LANCZOS)

        qt_image = QPixmap.fromImage(
            QPixmap(pil_image.filename).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        self.preview_label.setPixmap(qt_image)
        self.preview_label.setText("")
