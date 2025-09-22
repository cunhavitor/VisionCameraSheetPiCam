import sys
import os
import json
import cv2
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from picamera2 import Picamera2
from config.config import INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT
from widgets.custom_widgets import ButtonMain, ImageLabel
import numpy as np


class CaptureSheetWindow(QDialog):
    def __init__(self, parent=None, picam2=None, template_path=None):
        super().__init__(parent)
        self.setWindowTitle("Captar Template")
        self.setFixedSize(1400, 700)

        # Centralizar a janela
        screen = self.screen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

        self.picam2 = picam2
        self.capturing_live = True
        self.captured_image = None

        # Layout principal
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(20)  # Espaço entre frames

        # ----------------- Frame esquerdo -----------------
        left_frame = QVBoxLayout()
        left_frame.setSpacing(15)
        main_layout.addLayout(left_frame)

        self.capture_button = ButtonMain("Capturar Foto")
        self.capture_button.clicked.connect(self.capture_photo)
        left_frame.addWidget(self.capture_button)

        self.save_button = ButtonMain("Guardar Foto")
        self.save_button.clicked.connect(self.save_photo)
        left_frame.addWidget(self.save_button)

        self.delete_button = ButtonMain("Eliminar Foto")
        self.delete_button.clicked.connect(self.delete_photo)
        left_frame.addWidget(self.delete_button)

        left_frame.addStretch()

        # ----------------- Frame direito (imagem) -----------------
        self.image_label = ImageLabel()
        main_layout.addSpacing(50)
        main_layout.addWidget(self.image_label, 1)

        # Inicializa Picamera2
        camera_config = self.picam2.create_still_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        self.update_camera_params()

        # Timer para atualizar frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

        self.finished.connect(self.on_close)

    def update_camera_params(self):
        try:
            with open("camera_params.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            print("camera_params.json nao encontrado. Usando valores padrao.")
            params = {}

        exposure_time = int(params.get("ExposureTime", 10000))
        analogue_gain = float(params.get("AnalogueGain", 1.0))
        brightness = float(params.get("Brightness", 0.5))
        contrast = float(params.get("Contrast", 1.0))
        colour_gains = params.get("ColourGains", [1.0, 1.0])
        red_gain, blue_gain = colour_gains

        try:
            self.picam2.set_controls({
                "AeEnable": False,
                "AwbEnable": False,
                "ExposureTime": exposure_time,
                "AnalogueGain": analogue_gain,
                "Brightness": brightness,
                "Contrast": contrast,
                "ColourGains": (red_gain, blue_gain)
            })
            print("Parametros da camera atualizados com sucesso.")
        except Exception as e:
            print(f"Erro ao atualizar parametros da camera: {e}")

    def update_frame(self):
        if self.capturing_live:
            frame = self.picam2.capture_array() 
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.image_label.set_image(QPixmap.fromImage(image))

    def capture_photo(self):
        self.image_label.set_border_color("#D3D3D3")
        self.picam2.stop()
        camera_config = self.picam2.create_still_configuration(
            main={"size": (4056, 3040), "format": "BGR888"}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        frame = self.picam2.capture_array()  # Numpy array BGR888
        self.captured_image = frame.copy()   # Guarda para salvar depois
        self.capturing_live = False

        # Converte para QImage para mostrar
        qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.show_captured_image(qimage)

    def show_captured_image(self, qimage):
        if self.captured_image is not None:
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.set_border_color("#00FF00")
            self.image_label.set_image(pixmap)

    def delete_photo(self):
        if self.captured_image is None:
            return  # Não há nada para deletar

        # Criar diálogo de confirmação
        reply = QMessageBox.question(
            self,
            "Confirmar eliminação",
            "Tem a certeza que deseja eliminar a foto?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.captured_image = None
            self.capturing_live = True
            self.image_label.set_border_color("#D3D3D3")
            print("Foto eliminada.")

    def save_photo(self):
        if self.captured_image is not None:
            os.makedirs("data/raw", exist_ok=True)
            save_path = "data/raw/fba_template.jpg"

            # Converte RGB -> BGR antes de salvar
            cv2.imwrite(save_path, self.captured_image)
            print(f"Imagem guardada em {save_path}")

            # Mostrar diálogo de sucesso
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Sucesso")
            msg_box.setText(f"Imagem guardada com sucesso em:\n{save_path}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()

    def on_close(self):
        self.picam2.stop()
        self.timer.stop()
