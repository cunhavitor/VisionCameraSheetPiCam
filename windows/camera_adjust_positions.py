import sys
import os
import json
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QTextEdit, QFrame
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from config.utils import load_params, center_window

INSPECTION_PREVIEW_WIDTH = 640
INSPECTION_PREVIEW_HEIGHT = 480

class CameraAdjustPosition(QDialog):
    def __init__(self, parent=None, picam2=None):
        super().__init__()
        self.setWindowTitle("Verificacao do angulo da Camera")
        self.resize(1600, 1100)

        # Estado
        self.picam2 = picam2
        self.capturing_live = True
        self.captured_image = None
        self.last_frame = None

        # Carregar parametros
        self.param_path = "config/config_lines_align_camera.json"
        params = load_params(self.param_path)
        self.line_top = params.get("line_top", 10)
        self.line_bottom = params.get("line_bottom", 90)
        self.line_left = params.get("line_left", 10)
        self.line_right = params.get("line_right", 90)
        self.gaussian_blur = params.get("gaussian_blur", 7)
        self.canny_threshold1 = params.get("canny_threshold1", 30)
        self.canny_threshold2 = params.get("canny_threshold2", 120)

        self.auto_line_top = self.line_top
        self.auto_line_bottom = self.line_bottom
        self.auto_line_left = self.line_left
        self.auto_line_right = self.line_right

        # Layout principal
        main_layout = QHBoxLayout(self)

        # Painel esquerdo: sliders e botoes
        left_frame = QVBoxLayout()
        main_layout.addLayout(left_frame, 1)

        # Sliders
        self.sliders = {}
        for name, value, minv, maxv in [
            ("Linha Topo", self.line_top, 0, 100),
            ("Linha Fundo", self.line_bottom, 0, 100),
            ("Linha Esquerda", self.line_left, 0, 100),
            ("Linha Direita", self.line_right, 0, 100),
            ("Gaussian Blur", self.gaussian_blur, 1, 25),
            ("Canny Th1", self.canny_threshold1, 1, 255),
            ("Canny Th2", self.canny_threshold2, 1, 255)
        ]:
            lbl = QLabel(f"{name}: {value}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(minv)
            slider.setMaximum(maxv)
            slider.setValue(value)
            slider.valueChanged.connect(lambda v, l=lbl, n=name: l.setText(f"{n}: {v}"))
            left_frame.addWidget(lbl)
            left_frame.addWidget(slider)
            self.sliders[name] = slider

        # Botoes captura
        self.capture_btn = QPushButton("Capturar Foto")
        self.capture_btn.clicked.connect(self.capture_photo)
        self.save_btn = QPushButton("Guardar Foto")
        self.save_btn.clicked.connect(self.save_photo)
        self.delete_btn = QPushButton("Eliminar Foto")
        self.delete_btn.clicked.connect(self.delete_photo)
        self.verify_btn = QPushButton("Verificar Perspectiva")
        self.verify_btn.clicked.connect(self.verify_alignment)
        for btn in [self.capture_btn, self.save_btn, self.delete_btn, self.verify_btn]:
            left_frame.addWidget(btn)

        # Caixa de texto
        self.textbox = QTextEdit()
        left_frame.addWidget(self.textbox)

        # Painel direito: imagem
        self.image_label = QLabel()
        self.image_label.setFixedSize(1000, 900)
        self.image_label.setFrameShape(QFrame.Shape.Box)
        main_layout.addWidget(self.image_label, 2)

        # Configurar camera Picamera2
        camera_config = self.picam2.create_still_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()

        # Timer para atualizar frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        if self.capturing_live:
            frame = self.picam2.capture_array()
            frame = cv2.resize(frame, (self.image_label.width(), self.image_label.height()))
            self.last_frame = frame.copy()

            # Desenhar linhas sliders (verde)
            h, w = frame.shape[:2]
            line_top = int(h * (self.sliders["Linha Topo"].value() / 100))
            line_bottom = int(h * (self.sliders["Linha Fundo"].value() / 100))
            line_left = int(w * (self.sliders["Linha Esquerda"].value() / 100))
            line_right = int(w * (self.sliders["Linha Direita"].value() / 100))
            cv2.line(frame, (0, line_top), (w, line_top), (0, 255, 0), 2)
            cv2.line(frame, (0, line_bottom), (w, line_bottom), (0, 255, 0), 2)
            cv2.line(frame, (line_left, 0), (line_left, h), (0, 255, 0), 2)
            cv2.line(frame, (line_right, 0), (line_right, h), (0, 255, 0), 2)

            # Linhas auto (azul)
            auto_top = int(h * self.auto_line_top / 100)
            auto_bottom = int(h * self.auto_line_bottom / 100)
            auto_left = int(w * self.auto_line_left / 100)
            auto_right = int(w * self.auto_line_right / 100)
            cv2.line(frame, (0, auto_top), (w, auto_top), (255, 0, 0), 2)
            cv2.line(frame, (0, auto_bottom), (w, auto_bottom), (255, 0, 0), 2)
            cv2.line(frame, (auto_left, 0), (auto_left, h), (255, 0, 0), 2)
            cv2.line(frame, (auto_right, 0), (auto_right, h), (255, 0, 0), 2)

            # Mostrar frame
            self.show_frame(frame)

    def show_frame(self, frame):
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch*w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def capture_photo(self):
        self.capturing_live = False
        frame = self.picam2.capture_array()
        self.captured_image = frame
        self.show_frame(frame)

    def delete_photo(self):
        self.capturing_live = True
        self.captured_image = None

    def save_photo(self):
        if self.captured_image is not None:
            os.makedirs("data/raw", exist_ok=True)
            path = "data/raw/fba_template.jpg"
            cv2.imwrite(path, self.captured_image)
            print(f"Imagem guardada em {path}")

    def verify_alignment(self):
        if self.last_frame is None:
            return
        frame = self.last_frame.copy()
        self.textbox.clear()

        k = self.sliders["Gaussian Blur"].value()
        k = max(3, k + 1 if k % 2 == 0 else k)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(blurred,
                          self.sliders["Canny Th1"].value(),
                          self.sliders["Canny Th2"].value())
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

        if not filtered:
            self.textbox.append("? Nenhum contorno valido encontrado")
            return

        largest = max(filtered, key=cv2.contourArea)
        all_points = np.vstack(filtered)
        rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rect).astype(int)

        xs, ys = box[:, 0], box[:, 1]
        h, w = frame.shape[:2]
        self.auto_line_left = int(xs.min() / w * 100)
        self.auto_line_right = int(xs.max() / w * 100)
        self.auto_line_top = int(ys.min() / h * 100)
        self.auto_line_bottom = int(ys.max() / h * 100)
        self.textbox.append(
            f"? Linhas auto: Top={self.auto_line_top}%, Bottom={self.auto_line_bottom}%, "
            f"Left={self.auto_line_left}%, Right={self.auto_line_right}%"
        )

        # Desenhar contorno
        output = frame.copy()
        cv2.drawContours(output, [largest], -1, (0, 255, 0), 2)
        hull = cv2.convexHull(largest)
        pts = hull.reshape(-1, 2)
        for i in range(len(pts)):
            cv2.line(output, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 0, 255), 2)

        self.show_frame(output)
