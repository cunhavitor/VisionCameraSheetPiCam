import os
import json
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QSpacerItem, QSizePolicy, QSlider, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage, QColor
from PIL import Image, ImageDraw, ImageQt
from picamera2 import Picamera2

from config.config import PREVIEW_WIDTH, PREVIEW_HEIGHT


class AlignmentWindow(QDialog):
    def __init__(self, parent=None, picam2=None, output_path="data/mask/leaf_mask.png"):
        super().__init__(parent)
        self.setWindowTitle("Ajustar Alinhamento")
        self.setFixedSize(1400, 700)
        self.picam2 = picam2
        self.output_path = output_path

        self.config_path = "config/config_alignment.json"
        self._load_alignment_config()

        # --- Layout principal ---
        main_layout = QHBoxLayout(self)

        # Frame lateral
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        main_layout.addWidget(self.left_widget, 0)

        # Frame de imagem
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        main_layout.addWidget(self.right_widget, 1)

        self.image_label = QLabel()
        self.image_label.setFixedSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.right_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # --- Entradas e sliders ---
        self.max_features_entry = self._add_label_lineedit("max_features", str(self.alignment_config["max_features"]))
        self.good_match_percent_entry = self._add_label_lineedit("good_match_percent", str(self.alignment_config["good_match_percent"]))

        self.x_min_slider = self._add_slider("x_min", 0, PREVIEW_WIDTH, self.alignment_config["x_min"])
        self.x_max_slider = self._add_slider("x_max", 0, PREVIEW_WIDTH, self.alignment_config["x_max"])
        self.y_min_slider = self._add_slider("y_min", 0, PREVIEW_HEIGHT, self.alignment_config["y_min"])
        self.y_max_slider = self._add_slider("y_max", 0, PREVIEW_HEIGHT, self.alignment_config["y_max"])

        self.sheet_xDim_entry = self._add_label_lineedit("Sheet X dim (mm)", "1030")
        self.sheet_yDim_entry = self._add_label_lineedit("Sheet Y dim (mm)", "820")

        self.density_label = QLabel("Densidade: —")
        self.left_layout.addWidget(self.density_label)

        # Botões
        self.update_btn = QPushButton("Atualizar Alinhamento")
        self.update_btn.clicked.connect(self._update_alignment)
        self.left_layout.addWidget(self.update_btn)

        self.save_btn = QPushButton("Guardar Configuração")
        self.save_btn.clicked.connect(self._save_alignment_config)
        self.left_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("Reset Padrão")
        self.reset_btn.setStyleSheet("background-color: #ff4444; color: white;")
        self.reset_btn.clicked.connect(self._reset_defaults)
        self.left_layout.addWidget(self.reset_btn)

        self.left_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Status rodapé
        self.status_label = QLabel("[INFO] A aguardar inicialização...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: cyan;")
        main_layout.addWidget(self.status_label, 0, alignment=Qt.AlignBottom)

        # Carrega máscara
        self._initialize_mask()

        # Start live camera
        if self.picam2:
            self.start_camera_preview()
        else:
            self._update_frame_placeholder()

    def _add_label_lineedit(self, label_text, default=""):
        lbl = QLabel(label_text)
        self.left_layout.addWidget(lbl)
        le = QLineEdit()
        le.setText(default)
        self.left_layout.addWidget(le)
        return le

    def _add_slider(self, label_text, min_val, max_val, default_val):
        lbl = QLabel(f"{label_text}: {default_val}")
        self.left_layout.addWidget(lbl)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(lambda v, l=lbl, t=label_text: l.setText(f"{t}: {v}"))
        self.left_layout.addWidget(slider)
        return slider

    def _load_alignment_config(self):
        try:
            with open(self.config_path, "r") as f:
                self.alignment_config = json.load(f)
        except Exception:
            self.alignment_config = {
                "max_features": 1000,
                "good_match_percent": 0.2,
                "x_min": 50,
                "x_max": PREVIEW_WIDTH - 50,
                "y_min": 50,
                "y_max": PREVIEW_HEIGHT - 50
            }

    def _initialize_mask(self):
        mask_img = cv2.imread(self.output_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            self.mask_resized = cv2.resize(mask_img, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_NEAREST)
        else:
            self.mask_resized = np.ones((PREVIEW_HEIGHT, PREVIEW_WIDTH), dtype=np.uint8) * 255

    def start_camera_preview(self):
        try:
            config = self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (PREVIEW_WIDTH, PREVIEW_HEIGHT)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._update_frame)
            self.timer.start(30)  # ~30 FPS
            self.status_label.setText("[INFO] Live ativo")
        except Exception as e:
            self.status_label.setText(f"[ERRO] Falha na câmera: {e}")

    def _update_frame(self):
        frame = self.picam2.capture_array()
        if frame is None:
            return
        self._process_frame(frame)

    def _update_frame_placeholder(self):
        dummy = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
        self._process_frame(dummy)

    def _process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        x_min, x_max = self.x_min_slider.value(), self.x_max_slider.value()
        y_min, y_max = self.y_min_slider.value(), self.y_max_slider.value()

        # retângulo semi-transparente verde
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 255, 0, 255), width=3, fill=(0, 255, 0, 60))

        # sobrepor máscara
        mask_rgba = Image.fromarray(self.mask_resized).convert("L").point(lambda p: 120 if p > 0 else 0)
        mask_colored = Image.new("RGBA", pil_img.size, (255, 0, 0, 80))
        mask_colored.putalpha(mask_rgba)

        pil_img = Image.alpha_composite(pil_img, mask_colored)


        # cálculo densidade
        try:
            px_w, px_h = x_max - x_min, y_max - y_min
            mm_w = float(self.sheet_xDim_entry.text())
            mm_h = float(self.sheet_yDim_entry.text())
            pix_per_mm2 = (px_w * px_h) / (mm_w * mm_h)
            self.density_label.setText(f"Densidade: {pix_per_mm2:.2f} px/mm²")
        except Exception:
            self.density_label.setText("Densidade: -")

        # render final
        final_img = Image.alpha_composite(pil_img, overlay)
        qt_img = ImageQt.ImageQt(final_img.convert("RGB"))
        pixmap = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap)

    def _update_alignment(self):
        self._initialize_mask()
        self.status_label.setText("[INFO] Máscara atualizada")

    def _save_alignment_config(self):
        try:
            self.alignment_config["max_features"] = int(self.max_features_entry.text())
            self.alignment_config["good_match_percent"] = float(self.good_match_percent_entry.text())
            self.alignment_config["x_min"] = self.x_min_slider.value()
            self.alignment_config["x_max"] = self.x_max_slider.value()
            self.alignment_config["y_min"] = self.y_min_slider.value()
            self.alignment_config["y_max"] = self.y_max_slider.value()
            with open(self.config_path, "w") as f:
                json.dump(self.alignment_config, f, indent=4)
            QMessageBox.information(self, "Sucesso", "Configuração guardada com sucesso.")
            self.status_label.setText("[INFO] Configuração guardada")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao guardar configuração: {e}")

    def _reset_defaults(self):
        self.x_min_slider.setValue(50)
        self.x_max_slider.setValue(PREVIEW_WIDTH - 50)
        self.y_min_slider.setValue(50)
        self.y_max_slider.setValue(PREVIEW_HEIGHT - 50)
        self.max_features_entry.setText("1000")
        self.good_match_percent_entry.setText("0.2")
        self.status_label.setText("[INFO] Reset aplicado")

    def closeEvent(self, event):
        if self.picam2:
            try:
                self.timer.stop()
                self.picam2.stop()
            except Exception:
                pass
        event.accept()
