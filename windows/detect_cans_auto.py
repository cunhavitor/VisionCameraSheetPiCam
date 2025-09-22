import sys
import json
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSlider, QSpinBox, QDoubleSpinBox, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from config.utils import load_params


class AutoDetectCans(QDialog):
    def __init__(self, parent=None, image_path=None):
        super().__init__(parent)
        self.setWindowTitle("Detecção de Latas")
        self.resize(1200, 900)

        self.image_path = image_path
        self.img_cv = cv2.imread(image_path)
        self.canvas_width = 1000
        self.canvas_height = 900

        # params carregados
        self.param_path = "config/config_auto_detect_cans_params.json"
        params = load_params(self.param_path)

        self.gaussian = params.get("gaussian", 10)
        self.treshold1 = params.get("treshold1", 90)
        self.treshold2 = params.get("treshold2", 10)
        self.kernell = params.get("kernell", 90)
        self.area_min = params.get("area_min", 10)
        self.area_max = params.get("area_max", 100000)
        self.circularity_min = params.get("circularity_min", 0.1)

        # layout principal
        main_layout = QHBoxLayout(self)

        # painel de controles
        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout, 0)

        # sliders
        self.gaussian_spin = self.create_spinbox(controls_layout, "Gaussian Blur", 1, 99, self.gaussian, int)
        self.t1_spin = self.create_spinbox(controls_layout, "Threshold1", 0, 255, self.treshold1, int)
        self.t2_spin = self.create_spinbox(controls_layout, "Threshold2", 0, 255, self.treshold2, int)
        self.kernel_spin = self.create_spinbox(controls_layout, "Kernel", 1, 255, self.kernell, int)
        self.area_min_spin = self.create_spinbox(controls_layout, "Área Min", 1, 100000, self.area_min, int)
        self.area_max_spin = self.create_spinbox(controls_layout, "Área Max", 1, 200000, self.area_max, int)
        self.circ_spin = self.create_spinbox(controls_layout, "Circularidade Min", 0.0, 1.0, self.circularity_min, float)

        # botões
        btn_update = QPushButton("Atualizar Detecção")
        btn_update.clicked.connect(self.update_canvas)
        controls_layout.addWidget(btn_update)

        btn_steps = QPushButton("Mostrar Etapas")
        btn_steps.clicked.connect(self.mostrar_etapas_processamento)
        controls_layout.addWidget(btn_steps)

        controls_layout.addStretch()

        # área da imagem
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label, 1)

        # render inicial
        self.update_canvas()

    def create_spinbox(self, layout, label_text, min_val, max_val, default, val_type):
        """Cria um spinbox numérico ligado ao update"""
        container = QVBoxLayout()
        label = QLabel(label_text)
        container.addWidget(label)

        if val_type == float:
            spin = QDoubleSpinBox()
            spin.setSingleStep(0.1)
        else:
            spin = QSpinBox()
            spin.setSingleStep(1)

        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.valueChanged.connect(self.update_canvas)

        container.addWidget(spin)
        layout.addLayout(container)
        return spin

    def update_canvas(self):
        try:
            img, latas = self.detectar_latas()
        except Exception as e:
            print("Erro na detecção:", e)
            return

        output = img.copy()
        for cnt in latas:
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.canvas_width, self.canvas_height, Qt.KeepAspectRatio
        ))

        self._save_params()

    def detectar_latas(self):
        img = cv2.resize(self.img_cv, (self.canvas_width, self.canvas_height))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = int(self.gaussian_spin.value())
        blurred = cv2.GaussianBlur(gray, (g if g % 2 else g + 1, g if g % 2 else g + 1), 0)

        edges = cv2.Canny(blurred, self.t1_spin.value(), self.t2_spin.value())

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_spin.value(), self.kernel_spin.value()))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        separated = cv2.erode(closed, kernel_erode, iterations=2)

        contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.area_min_spin.value() or area > self.area_max_spin.value():
                continue

            if len(cnt) < 5:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            aspect_ratio = min(MA, ma) / max(MA, ma)

            if aspect_ratio < 0.5 or aspect_ratio > 1.0:
                continue

            detected.append(cnt)

        return img, detected

    def mostrar_etapas_processamento(self):
        img = cv2.resize(self.img_cv, (self.canvas_width, self.canvas_height))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 10, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        closed_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

        width = 400
        gray_bgr = cv2.resize(gray_bgr, (width, 300))
        blurred_bgr = cv2.resize(blurred_bgr, (width, 300))
        edges_bgr = cv2.resize(edges_bgr, (width, 300))
        closed_bgr = cv2.resize(closed_bgr, (width, 300))

        resultado = np.hstack((gray_bgr, blurred_bgr, edges_bgr, closed_bgr))

        cv2.imshow("Etapas do Processamento (Gray → Blur → Canny → Morph)", resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self._save_params()

    def _save_params(self):
        params = {
            "gaussian": int(self.gaussian_spin.value()),
            "treshold1": int(self.t1_spin.value()),
            "treshold2": int(self.t2_spin.value()),
            "kernell": int(self.kernel_spin.value()),
            "area_min": int(self.area_min_spin.value()),
            "area_max": int(self.area_max_spin.value()),
            "circularity_min": float(self.circ_spin.value())
        }
        with open(self.param_path, "w") as f:
            json.dump(params, f, indent=4)
