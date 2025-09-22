import sys
import os
import json
import cv2
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from picamera2 import Picamera2
from widgets.custom_widgets import LabelNumeric, ButtonMain, ImageLabel


class CameraAdjustParamsWindow(QDialog):
    def __init__(self, parent=None, picam2=None):
        super().__init__(parent)
        self.setWindowTitle("Ajuste da PiCam")
        self.setFixedSize(1400, 700)

        # Centralizar a janela
        screen = self.screen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

        self.picam2 = picam2

        # Caminho do ficheiro de parâmetros
        self.params_path = os.path.join("config", "camera_params.json")
        os.makedirs("config", exist_ok=True)

        # Layout principal vertical
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30,30,30,30)

        # ---------------- Parte de cima: controles + imagem ----------------
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout, 1)

        # ---------------- Frame esquerdo: controles ----------------
        controls_layout = QVBoxLayout()
        top_layout.addLayout(controls_layout, 0)

        # Widgets numéricos
        self.exposure_spin = LabelNumeric("Exposure (µs)", value=20000, step=100, minimum=1000, maximum=100000, is_float=False)
        self.exposure_spin.valueChanged_connect(lambda val: self.update_camera("exposure", val))

        self.gain_spin = LabelNumeric("Analogue Gain", value=1.0, step=0.1, minimum=1.0, maximum=10.0, is_float=True)
        self.gain_spin.valueChanged_connect(lambda val: self.update_camera("gain", val))

        self.brightness_spin = LabelNumeric("Brightness", value=0.0, step=0.1, minimum=-1.0, maximum=1.0, is_float=True)
        self.brightness_spin.valueChanged_connect(lambda val: self.update_camera("brightness", val))

        self.contrast_spin = LabelNumeric("Contrast", value=1.0, step=0.1, minimum=0.0, maximum=3.0, is_float=True)
        self.contrast_spin.valueChanged_connect(lambda val: self.update_camera("contrast", val))

        self.red_gain_spin = LabelNumeric("Red Gain", value=1.5, step=0.1, minimum=0.0, maximum=4.0, is_float=True)
        self.red_gain_spin.valueChanged_connect(lambda val: self.update_camera("red_gain", val))

        self.blue_gain_spin = LabelNumeric("Blue Gain", value=1.5, step=0.1, minimum=0.0, maximum=4.0, is_float=True)
        self.blue_gain_spin.valueChanged_connect(lambda val: self.update_camera("blue_gain", val))

        controls_layout.addSpacing(30)
        for widget in [
            self.exposure_spin, self.gain_spin, self.brightness_spin,
            self.contrast_spin, self.red_gain_spin, self.blue_gain_spin
        ]:
            controls_layout.addWidget(widget)

        # Botões
        controls_layout.addSpacing(30)
        save_button = ButtonMain("💾 Guardar Parâmetros")
        save_button.clicked.connect(self.save_params)
        controls_layout.addWidget(save_button)

        reset_button = ButtonMain("🔄 Reset Parâmetros")
        reset_button.clicked.connect(self.reset_params)
        controls_layout.addWidget(reset_button)

        capture_button = ButtonMain("📸 Capturar Foto")
        capture_button.clicked.connect(self.capture_frame)
        controls_layout.addWidget(capture_button)

        save_button_img = ButtonMain("💾 Guardar Foto")
        save_button_img.clicked.connect(self.save_frame)
        controls_layout.addWidget(save_button_img)

        resume_button = ButtonMain("▶️ Voltar ao Live")
        resume_button.clicked.connect(self.resume_live)
        controls_layout.addWidget(resume_button)

        controls_layout.addStretch()

        # ---------------- Frame direito: imagem ----------------
        self.image_label = ImageLabel()
        top_layout.addSpacing(50)
        top_layout.addWidget(self.image_label, 1)

        # ---------------- Rodapé: status ----------------
        self.status_label = QLabel("[INFO] Modo Live ativo")
        self.status_label.setAlignment(Qt.AlignCenter)

        # aumenta o tamanho da letra sem mexer no tema global
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
            }
        """)

        main_layout.addWidget(self.status_label)

        # Inicializa Picamera2
        self.preview_config = self.picam2.create_still_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.fullres_config = self.picam2.create_still_configuration(
            main={"format": "RGB888"}
        )

        self.picam2.configure(self.preview_config)
        self.update_camera_params()
        self.picam2.start()

        # Timer para atualizar imagem
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(50)  # ~20 FPS

        self.running = True
        self.live_mode = True

    # ---------------- Métodos ----------------
    def safe_set_controls(self, controls):
        try:
            self.picam2.set_controls(controls)
        except Exception as e:
            print(f"[WARN] Não foi possível aplicar {controls}: {e}")

    def update_camera_params(self):
        try:
            with open(self.params_path, "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {
                "ExposureTime": 10000,
                "AnalogueGain": 1.0,
                "Brightness": 0.0,
                "Contrast": 1.0,
                "ColourGains": [1.0, 1.0]
            }

        self.exposure_spin.setValue(params.get("ExposureTime", 10000))
        self.gain_spin.setValue(params.get("AnalogueGain", 1.0))
        self.brightness_spin.setValue(params.get("Brightness", 0.0))
        self.contrast_spin.setValue(params.get("Contrast", 1.0))
        self.red_gain_spin.setValue(params.get("ColourGains", [1.0, 1.0])[0])
        self.blue_gain_spin.setValue(params.get("ColourGains", [1.0, 1.0])[1])

        self.update_camera("exposure", self.exposure_spin.value())
        self.update_camera("gain", self.gain_spin.value())
        self.update_camera("brightness", self.brightness_spin.value())
        self.update_camera("contrast", self.contrast_spin.value())
        self.update_camera("red_gain", self.red_gain_spin.value())
        self.update_camera("blue_gain", self.blue_gain_spin.value())

    def update_image(self):
        if not self.live_mode:
            return
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.image_label.set_image(QPixmap.fromImage(image))

    def update_camera(self, param, value):
        value = float(value)
        if param == "exposure":
            self.safe_set_controls({"AeEnable": False, "ExposureTime": int(value)})
        elif param == "gain":
            self.safe_set_controls({"AnalogueGain": value})
        elif param == "brightness":
            self.safe_set_controls({"Brightness": value})
        elif param == "contrast":
            self.safe_set_controls({"Contrast": value})
        elif param == "red_gain":
            self.safe_set_controls({"AwbEnable": False, "ColourGains": (value, self.blue_gain_spin.value())})
        elif param == "blue_gain":
            self.safe_set_controls({"AwbEnable": False, "ColourGains": (self.red_gain_spin.value(), value)})

    def save_params(self):
        params = {
            "ExposureTime": int(self.exposure_spin.value()),
            "AnalogueGain": float(self.gain_spin.value()),
            "Brightness": float(self.brightness_spin.value()),
            "Contrast": float(self.contrast_spin.value()),
            "ColourGains": [float(self.red_gain_spin.value()), float(self.blue_gain_spin.value())]
        }
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"[INFO] Parâmetros guardados em {self.params_path}")
        self.status_label.setText("[INFO] Parâmetros guardados com sucesso.")

    def reset_params(self):
        reply = QMessageBox.question(
            self,
            "Confirmar Reset",
            "Tens a certeza que queres repor os parâmetros padrão?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            defaults = {
                "ExposureTime": 10000,
                "AnalogueGain": 1.0,
                "Brightness": 0.0,
                "Contrast": 1.0,
                "ColourGains": [1.0, 1.0]
            }
            with open(self.params_path, "w") as f:
                json.dump(defaults, f, indent=4)
            print("[INFO] Reset para parâmetros padrão")
            self.status_label.setText("[INFO] Parâmetros resetados para padrão.")
            self.update_camera_params()
        else:
            self.status_label.setText("[INFO] Reset cancelado.")

    def capture_frame(self):
        if self.timer.isActive():
            self.timer.stop()
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.captured_frame = frame.copy()
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.image_label.set_image(QPixmap.fromImage(image))
        self.live_mode = False
        self.status_label.setText("[CAPTURA] Frame congelado.")

    def save_frame(self):
        if hasattr(self, "captured_frame") and self.captured_frame is not None:
            os.makedirs("data/raw", exist_ok=True)
            save_path = os.path.join("data/raw", "fba_template.jpg")

            self.picam2.stop()
            self.picam2.configure(self.fullres_config)
            self.picam2.start()
            frame_full = self.picam2.capture_array()
            frame_full = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path, cv2.cvtColor(frame_full, cv2.COLOR_RGB2BGR))
            print(f"[INFO] Foto guardada em alta resolução: {save_path}")
            self.status_label.setText(f"[INFO] Foto guardada em {save_path}")

            self.picam2.stop()
            self.picam2.configure(self.preview_config)
            self.picam2.start()
        else:
            self.status_label.setText("[WARN] Nenhum frame capturado para guardar.")

    def resume_live(self):
        if not self.timer.isActive():
            self.timer.start(50)
            self.live_mode = True
            self.status_label.setText("[LIVE] Stream retomado.")

    def closeEvent(self, event):
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        if self.picam2:
            self.picam2.stop()
        event.accept()
