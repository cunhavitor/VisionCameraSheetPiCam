import sys
import os
import json
import cv2
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QFrame, QWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QKeySequence, QShortcut
from picamera2 import Picamera2
from widgets.custom_widgets import LabelNumeric, ButtonMain, ImageLabel, Switch, TitleLabelMain


class CameraAdjustParamsWindow(QDialog):
    def __init__(self, parent=None, picam2=None):
        super().__init__(parent)
        self.setWindowTitle("Ajuste da PiCam")
        # Tema escuro consistente
        self.setStyleSheet("background-color: #121212; color: #f0f0f0;")
        self.resize(1400, 800)

        self.picam2 = picam2
        # Estados de UI/câmara
        self.ae_enabled = False
        self.awb_enabled = False
        self.show_grid = False
        self._dirty = False

        # Caminho do ficheiro de parâmetros
        self.params_path = os.path.join("config", "camera_params.json")
        os.makedirs("config", exist_ok=True)

        # Layout principal vertical
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Título
        main_layout.addWidget(TitleLabelMain("Ajuste da PiCam"))

        # ---------------- Parte de cima: controles + imagem ----------------
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout, 1)

        # ---------------- Painel esquerdo: controles (card) ----------------
        left_card = QFrame()
        left_card.setStyleSheet("background:#1e1e1e; border:1px solid #333333; border-radius:8px;")
        controls_layout = QVBoxLayout(left_card)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(10)
        top_layout.addWidget(left_card, 0)

        # Widgets numéricos
        self.exposure_spin = LabelNumeric("Exposure (µs)", value=20000, step=100, minimum=1000, maximum=100000, is_float=False)
        self.exposure_spin.valueChanged_connect(lambda val: self.update_camera("exposure", val))
        self.exposure_spin.setToolTip("Tempo de exposição em µs. Desative AE para editar.")

        self.gain_spin = LabelNumeric("Analogue Gain", value=1.0, step=0.1, minimum=1.0, maximum=10.0, is_float=True)
        self.gain_spin.valueChanged_connect(lambda val: self.update_camera("gain", val))
        self.gain_spin.setToolTip("Ganho analógico. Desative AE para editar.")

        self.brightness_spin = LabelNumeric("Brightness", value=0.0, step=0.1, minimum=-1.0, maximum=1.0, is_float=True)
        self.brightness_spin.valueChanged_connect(lambda val: self.update_camera("brightness", val))
        self.brightness_spin.setToolTip("Brilho global da imagem.")

        self.contrast_spin = LabelNumeric("Contrast", value=1.0, step=0.1, minimum=0.0, maximum=3.0, is_float=True)
        self.contrast_spin.valueChanged_connect(lambda val: self.update_camera("contrast", val))
        self.contrast_spin.setToolTip("Contraste global da imagem.")

        self.red_gain_spin = LabelNumeric("Red Gain", value=1.5, step=0.1, minimum=0.0, maximum=4.0, is_float=True)
        self.red_gain_spin.valueChanged_connect(lambda val: self.update_camera("red_gain", val))
        self.red_gain_spin.setToolTip("Ganho do canal R. Desative AWB para editar.")

        self.blue_gain_spin = LabelNumeric("Blue Gain", value=1.5, step=0.1, minimum=0.0, maximum=4.0, is_float=True)
        self.blue_gain_spin.valueChanged_connect(lambda val: self.update_camera("blue_gain", val))
        self.blue_gain_spin.setToolTip("Ganho do canal B. Desative AWB para editar.")

        # Switches AE/AWB e grelha
        self.switch_ae = Switch("Auto Exposure (AE)")
        self.switch_ae.setChecked(False)
        self.switch_ae.stateChanged.connect(self._toggle_ae)

        self.switch_awb = Switch("Auto White Balance (AWB)")
        self.switch_awb.setChecked(False)
        self.switch_awb.stateChanged.connect(self._toggle_awb)

        self.switch_grid = Switch("Mostrar grelha 3x3")
        self.switch_grid.setChecked(False)
        self.switch_grid.stateChanged.connect(self._toggle_grid)

        # Histograma ao vivo (toggle)
        self.switch_hist = Switch("Mostrar histograma")
        self.switch_hist.setChecked(True)
        self.switch_hist.stateChanged.connect(lambda on: self.hist_label.setVisible(bool(on)))

        controls_layout.addSpacing(10)
        for widget in [
            self.switch_ae,
            self.exposure_spin,
            self.gain_spin,
            self.brightness_spin,
            self.contrast_spin,
            self.switch_awb,
            self.red_gain_spin,
            self.blue_gain_spin,
            self.switch_grid,
            # toggle histograma
            self.switch_hist if hasattr(self, 'switch_hist') else None,
        ]:
            if widget is not None:
                controls_layout.addWidget(widget)

        # Botões
        controls_layout.addSpacing(20)
        save_button = ButtonMain("💾 Guardar Parâmetros  (S)")
        save_button.clicked.connect(self.save_params)
        controls_layout.addWidget(save_button)

        reset_button = ButtonMain("🔄 Reset Parâmetros  (R)")
        reset_button.clicked.connect(self.reset_params)
        controls_layout.addWidget(reset_button)

        capture_button = ButtonMain("📸 Capturar Foto  (C)")
        capture_button.clicked.connect(self.capture_frame)
        controls_layout.addWidget(capture_button)

        self.save_button_img = ButtonMain("💾 Guardar Foto  (G)")
        self.save_button_img.clicked.connect(self.save_frame)
        self.save_button_img.setEnabled(False)
        controls_layout.addWidget(self.save_button_img)

        resume_button = ButtonMain("▶️ Voltar ao Live  (L)")
        resume_button.clicked.connect(self.resume_live)
        controls_layout.addWidget(resume_button)

        controls_layout.addStretch()

        # ---------------- Frame direito: imagem + histograma ----------------
        right_container = QWidget()
        right_v = QVBoxLayout(right_container)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(8)

        self.image_label = ImageLabel()
        right_v.addWidget(self.image_label, 1)

        self.hist_label = QLabel()
        self.hist_label.setFixedHeight(140)
        self.hist_label.setAlignment(Qt.AlignCenter)
        self.hist_label.setStyleSheet("border:1px solid #333333; border-radius:6px; background:#0f0f0f;")
        right_v.addWidget(self.hist_label, 0)

        top_layout.addSpacing(50)
        top_layout.addWidget(right_container, 1)

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

        # Atalhos de teclado
        QShortcut(QKeySequence("S"), self, activated=self.save_params)
        QShortcut(QKeySequence("R"), self, activated=self.reset_params)
        QShortcut(QKeySequence("C"), self, activated=self.capture_frame)
        QShortcut(QKeySequence("G"), self, activated=self.save_frame)
        QShortcut(QKeySequence("L"), self, activated=self.resume_live)
        QShortcut(QKeySequence("A"), self, activated=lambda: self.switch_ae.setChecked(not self.switch_ae.isChecked()))
        QShortcut(QKeySequence("W"), self, activated=lambda: self.switch_awb.setChecked(not self.switch_awb.isChecked()))
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)

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
                "ColourGains": [1.0, 1.0],
                "AeEnable": False,
                "AwbEnable": False
            }

        self.exposure_spin.setValue(params.get("ExposureTime", 10000))
        self.gain_spin.setValue(params.get("AnalogueGain", 1.0))
        self.brightness_spin.setValue(params.get("Brightness", 0.0))
        self.contrast_spin.setValue(params.get("Contrast", 1.0))
        self.red_gain_spin.setValue(params.get("ColourGains", [1.0, 1.0])[0])
        self.blue_gain_spin.setValue(params.get("ColourGains", [1.0, 1.0])[1])

        # Atualiza AE/AWB a partir do JSON sem disparar sinais duplicados
        ae = bool(params.get("AeEnable", False))
        awb = bool(params.get("AwbEnable", False))
        if hasattr(self, 'switch_ae'):
            self.switch_ae.blockSignals(True)
            self.switch_ae.setChecked(ae)
            self.switch_ae.blockSignals(False)
            self.ae_enabled = ae
        if hasattr(self, 'switch_awb'):
            self.switch_awb.blockSignals(True)
            self.switch_awb.setChecked(awb)
            self.switch_awb.blockSignals(False)
            self.awb_enabled = awb

        # Aplicar estado AE/AWB e reenviar valores atuais
        self._apply_ae_awb_state()
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
        # Atualiza UI com valores automáticos atuais (se AE/AWB ativos)
        try:
            md = self.picam2.capture_metadata()
            self._update_ui_from_metadata(md)
        except Exception:
            md = None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pix = self._to_pixmap_with_overlay(frame)
        self.image_label.set_image(pix)
        # Atualiza histograma se ativo
        if hasattr(self, 'switch_hist') and self.switch_hist.isChecked():
            self._update_histogram(frame)

    def update_camera(self, param, value):
        value = float(value)
        self._mark_dirty()
        if param == "exposure":
            if not self.ae_enabled:
                self.safe_set_controls({"AeEnable": False, "ExposureTime": int(value)})
        elif param == "gain":
            if not self.ae_enabled:
                self.safe_set_controls({"AnalogueGain": value})
        elif param == "brightness":
            self.safe_set_controls({"Brightness": value})
        elif param == "contrast":
            self.safe_set_controls({"Contrast": value})
        elif param == "red_gain":
            if not self.awb_enabled:
                self.safe_set_controls({"AwbEnable": False, "ColourGains": (value, self.blue_gain_spin.value())})
        elif param == "blue_gain":
            if not self.awb_enabled:
                self.safe_set_controls({"AwbEnable": False, "ColourGains": (self.red_gain_spin.value(), value)})

    def save_params(self):
        # Quando AE/AWB estão ligados, queremos guardar os valores automáticos atuais.
        md = {}
        try:
            md = self.picam2.capture_metadata() or {}
        except Exception:
            md = {}

        exp_value = int(self.exposure_spin.value())
        gain_value = float(self.gain_spin.value())
        if self.ae_enabled:
            exp_value = int(md.get("ExposureTime", exp_value))
            try:
                gain_value = float(md.get("AnalogueGain", gain_value))
            except Exception:
                pass

        red_value = float(self.red_gain_spin.value())
        blue_value = float(self.blue_gain_spin.value())
        if self.awb_enabled:
            cg = md.get("ColourGains")
            if isinstance(cg, (list, tuple)) and len(cg) == 2:
                try:
                    red_value = float(cg[0])
                    blue_value = float(cg[1])
                except Exception:
                    pass

        params = {
            "ExposureTime": exp_value,
            "AnalogueGain": gain_value,
            "Brightness": float(self.brightness_spin.value()),
            "Contrast": float(self.contrast_spin.value()),
            "ColourGains": [red_value, blue_value],
            "AeEnable": bool(self.ae_enabled),
            "AwbEnable": bool(self.awb_enabled)
        }
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"[INFO] Parâmetros guardados em {self.params_path}")
        self.status_label.setText("[INFO] Parâmetros guardados com sucesso.")
        self._dirty = False

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
                "ColourGains": [1.0, 1.0],
                "AeEnable": False,
                "AwbEnable": False
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
        pix = self._to_pixmap_with_overlay(frame)
        self.image_label.set_image(pix)
        self.live_mode = False
        self.status_label.setText("[CAPTURA] Frame congelado.")
        self.save_button_img.setEnabled(True)
        if hasattr(self, 'switch_hist') and self.switch_hist.isChecked():
            self._update_histogram(frame)

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
            self.save_button_img.setEnabled(False)
        else:
            self.status_label.setText("[WARN] Nenhum frame capturado para guardar.")

    def resume_live(self):
        if not self.timer.isActive():
            self.timer.start(50)
            self.live_mode = True
            self.status_label.setText("[LIVE] Stream retomado.")
            self.save_button_img.setEnabled(False)

    def closeEvent(self, event):
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        if self.picam2:
            self.picam2.stop()
        event.accept()

    # ---------------- Helpers e estados ----------------
    def _toggle_ae(self, enabled: bool):
        self.ae_enabled = bool(enabled)
        self._apply_ae_awb_state()
        # Desativar/ativar controlos relacionados
        self.exposure_spin.setEnabled(not self.ae_enabled)
        self.gain_spin.setEnabled(not self.ae_enabled)
        self.status_label.setText(f"[INFO] AE {'ativado' if self.ae_enabled else 'desativado'}.")
        # Atualiza UI com valores atuais do AE para refletir o que a câmara está a usar
        try:
            md = self.picam2.capture_metadata()
            self._update_ui_from_metadata(md)
        except Exception:
            pass

    def _toggle_awb(self, enabled: bool):
        self.awb_enabled = bool(enabled)
        self._apply_ae_awb_state()
        self.red_gain_spin.setEnabled(not self.awb_enabled)
        self.blue_gain_spin.setEnabled(not self.awb_enabled)
        self.status_label.setText(f"[INFO] AWB {'ativado' if self.awb_enabled else 'desativado'}.")
        # Atualiza UI com valores atuais do AWB para refletir o que a câmara está a usar
        try:
            md = self.picam2.capture_metadata()
            self._update_ui_from_metadata(md)
        except Exception:
            pass

    def _apply_ae_awb_state(self):
        # Aplica estado aos controlos da câmera
        self.safe_set_controls({"AeEnable": bool(self.ae_enabled)})
        self.safe_set_controls({"AwbEnable": bool(self.awb_enabled)})
        # Se AE/AWB estiverem off, reenvia manuais atuais
        if not self.ae_enabled:
            self.safe_set_controls({
                "ExposureTime": int(self.exposure_spin.value()),
                "AnalogueGain": float(self.gain_spin.value()),
            })
        if not self.awb_enabled:
            self.safe_set_controls({
                "ColourGains": (float(self.red_gain_spin.value()), float(self.blue_gain_spin.value()))
            })

    def _toggle_grid(self, enabled: bool):
        self.show_grid = bool(enabled)
        self.status_label.setText(f"[INFO] Grelha {'ativada' if self.show_grid else 'desativada'}.")

    def _to_pixmap_with_overlay(self, frame_rgb):
        """Converte numpy RGB em QPixmap e desenha grelha 3x3 opcional."""
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, int(frame_rgb.strides[0]), QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if self.show_grid:
            pix = pix.copy()
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(255, 255, 255, 120))
            pen.setWidth(2)
            painter.setPen(pen)
            # 3x3 grid (regra dos terços)
            x1 = w // 3
            x2 = 2 * w // 3
            y1 = h // 3
            y2 = 2 * h // 3
            painter.drawLine(x1, 0, x1, h)
            painter.drawLine(x2, 0, x2, h)
            painter.drawLine(0, y1, w, y1)
            painter.drawLine(0, y2, w, y2)
            painter.end()
        return pix

    def _update_histogram(self, frame_rgb):
        try:
            import numpy as np
            hist_h, hist_w = 120, 256
            canvas = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
            # R, G, B channels
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR for cv2
            for ch in range(3):
                hist = cv2.calcHist([frame_rgb], [ch], None, [256], [0, 256]).flatten()
                if hist.max() > 0:
                    hist = hist / hist.max()
                hist = (hist * (hist_h - 10)).astype(int)
                for x in range(1, 256):
                    y1 = hist_h - 1 - hist[x - 1]
                    y2 = hist_h - 1 - hist[x]
                    cv2.line(canvas, (x - 1, y1), (x, y2), colors[ch], 1)
            qimg = QImage(canvas.data, hist_w, hist_h, int(canvas.strides[0]), QImage.Format_RGB888)
            self.hist_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self.status_label.setText("[EDIT] Alterações não guardadas (S para guardar).")

    # ---------------- UI sync helpers ----------------
    def _set_spin_quietly(self, spin: LabelNumeric, value):
        try:
            spin.spinbox.blockSignals(True)
            spin.setValue(value)
        finally:
            spin.spinbox.blockSignals(False)

    def _update_ui_from_metadata(self, md: dict):
        if not isinstance(md, dict):
            return
        # Se AE ativo, refletir ExposureTime e AnalogueGain automáticos
        if self.ae_enabled:
            if "ExposureTime" in md:
                self._set_spin_quietly(self.exposure_spin, int(md["ExposureTime"]))
            if "AnalogueGain" in md:
                try:
                    self._set_spin_quietly(self.gain_spin, float(md["AnalogueGain"]))
                except Exception:
                    pass
        # Se AWB ativo, refletir ColourGains automáticos
        if self.awb_enabled:
            cg = md.get("ColourGains")
            if isinstance(cg, (list, tuple)) and len(cg) == 2:
                try:
                    self._set_spin_quietly(self.red_gain_spin, float(cg[0]))
                    self._set_spin_quietly(self.blue_gain_spin, float(cg[1]))
                except Exception:
                    pass
