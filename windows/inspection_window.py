import os
import json
import time
import csv
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from PySide6.QtWidgets import (
    QDialog, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFrame,
    QGridLayout, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from picamera2 import Picamera2

from windows.defect_tuner_window import DefectTunerWindow
from models.align_image import align_with_template
from models.defect_detector import detect_defects
from config.utils import load_params
from config.config import INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT
from widgets.custom_widgets import (
    ButtonMain, ImageLabel, Switch,
    LabeledValue, TitleLabel, LabeledText,
    LabeledIndicator, Indicator, TitleLabelMain
)
from utils.gpio_rapsberry import RaspberryGPIO

import os, json, time

def _load_camera_params_from_json(path="config/camera_params.json"):
    """
    Lê os parâmetros guardados do template. Exemplo esperado:
    {
        "ExposureTime": 34900,
        "AnalogueGain": 2.1,
        "Brightness": 0.8,
        "Contrast": 2.3,
        "ColourGains": [2.3, 1.7]
    }
    """
    try:
        with open(path, "r") as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"⚠️ Não consegui ler {path}: {e}")
        return {}

def _build_controls_from_params(params: dict):
    """
    Converte o JSON em dict de controls para Picamera2.
    Desativa AE/AWB e aplica Exposure/Gain/ColourGains/Brightness/Contrast.
    """
    controls = {
        "AeEnable": False,
        "AwbEnable": False,
    }
    if "ExposureTime" in params:
        controls["ExposureTime"] = int(params["ExposureTime"])  # µs
        # Opcional: travar frame duration para estabilizar ainda mais
        # controls["FrameDurationLimits"] = (int(params["ExposureTime"]), int(params["ExposureTime"]))
    if "AnalogueGain" in params:
        controls["AnalogueGain"] = float(params["AnalogueGain"])
    if "ColourGains" in params and isinstance(params["ColourGains"], (list, tuple)) and len(params["ColourGains"]) == 2:
        rg, bg = params["ColourGains"]
        controls["ColourGains"] = (float(rg), float(bg))
    if "Brightness" in params:
        controls["Brightness"] = float(params["Brightness"])
    if "Contrast" in params:
        controls["Contrast"] = float(params["Contrast"])
    # Se tiveres "Saturation" e "Sharpness" no futuro, também dá:
    # if "Saturation" in params: controls["Saturation"] = float(params["Saturation"])
    # if "Sharpness"  in params: controls["Sharpness"]  = float(params["Sharpness"])
    return controls

class InspectionWindow(QDialog):
    def __init__(self, parent=None, picam2=None, template_path="", mask_path="", user_type="User", user=""):
        super().__init__(parent)
        self.setWindowTitle("Janela de Inspeção")
        self.showMaximized()
        # Dark theme background + default text color
        self.setStyleSheet("background-color: #121212; color: #f0f0f0;")

        self.picam2 = picam2
        self.template_path = template_path
        self.mask_path = mask_path
        self.user_type = user_type
        self.user = user

        self.defect_contours = []
        # Cumulative counters
        self.count_sheets = 0
        self.count_total_cans = 0
        self.count_good_cans = 0
        self.count_defect_cans = 0

        try:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(4)  # Pi 5 tem 4 cores
            print("Otpimization True")
        except Exception:
            pass

        # --- novos estados de visualização ---
        self.last_aligned = None     # última imagem alinhada analisada (color)
        self.last_vis_bw  = None     # última imagem B/W com círculos
        self.last_vis_color = None   # última imagem COLOR com círculos

        # Layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        main_layout2 = QHBoxLayout()
        main_layout2.setSpacing(16)
        main_layout.addLayout(main_layout2)

        # Bottom bar (fixed height 100)
        self.bottom_container = QWidget()
        self.bottom_container.setFixedHeight(100)
        self.bottom_container.setStyleSheet("background:#151515; border-top:1px solid #333333;")
        self.bottom_layout = QHBoxLayout(self.bottom_container)
        self.bottom_layout.setContentsMargins(12, 8, 12, 8)
        self.bottom_layout.setSpacing(16)
        # Optional separator line above bottom bar
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        # Add main content stretch then separator and bottom bar
        main_layout.addWidget(sep)
        main_layout.addWidget(self.bottom_container)
        # Ensure the main content area expands and bottom stays fixed
        main_layout.setStretch(0, 1)

        # GPIO indicators row (22, 23, 24, 25)
        self.gpio_pins = [22, 23, 24, 25]
        self.gpio = RaspberryGPIO(self.gpio_pins, mode='BCM', pull='UP')
        self.gpio_indicators = {}

        # Title on footer
        footer_title = QLabel("GPIO Inputs")
        footer_title.setStyleSheet("font-weight:600; color:#dddddd; font-size:14px;")
        self.bottom_layout.addWidget(footer_title)

        # Grid for indicators
        grid_wrap = QWidget()
        grid = QGridLayout(grid_wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(2)

        for idx, p in enumerate(self.gpio_pins):
            pin_widget = QWidget()
            pin_layout = QVBoxLayout(pin_widget)
            pin_layout.setContentsMargins(0, 0, 0, 0)
            pin_layout.setSpacing(4)
            lbl = QLabel(f"{p}")
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            lbl.setStyleSheet("color:#cccccc; font-size:13px;")
            ind = Indicator(state=False, diameter=18)
            ind_wrap = QWidget()
            ind_h = QHBoxLayout(ind_wrap)
            ind_h.setContentsMargins(0, 0, 0, 0)
            ind_h.addStretch(1)
            ind_h.addWidget(ind)
            ind_h.addStretch(1)
            pin_layout.addWidget(lbl)
            pin_layout.addWidget(ind_wrap)
            grid.addWidget(pin_widget, 0, idx)
            self.gpio_indicators[p] = ind

        self.bottom_layout.addWidget(grid_wrap)
        self.bottom_layout.addStretch(1)

        # Timer to poll GPIO states
        self.gpio_timer = QTimer(self)
        self.gpio_timer.setInterval(200)
        self.gpio_timer.timeout.connect(self._update_gpio_indicators)
        self.gpio_timer.start()

        # Painel esquerdo (card)
        card_style = "background:#1e1e1e; border:1px solid #333333; border-radius:8px;"
        left_card = QFrame()
        left_card.setFixedWidth(700)
        left_card.setStyleSheet(card_style)
        self.left_panel = QVBoxLayout(left_card)
        self.left_panel.setContentsMargins(12, 12, 12, 12)
        self.left_panel.setSpacing(10)
        main_layout2.addWidget(left_card)

        # Slim header
        self.titleLeftPanel = TitleLabelMain("Inspeção")
        self.left_panel.addWidget(self.titleLeftPanel)

        systemState = LabeledIndicator("Estado: ", True)
        self.left_panel.addWidget(systemState)

        systemInitTime = LabeledText("Início: ", "19/09/2025 - 10:30")
        self.left_panel.addWidget(systemInitTime)

        # Cumulative production counters
        self.systemTotalSheets = LabeledValue("Total Sheets: ", 0)
        self.left_panel.addWidget(self.systemTotalSheets)

        self.systemTotalCans = LabeledValue("Total Cans: ", 0)
        self.left_panel.addWidget(self.systemTotalCans)

        self.systemGoodCans = LabeledValue("Good Cans: ", 0)
        self.left_panel.addWidget(self.systemGoodCans)

        self.systemTotalDefects = LabeledValue("Defect Cans (Total): ", 0)
        self.left_panel.addWidget(self.systemTotalDefects)

        self.systemDefectPercent = LabeledValue("Defect %: ", "0.0%")
        self.left_panel.addWidget(self.systemDefectPercent)

        self.systemCansDefects = LabeledText("Latas c/ Defeito: ", "")
        self.left_panel.addWidget(self.systemCansDefects)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        self.left_panel.addWidget(line)

        layoutStartStopButtons = QHBoxLayout()
        layoutStartStopButtons.setSpacing(8)
        self.btn_start = ButtonMain("▶️ Start", font_size=16)
        layoutStartStopButtons.addWidget(self.btn_start)
        self.btn_stop = ButtonMain("⏹ Stop", font_size=16)
        layoutStartStopButtons.addWidget(self.btn_stop)
        self.left_panel.addLayout(layoutStartStopButtons)

        self.btn_defects = ButtonMain("🧪 Mostrar Defeitos", font_size=16)
        self.btn_defects.clicked.connect(self._show_defects)
        self.left_panel.addWidget(self.btn_defects)

        # Painel direito (imagem) em card
        right_card = QFrame()
        right_card.setStyleSheet(card_style)
        self.right_panel = QVBoxLayout(right_card)
        self.right_panel.setContentsMargins(12, 12, 12, 12)
        self.right_panel.setSpacing(10)
        main_layout2.addWidget(right_card)

        switches_layout = QHBoxLayout()
        switches_layout.setSpacing(12)
        self.toggle_template = Switch("🖼 Mostrar Template")
        self.toggle_template.stateChanged.connect(self._toggle_image)
        switches_layout.addWidget(self.toggle_template)

        self.toggle_bw = Switch("⬛ Preto e Branco")
        self.toggle_bw.setChecked(True)  # começa em B/W se quiseres
        self.toggle_bw.stateChanged.connect(self._toggle_bw)
        switches_layout.addWidget(self.toggle_bw)

        self.toggle_contours = Switch("🔍 Contornos dos Defeitos")
        self.toggle_contours.setChecked(True)
        self.toggle_contours.stateChanged.connect(self._toggle_defect_contours)
        switches_layout.addWidget(self.toggle_contours)

        self.btn_tuner_window = ButtonMain("🎛️ Tuner Window", font_size=16)
        self.btn_tuner_window.clicked.connect(self.open_tuner_window)
        switches_layout.addWidget(self.btn_tuner_window)

        self.right_panel.addLayout(switches_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        self.right_panel.addWidget(line)

        self.frame_img = ImageLabel()
        self.frame_img.setFixedSize(INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT)
        self.right_panel.addWidget(self.frame_img, alignment=Qt.AlignCenter)

        # Tooltip
        self.tooltip_img = QLabel(self)
        self.tooltip_img.setFixedSize(150, 150)
        self.tooltip_img.setVisible(False)

        # ----------------- Inicialização Picamera2 -----------------
        camera_params = _load_camera_params_from_json("config/camera_params.json")
        controls = _build_controls_from_params(camera_params)

        self.picam2 = picam2

        # passa os controls logo na configuração (garante que arranca no modo correto)
        config = self.picam2.create_still_configuration(
            main={"size": (4056, 3040), "format": "BGR888"},
            controls=controls
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.3)  # breve estabilização

        # reforça os controls após start (algumas versões aplicam melhor depois do start)
        try:
            self.picam2.set_controls(controls)
        except Exception as e:
            print("⚠️ set_controls após start falhou:", e)

        print("[INFO] Controles da câmara aplicados:", controls)

        # Carrega template e máscara
        self.template_full = cv2.imread(self.template_path)
        self.mask_full = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        self.aligned_full = self.template_full.copy()
        self.current_full = self.capture_picam_frame()

        # --- pré-computos do template (1x) ---
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

        tpl_gray = cv2.cvtColor(self.template_full, cv2.COLOR_BGR2GRAY)
        self.tpl_gray_eq = self.clahe.apply(cv2.GaussianBlur(tpl_gray, (3, 3), 0))
        self.tpl_lab = cv2.cvtColor(self.template_full, cv2.COLOR_BGR2LAB)

        # ROI seguro (afasta borda da máscara) + cache bbox
        self.safe_mask = cv2.erode(self.mask_full, np.ones((5,5), np.uint8), 1)
        nz = cv2.findNonZero(self.safe_mask)
        x0, y0, w0, h0 = cv2.boundingRect(nz) if nz is not None else (0, 0, self.mask_full.shape[1], self.mask_full.shape[0])
        self._mask_bbox = (x0, y0, w0, h0)

        # Carrega parâmetros
        self._load_params()

        # Carrega forma_base e instâncias
        self.instancias_poligonos = []
        try:
            with open("data/mask/forma_base.json", "r") as f:
                self.forma_base = json.load(f)
            with open("data/mask/instancias_poligonos.txt") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) != 2:
                        continue
                    idx_str, rest = parts
                    cx_str, cy_str, s_str = rest.split(",")
                    idx = int(idx_str)
                    cx, cy, s = float(cx_str), float(cy_str), float(s_str)
                    pontos = [(cx + x * s, cy + y * s) for x, y in self.forma_base]
                    poly = Polygon(pontos)
                    self.instancias_poligonos.append({
                        "numero_lata": idx,
                        "polygon": poly,
                        "center": (cx, cy),
                        "scale": s
                    })
            print(f"[INFO] Carregadas {len(self.instancias_poligonos)} instâncias de latas.")
        except Exception as e:
            print("❌ Erro ao carregar forma_base ou instâncias:", e)

        # Mostra template inicial
        self.show_image(self.template_full)

    # ----------------- Funções -----------------
    def capture_picam_frame(self):
        frame = self.picam2.capture_array("main")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.current_full = frame
        return frame

    def show_image(self, img_cv, draw_contours=None):
        if img_cv is None:
            return
        img_to_show = img_cv.copy()
        if draw_contours:
            cv2.drawContours(img_to_show, draw_contours, -1, (0, 0, 255), 2)
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, int(img_rgb.strides[0]), QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.frame_img.width(), self.frame_img.height(), Qt.KeepAspectRatio)
        self.frame_img.setPixmap(pixmap)
        self.frame_img.setAlignment(Qt.AlignCenter)

    def _toggle_image(self):
        self._refresh_view()

    def _to_gray_bgr(self, img):
        if img is None:
            return None
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def _refresh_view(self):
        """Atualiza a imagem mostrada conforme switches."""
        use_template = self.toggle_template.isChecked()
        show_contours = self.toggle_contours.isChecked()
        bw = self.toggle_bw.isChecked()

        if use_template:
            base = self.template_full
            img = self._to_gray_bgr(base) if bw else base
            self.show_image(img)
            return

        # caso: última análise
        if show_contours:
            if bw and self.last_vis_bw is not None:
                self.show_image(self.last_vis_bw)
            elif (not bw) and self.last_vis_color is not None:
                self.show_image(self.last_vis_color)
            else:
                # fallback se ainda não há análise com círculos
                base = self.last_aligned if self.last_aligned is not None else self.current_full
                img = self._to_gray_bgr(base) if bw else base
                self.show_image(img)
        else:
            base = self.last_aligned if self.last_aligned is not None else self.current_full
            img = self._to_gray_bgr(base) if bw else base
            self.show_image(img)

    def _toggle_image(self):
        self._refresh_view()

    def _toggle_defect_contours(self):
        self._refresh_view()

    def _toggle_bw(self):
        self._refresh_view()


    def _toggle_defect_contours(self):
        if self.toggle_contours.isChecked() and self.defect_contours:
            self.show_image(self.aligned_full, draw_contours=self.defect_contours)
        else:
            self.show_image(self.aligned_full)

    def _load_params(self):
        params = load_params("config/inspection_params.json") or {}

        # ---- existentes ----
        self.dark_threshold           = int(params.get("dark_threshold", 30))
        self.bright_threshold         = int(params.get("bright_threshold", 30))
        self.dark_morph_kernel_size   = int(params.get("dark_morph_kernel_size", 3))
        self.dark_morph_iterations    = int(params.get("dark_morph_iterations", 1))
        self.bright_morph_kernel_size = int(params.get("bright_morph_kernel_size", 3))
        self.bright_morph_iterations  = int(params.get("bright_morph_iterations", 1))
        # aceita detect_area (antigo) OU min_defect_area (novo)
        self.min_defect_area          = int(params.get("detect_area", params.get("min_defect_area", 1)))
        self.dark_gradient_threshold  = int(params.get("dark_gradient_threshold", 10))
        self.blue_threshold           = int(params.get("blue_threshold", 25))
        self.red_threshold            = int(params.get("red_threshold", 25))

        # ---- NOVOS: MS-SSIM ----
        self.use_ms_ssim              = bool(int(params.get("use_ms_ssim", 1)))
        self.msssim_percentile        = float(params.get("msssim_percentile", 99.5))
        self.msssim_weight            = float(params.get("msssim_weight", 0.5))

        # kernels por escala (ímpares)
        k1 = int(params.get("msssim_kernel_size_s1", 7))
        k2 = int(params.get("msssim_kernel_size_s2", 5))
        k3 = int(params.get("msssim_kernel_size_s3", 3))
        if k1 < 1: k1 = 1
        if k2 < 1: k2 = 1
        if k3 < 1: k3 = 1
        if k1 % 2 == 0: k1 += 1
        if k2 % 2 == 0: k2 += 1
        if k3 % 2 == 0: k3 += 1
        self.msssim_kernel_sizes = (k1, k2, k3)

        # sigmas por escala
        s1 = float(params.get("msssim_sigma_s1", 1.5))
        s2 = float(params.get("msssim_sigma_s2", 1.0))
        s3 = float(params.get("msssim_sigma_s3", 0.8))
        self.msssim_sigmas = (s1, s2, s3)

        # morfologia do mapa MS-SSIM
        self.msssim_morph_kernel_size = int(params.get("msssim_morph_kernel_size", 3))
        if self.msssim_morph_kernel_size < 1:
            self.msssim_morph_kernel_size = 1
        if self.msssim_morph_kernel_size % 2 == 0:
            self.msssim_morph_kernel_size += 1

        self.msssim_morph_iterations  = int(params.get("msssim_morph_iterations", 1))
        if self.msssim_morph_iterations < 0:
            self.msssim_morph_iterations = 0

        # ---- NOVOS: Mapas morfológicos L, Δa/Δb e Fusão ----
        self.use_morph_maps      = bool(int(params.get("use_morph_maps", 1)))
        self.th_top_percentile   = float(params.get("th_top_percentile", 99.5))
        self.th_black_percentile = float(params.get("th_black_percentile", 99.5))
        # clamps percentis
        self.th_top_percentile   = max(0.0, min(100.0, self.th_top_percentile))
        self.th_black_percentile = max(0.0, min(100.0, self.th_black_percentile))

        self.se_top   = int(params.get("se_top", 9))
        self.se_black = int(params.get("se_black", 9))
        if self.se_top < 1: self.se_top = 1
        if self.se_black < 1: self.se_black = 1
        if self.se_top % 2 == 0: self.se_top += 1
        if self.se_black % 2 == 0: self.se_black += 1

        self.use_color_delta  = bool(int(params.get("use_color_delta", 1)))
        self.color_metric     = str(params.get("color_metric", "maxab"))
        self.color_percentile = float(params.get("color_percentile", 99.0))
        self.color_percentile = max(0.0, min(100.0, self.color_percentile))

        self.fusion_mode = str(params.get("fusion_mode", "or"))
        self.w_struct    = float(params.get("w_struct", 0.50))
        self.w_top       = float(params.get("w_top", 0.25))
        self.w_black     = float(params.get("w_black", 0.15))
        self.w_color     = float(params.get("w_color", 0.10))
        self.fused_percentile = float(params.get("fused_percentile", 99.5))
        # clamps
        for attr in ("w_struct","w_top","w_black","w_color"):
            v = getattr(self, attr)
            setattr(self, attr, max(0.0, min(1.0, float(v))))
        self.fused_percentile = max(0.0, min(100.0, self.fused_percentile))

        # ---- clamps básicos úteis ----
        self.dark_threshold   = max(0, min(255, self.dark_threshold))
        self.bright_threshold = max(0, min(255, self.bright_threshold))
        self.blue_threshold   = max(0, min(255, self.blue_threshold))
        self.red_threshold    = max(0, min(255, self.red_threshold))

        # kernels ímpares nas morfologias principais
        if self.dark_morph_kernel_size < 1: self.dark_morph_kernel_size = 1
        if self.dark_morph_kernel_size % 2 == 0: self.dark_morph_kernel_size += 1
        if self.bright_morph_kernel_size < 1: self.bright_morph_kernel_size = 1
        if self.bright_morph_kernel_size % 2 == 0: self.bright_morph_kernel_size += 1

        self.dark_morph_iterations   = max(0, self.dark_morph_iterations)
        self.bright_morph_iterations = max(0, self.bright_morph_iterations)

        self.min_defect_area = max(1, self.min_defect_area)


    
    @staticmethod
    def _normalize_lab_to_template(tpl_bgr, img_bgr, mask):
        
        eps = 1e-6
        tpl_lab = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        img_lab = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2LAB).astype(np.float32)
        m = (mask > 0)
        norm = img_lab.copy()
        for c in range(3):
            mu_t, sd_t = float(np.mean(tpl_lab[:,:,c][m])), float(np.std(tpl_lab[:,:,c][m]) + eps)
            mu_i, sd_i = float(np.mean(img_lab[:,:,c][m])), float(np.std(img_lab[:,:,c][m]) + eps)
            gain = sd_t / sd_i
            bias = mu_t - gain * mu_i
            norm[:,:,c] = gain * img_lab[:,:,c] + bias
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_LAB2BGR)

    def _show_defects(self):
        total_start = time.perf_counter()

        try:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(4)
        except Exception:
            pass

        # 1) Captura (stream "main") e trava AE/AWB
        frame = self.picam2.capture_array("main")
        self.current_full = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.picam2.set_controls({"AeEnable": False, "AwbEnable": False})

        # 2) Alinhamento (current -> template). Mantém também a H.
        H = None
        try:
            self.aligned_full, H = align_with_template(self.current_full, self.template_full)
            if self.aligned_full is None or H is None:
                raise ValueError("Falhou alinhamento")
        except Exception as e:
            print("⚠️ Erro no alinhamento, usando imagem original:", e)
            self.aligned_full = self.current_full.copy()
            # Homografia identidade como fallback (desenha sem reprojetar)
            H = np.eye(3, dtype=np.float32)

        # Inversa: template -> current (para reprojetar desenho)
        try:
            H_inv = np.linalg.inv(H)
        except Exception:
            H_inv = np.eye(3, dtype=np.float32)

        # 3) ROI da máscara (cálculo/uso)
        if not hasattr(self, "_mask_bbox"):
            nz = cv2.findNonZero(self.mask_full)
            if nz is None:
                print("❌ Máscara vazia.")
                return
            x0, y0, w0, h0 = cv2.boundingRect(nz)
            self._mask_bbox = (x0, y0, w0, h0)
        else:
            x0, y0, w0, h0 = self._mask_bbox

        # 4) Recortes ROI no espaço do TEMPLATE (porque aligned_full está nesse espaço)
        tpl_roi  = self.template_full[y0:y0+h0, x0:x0+w0]
        cur_roi  = self.aligned_full[y0:y0+h0, x0:x0+w0]
        mask_roi = self.safe_mask[y0:y0+h0, x0:x0+w0]

        # Normalização fotométrica condicional (como no Tuner)
        tpl_masked_roi = cv2.bitwise_and(tpl_roi, tpl_roi, mask=mask_roi)
        cur_masked_roi = cv2.bitwise_and(cur_roi, cur_roi, mask=mask_roi)
        tpl_mean = cv2.mean(cv2.cvtColor(tpl_masked_roi, cv2.COLOR_BGR2GRAY), mask=mask_roi)[0]
        ali_mean = cv2.mean(cv2.cvtColor(cur_masked_roi, cv2.COLOR_BGR2GRAY), mask=mask_roi)[0]
        if abs(tpl_mean - ali_mean) > 1.0:
            cur_masked_roi = self._normalize_lab_to_template(tpl_masked_roi, cur_masked_roi, mask_roi)

        # 5) Deteção de defeitos (em coords do TEMPLATE/ROI)
        t_det = time.perf_counter()
        result = detect_defects(
            tpl_masked_roi, cur_masked_roi, mask_roi,
            self.dark_threshold, self.bright_threshold,
            self.dark_morph_kernel_size,  self.dark_morph_iterations,
            self.bright_morph_kernel_size, self.bright_morph_iterations,
            self.min_defect_area, self.dark_gradient_threshold,
            self.blue_threshold, self.red_threshold,
            # ---- MS-SSIM ----
            use_ms_ssim=self.use_ms_ssim,
            msssim_percentile=self.msssim_percentile,
            msssim_weight=self.msssim_weight,
            msssim_kernel_sizes=self.msssim_kernel_sizes,
            msssim_sigmas=self.msssim_sigmas,
            msssim_morph_kernel_size=self.msssim_morph_kernel_size,
            msssim_morph_iterations=self.msssim_morph_iterations,
            # ---- Mapas adicionais + Fusão ----
            use_morph_maps=self.use_morph_maps,
            th_top_percentile=self.th_top_percentile,
            th_black_percentile=self.th_black_percentile,
            se_top=self.se_top,
            se_black=self.se_black,
            use_color_delta=self.use_color_delta,
            color_metric=self.color_metric,
            color_percentile=self.color_percentile,
            fusion_mode=self.fusion_mode,
            w_struct=self.w_struct,
            w_top=self.w_top,
            w_black=self.w_black,
            w_color=self.w_color,
            fused_percentile=self.fused_percentile,
        )
        if len(result) == 7:
            final_mask, contours_roi, darker_mask_roi, brighter_mask_roi, blue_mask_roi, red_mask_roi, _ = result
        else:
            final_mask, contours_roi, darker_mask_roi, brighter_mask_roi, blue_mask_roi, red_mask_roi = result

        # 6) Preparar visualizações sobre a IMAGEM ORIGINAL (sem warp)
        gray_full = cv2.cvtColor(self.current_full, cv2.COLOR_BGR2GRAY)
        vis_bw    = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        vis_color = self.current_full.copy()

        # helpers para reprojetar um ponto e um raio
        def _warp_point_T2C(Hinv, x_t, y_t):
            v = np.array([x_t, y_t, 1.0], dtype=np.float32)
            w = Hinv @ v
            w /= (w[2] + 1e-12)
            return float(w[0]), float(w[1])

        def _warp_radius_T2C(Hinv, cx_t, cy_t, r_t):
            # aproxima o raio transformando um ponto deslocado no template
            x2_t, y2_t = cx_t + r_t, cy_t
            cx_c, cy_c = _warp_point_T2C(Hinv, cx_t, cy_t)
            x2_c, y2_c = _warp_point_T2C(Hinv, x2_t, y2_t)
            return max(1.0, ((x2_c - cx_c)**2 + (y2_c - cy_c)**2) ** 0.5)

        mask_types = {
            "dark":   darker_mask_roi,
            "bright": brighter_mask_roi,
            "blue":   blue_mask_roi,
            "red":    red_mask_roi
        }
        color_map = {
            "dark":   (0, 255, 0),
            "bright": (255, 255, 0),
            "blue":   (0, 0, 255),
            "red":    (255, 0, 255)
        }

        defect_data = []
        self.defect_contours = []

        for cnt_roi in contours_roi:
            xr, yr, wr, hr = cv2.boundingRect(cnt_roi)
            # tipo dominante no ROI (template-space)
            label = max(
                mask_types,
                key=lambda k: cv2.countNonZero(mask_types[k][yr:yr+hr, xr:xr+wr])
            )
            color = color_map[label]

            # círculo mínimo no ROI (template-space)
            (cxr_f, cyr_f), rr_f = cv2.minEnclosingCircle(cnt_roi)
            cx_t, cy_t, r_t = float(cxr_f + x0), float(cyr_f + y0), float(max(rr_f, 8))

            # reprojetar centro/raio para CURRENT (sem warp)
            cx_c, cy_c = _warp_point_T2C(H_inv, cx_t, cy_t)
            r_c = max(24.0, _warp_radius_T2C(H_inv, cx_t, cy_t, r_t) + 6.0)

            # reprojetar contorno completo para guardar/mostrar (opcional)
            cnt_full_t = cnt_roi.copy()
            cnt_full_t[:, 0, 0] += x0
            cnt_full_t[:, 0, 1] += y0
            # aplica H_inv
            pts = cnt_full_t.reshape(-1, 2).astype(np.float32)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
            pts_c = (pts_h @ H_inv.T)
            pts_c[:, 0] /= (pts_c[:, 2] + 1e-12)
            pts_c[:, 1] /= (pts_c[:, 2] + 1e-12)
            cnt_full_c = pts_c[:, :2].reshape(-1, 1, 2).astype(np.int32)
            self.defect_contours.append(cnt_full_c)

            # desenhar nos visuais (CURRENT)
            cxi, cyi, ri = int(round(cx_c)), int(round(cy_c)), int(round(r_c))
            cv2.circle(vis_bw,    (cxi, cyi), ri, color, 3, lineType=cv2.LINE_AA)
            cv2.circle(vis_bw,    (cxi, cyi), 2,  color, -1, lineType=cv2.LINE_AA)
            cv2.circle(vis_color, (cxi, cyi), ri, color, 3, lineType=cv2.LINE_AA)
            cv2.circle(vis_color, (cxi, cyi), 2,  color, -1, lineType=cv2.LINE_AA)

            # nº da lata (encontra por polígono em coords CURRENT)
            lata_id = None
            pt = Point(cxi, cyi)
            for pol in self.instancias_poligonos:
                # Polígonos estão em coords do TEMPLATE? Se sim, reprojeta vértices 1x ao arranque.
                # Supondo que já tens os polígonos em CURRENT, senão comentar…
                if pol["polygon"].contains(pt) or pol["polygon"].distance(pt) <= 2.0:
                    lata_id = pol["numero_lata"]
                    break

            if lata_id is None and self.instancias_poligonos:
                nearest = min(self.instancias_poligonos,
                            key=lambda p: (p["center"][0] - cxi)**2 + (p["center"][1] - cyi)**2)
                lata_id = nearest["numero_lata"]

            if lata_id is not None:
                cv2.putText(vis_bw,    f"#{lata_id}", (max(cxi - ri, 0), max(cyi - ri - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(vis_color, f"#{lata_id}", (max(cxi - ri, 0), max(cyi - ri - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            defect_data.append({
                "lata": lata_id,
                "tipo": label,
                "area": round(cv2.contourArea(cnt_full_c), 2),
                "bbox": (int(xr + x0), int(yr + y0), int(wr), int(hr)),  # bbox ainda em template-space, se precisares reprojeta os 4 cantos
                "cx": int(cxi), "cy": int(cyi), "r": int(ri)
            })

        # 7) Atualiza contadores e UI (igual ao teu)
        can_ids = {d["lata"] for d in defect_data if d.get("lata") is not None}
        cans_with_defects = len(can_ids)
        per_sheet_total = len(self.instancias_poligonos) if hasattr(self, 'instancias_poligonos') else 0
        per_sheet_good = max(0, per_sheet_total - cans_with_defects)
        self.count_sheets += 1
        self.count_total_cans += per_sheet_total
        self.count_good_cans += per_sheet_good
        self.count_defect_cans += cans_with_defects
        defect_pct = (self.count_defect_cans / self.count_total_cans * 100.0) if self.count_total_cans > 0 else 0.0

        self.systemTotalSheets.set_value(self.count_sheets)
        self.systemTotalCans.set_value(self.count_total_cans)
        self.systemGoodCans.set_value(self.count_good_cans)
        self.systemTotalDefects.set_value(self.count_defect_cans)
        self.systemDefectPercent.set_value(f"{defect_pct:.1f}%")
        ids_text = ", ".join(str(i) for i in sorted(can_ids)) if can_ids else "—"
        self.systemCansDefects.update_value(ids_text)

        # 8) Mostra sobre o frame ORIGINAL (sem funil)
        self.show_image(vis_bw)
        print(f"[Tempo Total] _show_defects: {time.perf_counter() - total_start:.4f} s")

        self.last_aligned   = self.aligned_full.copy()   # ainda guardo, útil p/ debug
        self.last_vis_bw    = vis_bw
        self.last_vis_color = vis_color
        self._refresh_view()

    def open_tuner_window(self):
        # 1) capturar frame atual
        frame = self.picam2.capture_array("main")
        cur = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2) alinhar ao template em espaço do template
        aligned, _ = align_with_template(cur, self.template_full)

        # 3) aplicar máscara do template (em coords do template!)
        tpl_m = cv2.bitwise_and(self.template_full, self.template_full, mask=self.mask_full)
        ali_m = cv2.bitwise_and(aligned,        aligned,        mask=self.mask_full)

        # 4) abrir o Tuner com imagens realmente diferentes
        tuner = DefectTunerWindow(self, tpl_m, ali_m, self.mask_full, self.user_type, self.user)
        tuner.exec()


    def closeEvent(self, event):
        try:
            if hasattr(self, "gpio_timer"):
                self.gpio_timer.stop()
        except Exception:
            pass

        try:
            if hasattr(self, "gpio") and self.gpio is not None:
                self.gpio.cleanup()
        except Exception:
            pass

        self.picam2.stop()
        event.accept()

    def _update_gpio_indicators(self):
        # Poll GPIO states via utils wrapper
        try:
            if not hasattr(self, "gpio") or self.gpio is None or not self.gpio.available:
                return
            states = self.gpio.read_states()
            for p, ind in self.gpio_indicators.items():
                ind.set_state(bool(states.get(p, False)))
        except Exception:
            pass
