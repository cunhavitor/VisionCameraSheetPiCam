import os
import csv
import json
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QGroupBox,
    QFormLayout, QFrame, QTabWidget
)

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QKeySequence, QShortcut
import cv2
import numpy as np
from widgets.custom_widgets import ImageLabel
from models.defect_detector import detect_defects

INSPECTION_PREVIEW_WIDTH = 800
INSPECTION_PREVIEW_HEIGHT = 600


class DefectTunerWindow(QDialog):
    def __init__(self, parent, tpl_img, aligned_img, mask, user_type="User", user_name=""):
        super().__init__(parent)

        self.user_type = user_type
        self.user_name = user_name

        self.tpl = tpl_img
        self.aligned = aligned_img
        self.mask = mask

        print("DEBUG DefectTunerWindow tpl:", type(self.tpl), getattr(self.tpl, "shape", None))
        print("DEBUG DefectTunerWindow aligned:", type(self.aligned), getattr(self.aligned, "shape", None))
        print("DEBUG DefectTunerWindow mask:", type(self.mask), getattr(self.mask, "shape", None))

        self.setWindowTitle("Ajuste de Parâmetros de Defeitos")
        self.showMaximized()
        # Dark theme to match Inspection window
        self.setStyleSheet("background-color: #121212; color: #f0f0f0;")

        # Valores padrão
        self.params = {
        "dark_threshold": 30,
        "bright_threshold": 30,
        "blue_threshold": 25,
        "red_threshold": 25,
        "dark_morph_kernel_size": 3,
        "dark_morph_iterations": 1,
        "bright_morph_kernel_size": 3,
        "bright_morph_iterations": 1,
        "dark_gradient_threshold": 10,
        "min_defect_area": 1,

        # ⬇️ NOVOS: MS-SSIM
        "use_ms_ssim": 1,                 # bool como int (0/1) para simplicidade
        "msssim_percentile": 99.5,        # float
        "msssim_weight": 0.5,             # float
        "msssim_kernel_size_s1": 7,       # int (ímpar)
        "msssim_kernel_size_s2": 5,       # int (ímpar)
        "msssim_kernel_size_s3": 3,       # int (ímpar)
        "msssim_sigma_s1": 1.5,           # float
        "msssim_sigma_s2": 1.0,           # float
        "msssim_sigma_s3": 0.8,           # float
        "msssim_morph_kernel_size": 3,    # int
        "msssim_morph_iterations": 1,     # int

        # Mapas adicionais e fusão
        "use_morph_maps": 1,
        "th_top_percentile": 99.5,
        "th_black_percentile": 99.5,
        "se_top": 9,
        "se_black": 9,

        "use_color_delta": 1,
        "color_metric": "maxab",
        "color_percentile": 99.0,

        "fusion_mode": "or",
        "w_struct": 0.50,
        "w_top": 0.25,
        "w_black": 0.15,
        "w_color": 0.10,
        "fused_percentile": 99.5,
        # Final map behavior
        "final_mode": "extended",  # "classic" or "extended"
        "final_include_gradient": 1,
        # Visual
        "use_heatmap_bg": 1,
        # Overexposure ignore (glare/white saturation)
        "ignore_overexposed": 0
        }

        # ⬇️ carregar valores guardados (antes de criares os spinboxes)
        self._load_saved_params("config/inspection_params.json")

        self._update_scheduled = False
        self.last_preview = None
        # Tooltips (Português) para todos os parâmetros
        self._tooltips = {
            "dark_threshold": "Limiar para regiões mais escuras que o template (0–255). Valores mais altos tornam a deteção mais restrita.",
            "bright_threshold": "Limiar para regiões mais claras (0–255).",
            "blue_threshold": "Limiar de diferença no canal azul (0–255). Útil para variações cromáticas.",
            "red_threshold": "Limiar de diferença no canal vermelho (0–255).",
            "dark_gradient_threshold": "Intensidade mínima do gradiente após equalização (0–255). Ajuda a destacar contornos escuros.",
            "min_defect_area": "Área mínima (px) para considerar uma região como defeito.",
            "dark_morph_kernel_size": "Tamanho do kernel morfológico para mapa escuro (ímpar).",
            "dark_morph_iterations": "Número de iterações morfológicas no mapa escuro.",
            "bright_morph_kernel_size": "Tamanho do kernel morfológico para mapa claro (ímpar).",
            "bright_morph_iterations": "Número de iterações morfológicas no mapa claro.",
            "use_ms_ssim": "Ativa a métrica estrutural multi‑escala (MS‑SSIM).",
            "msssim_percentile": "Percentil do mapa MS‑SSIM usado como limiar (90–99.9).",
            "msssim_weight": "Peso do MS‑SSIM na fusão final (0–1).",
            "msssim_kernel_size_s1": "Janela (ímpar) da escala 1.",
            "msssim_kernel_size_s2": "Janela (ímpar) da escala 2.",
            "msssim_kernel_size_s3": "Janela (ímpar) da escala 3.",
            "msssim_sigma_s1": "Sigma do desfoque na escala 1.",
            "msssim_sigma_s2": "Sigma do desfoque na escala 2.",
            "msssim_sigma_s3": "Sigma do desfoque na escala 3.",
            "msssim_morph_kernel_size": "Kernel morfológico aplicado ao mapa MS‑SSIM (ímpar).",
            "msssim_morph_iterations": "Iterações morfológicas no mapa MS‑SSIM.",
            "use_morph_maps": "Ativa mapas Top‑hat/Black‑hat em L (luminosidade).",
            "th_top_percentile": "Percentil do limiar para Top‑hat (claros).",
            "th_black_percentile": "Percentil do limiar para Black‑hat (escuros).",
            "se_top": "Elemento estruturante do Top‑hat (ímpar).",
            "se_black": "Elemento estruturante do Black‑hat (ímpar).",
            "use_color_delta": "Ativa deteção por variação de cor Δa/Δb (LAB).",
            "color_metric": "Métrica de cor: maxab ou l2ab.",
            "color_percentile": "Percentil para limiar do mapa de cor.",
            "fusion_mode": "Combinação: or (qualquer) ou weighted (ponderada).",
            "w_struct": "Peso do mapa estrutural (MS‑SSIM).",
            "w_top": "Peso do mapa Top‑hat.",
            "w_black": "Peso do mapa Black‑hat.",
            "w_color": "Peso do mapa de cor.",
            "fused_percentile": "Percentil do limiar no mapa fundido (0–100).",
            "final_mode": "Como calcular o mapa Final: clássico (união dos 4 mapas) ou estendido (com fusão).",
            "final_include_gradient": "Se ligado, inclui o mapa de Gradiente na máscara Final (modo clássico).",
            "use_heatmap_bg": "Usa o mapa de calor do diff escuro (CLAHE) como fundo.",
            "ignore_overexposed": "Ignora zonas muito claras/brancas (reflexos/saturação) ao contabilizar defeitos."
        }

        # --- Layout principal: vertical (topo com 3 colunas + barra inferior) ---
        self.setLayout(QVBoxLayout())
        main_widget = QWidget(self)
        self.layout().addWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)     # coluna principal (vertical)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Linha de topo (3 colunas)
        top_row = QHBoxLayout()
        main_layout.addLayout(top_row, 1)

        # Colunas de controlos (em card com scroll)
        self.control_layout = QVBoxLayout()
        self.control_layout2 = QVBoxLayout()
        left_card = QFrame()
        left_card.setStyleSheet(
            "background:#1e1e1e; border:1px solid #333333; border-radius:8px;"
            " QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QLabel { font-size: 14px; }"
        )
        left_wrap = QVBoxLayout(left_card)
        left_wrap.setContentsMargins(12, 12, 12, 12)
        left_wrap.setSpacing(8)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_container = QWidget()
        controls_scroll.setWidget(controls_container)
        controls_v = QVBoxLayout(controls_container)
        controls_v.setSpacing(8)

        # Tabs container
        tabs = QTabWidget()
        controls_v.addWidget(tabs)

        # Helpers for sections with compact forms
        self.spin_boxes = {}
        def add_section(title, parent_v):
            box = QGroupBox(title)
            box.setStyleSheet("QGroupBox{font-weight:600; border:1px solid #333; border-radius:6px; margin-top:8px;} QGroupBox:title{left:8px; padding:0 4px;}")
            form = QFormLayout()
            form.setContentsMargins(8, 8, 8, 8)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(6)
            box.setLayout(form)
            parent_v.addWidget(box)
            return form

        def add_spin(label, key, a, b, form):
            w = QSpinBox(); w.setRange(a, b); w.setValue(int(self.params[key]))
            w.setToolTip(self._tooltips.get(key, ""))
            w.valueChanged.connect(lambda v, k=key: self._on_slider_change(k, v))
            self.spin_boxes[key] = w
            form.addRow(label, w)

        def add_dspin(label, key, a, b, form, step=0.1, dec=2):
            w = QDoubleSpinBox(); w.setDecimals(dec); w.setSingleStep(step); w.setRange(a, b); w.setValue(float(self.params[key]))
            w.setToolTip(self._tooltips.get(key, ""))
            w.valueChanged.connect(lambda v, k=key: self._on_slider_change_float(k, v))
            self.spin_boxes[key] = w
            form.addRow(label, w)

        def add_check(label, key, form):
            w = QCheckBox(label); w.setChecked(bool(int(self.params[key])))
            w.setToolTip(self._tooltips.get(key, ""))
            w.stateChanged.connect(lambda s, k=key: self._on_checkbox_change(k, s))
            self.spin_boxes[key] = w
            form.addRow("", w)

        # ----- Basics Tab -----
        basics_page = QWidget(); basics_v = QVBoxLayout(basics_page); basics_v.setSpacing(8)
        basics_header = QLabel("🧩 Basics"); basics_header.setStyleSheet("font-weight:600; font-size:16px;")
        basics_v.addWidget(basics_header)
        basics_desc = QLabel("Ajustes básicos de thresholds e parâmetros gerais que controlam a sensibilidade e o tamanho mínimo dos defeitos. Use para calibrar o nível de detecção inicial.")
        basics_desc.setWordWrap(True)
        basics_desc.setStyleSheet("color:#aaaaaa; font-size:12px;")
        basics_v.addWidget(basics_desc)
        sep_b = QFrame(); sep_b.setFrameShape(QFrame.HLine); sep_b.setStyleSheet("color:#333333;"); basics_v.addWidget(sep_b)
        f_basic = add_section("Thresholds / Basics", basics_v)
        add_spin("Threshold Escuro", "dark_threshold", 0, 255, f_basic)
        add_spin("Threshold Amarelo", "bright_threshold", 0, 255, f_basic)
        add_spin("Threshold Azul", "blue_threshold", 0, 255, f_basic)
        add_spin("Threshold Vermelho", "red_threshold", 0, 255, f_basic)
        add_spin("Gradiente Escuro (CLAHE)", "dark_gradient_threshold", 0, 255, f_basic)
        add_spin("Área mínima do defeito", "min_defect_area", 1, 100000, f_basic)
        add_check("Ignorar zonas superexpostas", "ignore_overexposed", f_basic)
        basics_idx = tabs.addTab(basics_page, "🧩 Basics")
        tabs.setTabToolTip(basics_idx, "Parâmetros gerais para calibrar sensibilidade e tamanho mínimo de defeitos.")
        self._basics_v = basics_v

        # ----- Morphology Tab -----
        morph_page = QWidget(); morph_v = QVBoxLayout(morph_page); morph_v.setSpacing(8)
        morph_header = QLabel("📐 Morphology"); morph_header.setStyleSheet("font-weight:600; font-size:16px;")
        morph_v.addWidget(morph_header)
        morph_desc = QLabel("Parâmetros morfológicos aplicados aos mapas de intensidade para reduzir ruído e consolidar regiões de defeito (dilata/erode zonas detectadas).")
        morph_desc.setWordWrap(True)
        morph_desc.setStyleSheet("color:#aaaaaa; font-size:12px;")
        morph_v.addWidget(morph_desc)
        sep_m = QFrame(); sep_m.setFrameShape(QFrame.HLine); sep_m.setStyleSheet("color:#333333;"); morph_v.addWidget(sep_m)
        f_morph = add_section("Morfologia (Intensidade)", morph_v)
        add_spin("Kernel morf. Escuro (ímpar)", "dark_morph_kernel_size", 1, 31, f_morph)
        add_spin("Iterações morf. Escuro", "dark_morph_iterations", 0, 10, f_morph)
        add_spin("Kernel morf. Amarelo (ímpar)", "bright_morph_kernel_size", 1, 31, f_morph)
        add_spin("Iterações morf. Amarelo", "bright_morph_iterations", 0, 10, f_morph)
        morph_idx = tabs.addTab(morph_page, "📐 Morphology")
        tabs.setTabToolTip(morph_idx, "Operações morfológicas para reduzir ruído e consolidar regiões detectadas.")

        # ----- MS-SSIM Tab -----
        msssim_page = QWidget(); msssim_v = QVBoxLayout(msssim_page); msssim_v.setSpacing(8)
        msssim_header = QLabel("🧪 MS-SSIM"); msssim_header.setStyleSheet("font-weight:600; font-size:16px;")
        msssim_v.addWidget(msssim_header)
        msssim_desc = QLabel("Métrica estrutural multi‑escala que realça diferenças de textura/estrutura entre o template e a imagem. Útil para detectar defeitos subtis de forma.")
        msssim_desc.setWordWrap(True)
        msssim_desc.setStyleSheet("color:#aaaaaa; font-size:12px;")
        msssim_v.addWidget(msssim_desc)
        sep_s = QFrame(); sep_s.setFrameShape(QFrame.HLine); sep_s.setStyleSheet("color:#333333;"); msssim_v.addWidget(sep_s)
        f_msssim = add_section("MS-SSIM", msssim_v)
        add_check("Ativar MS-SSIM", "use_ms_ssim", f_msssim)
        add_dspin("Percentil MS-SSIM", "msssim_percentile", 90.0, 99.9, f_msssim, step=0.1, dec=1)
        add_dspin("Peso MS-SSIM", "msssim_weight", 0.0, 1.0, f_msssim, step=0.05, dec=2)
        add_spin("Janela Escala 1", "msssim_kernel_size_s1", 3, 11, f_msssim)
        add_spin("Janela Escala 2", "msssim_kernel_size_s2", 3, 11, f_msssim)
        add_spin("Janela Escala 3", "msssim_kernel_size_s3", 3, 11, f_msssim)
        add_dspin("Sigma Escala 1", "msssim_sigma_s1", 0.5, 2.5, f_msssim, step=0.1, dec=2)
        add_dspin("Sigma Escala 2", "msssim_sigma_s2", 0.5, 2.5, f_msssim, step=0.1, dec=2)
        add_dspin("Sigma Escala 3", "msssim_sigma_s3", 0.5, 2.5, f_msssim, step=0.1, dec=2)
        add_spin("Kernel morf. MS-SSIM", "msssim_morph_kernel_size", 1, 7, f_msssim)
        add_spin("Iterações morf. MS-SSIM", "msssim_morph_iterations", 0, 3, f_msssim)
        msssim_idx = tabs.addTab(msssim_page, "🧪 MS-SSIM")
        tabs.setTabToolTip(msssim_idx, "Métrica estrutural multi‑escala para diferenças de textura/estrutura.")

        # ----- Color Tab -----
        color_page = QWidget(); color_v = QVBoxLayout(color_page); color_v.setSpacing(8)
        color_header = QLabel("🎨 Color"); color_header.setStyleSheet("font-weight:600; font-size:16px;")
        color_v.addWidget(color_header)
        color_desc = QLabel("Realça variações de cor (Δa/Δb no espaço LAB) entre template e imagem — útil para manchas, sujidade ou desvios cromáticos.")
        color_desc.setWordWrap(True)
        color_desc.setStyleSheet("color:#aaaaaa; font-size:12px;")
        color_v.addWidget(color_desc)
        sep_c = QFrame(); sep_c.setFrameShape(QFrame.HLine); sep_c.setStyleSheet("color:#333333;"); color_v.addWidget(sep_c)
        f_color = add_section("Cor Δa/Δb", color_v)
        add_check("Ativar Δa/Δb", "use_color_delta", f_color)
        lbl_metric = QLabel("Métrica de cor"); f_color.addRow(lbl_metric, QWidget())
        self.color_metric_cb = QComboBox(); self.color_metric_cb.addItems(["maxab", "l2ab"])
        self.color_metric_cb.setCurrentText(str(self.params["color_metric"]))
        self.color_metric_cb.currentTextChanged.connect(lambda val: self._on_text_param_change("color_metric", val))
        self.color_metric_cb.setToolTip(self._tooltips.get("color_metric", ""))
        f_color.addRow("", self.color_metric_cb)
        add_dspin("Percentil cor", "color_percentile", 90.0, 99.9, f_color, step=0.1, dec=1)
        color_idx = tabs.addTab(color_page, "🎨 Color")
        tabs.setTabToolTip(color_idx, "Variações de cor LAB (Δa/Δb): manchas e desvios cromáticos.")

        # ----- Fusion Tab -----
        fusion_page = QWidget(); fusion_v = QVBoxLayout(fusion_page); fusion_v.setSpacing(8)
        fusion_header = QLabel("🔗 Fusion"); fusion_header.setStyleSheet("font-weight:600; font-size:16px;")
        fusion_v.addWidget(fusion_header)
        fusion_desc = QLabel("Combina as diferentes pistas (estrutura, morfologia e cor) num mapa final. Ajuste pesos e percentis para equilibrar precisão e recall.")
        fusion_desc.setWordWrap(True)
        fusion_desc.setStyleSheet("color:#aaaaaa; font-size:12px;")
        fusion_v.addWidget(fusion_desc)
        sep_f = QFrame(); sep_f.setFrameShape(QFrame.HLine); sep_f.setStyleSheet("color:#333333;"); fusion_v.addWidget(sep_f)
        f_fusion = add_section("Fusão Final", fusion_v)
        lbl_fusion = QLabel("Modo de Fusão")
        self.fusion_mode_cb = QComboBox(); self.fusion_mode_cb.addItems(["or", "weighted"])
        self.fusion_mode_cb.setCurrentText(str(self.params["fusion_mode"]))
        self.fusion_mode_cb.currentTextChanged.connect(lambda val: self._on_text_param_change("fusion_mode", val))
        self.fusion_mode_cb.setToolTip(self._tooltips.get("fusion_mode", ""))
        f_fusion.addRow(lbl_fusion, self.fusion_mode_cb)
        add_dspin("Peso estrutura", "w_struct", 0.0, 1.0, f_fusion, step=0.05, dec=2)
        add_dspin("Peso top-hat", "w_top", 0.0, 1.0, f_fusion, step=0.05, dec=2)
        add_dspin("Peso black-hat", "w_black", 0.0, 1.0, f_fusion, step=0.05, dec=2)
        add_dspin("Peso cor", "w_color", 0.0, 1.0, f_fusion, step=0.05, dec=2)
        add_dspin("Percentil fusão", "fused_percentile", 90.0, 99.9, f_fusion, step=0.1, dec=1)
        # Final map mode selector
        lbl_finalmode = QLabel("Mapa Final")
        self.final_mode_cb = QComboBox(); self.final_mode_cb.addItems(["classic", "extended"])
        self.final_mode_cb.setCurrentText(str(self.params.get("final_mode", "extended")))
        self.final_mode_cb.currentTextChanged.connect(lambda val: self._on_text_param_change("final_mode", val))
        self.final_mode_cb.setToolTip(self._tooltips.get("final_mode", ""))
        f_fusion.addRow(lbl_finalmode, self.final_mode_cb)
        # Include Gradient in Final (classic)
        add_check("Incluir Gradiente no Final", "final_include_gradient", f_fusion)
        fusion_idx = tabs.addTab(fusion_page, "🔗 Fusion")
        tabs.setTabToolTip(fusion_idx, "Combinação das pistas (estrutura, morfologia, cor) em mapa final.")
        # Always-visible defect count (outside tabs)
        self.defect_count_label = QLabel("Total de defeitos: 0")
        self.defect_count_label.setStyleSheet("font-weight:600; font-size:14px;")
        left_wrap.addWidget(self.defect_count_label)
        left_wrap.addWidget(controls_scroll)
        top_row.addWidget(left_card, 2)

        # Coluna imagem (em card)
        right_card = QFrame()
        right_card.setStyleSheet("background:#1e1e1e; border:1px solid #333333; border-radius:8px;")
        self.image_layout = QVBoxLayout(right_card)
        self.image_layout.setContentsMargins(12, 12, 12, 12)
        self.image_layout.setSpacing(10)
        top_row.addWidget(right_card, 3)

        # Barra inferior (footer escuro)
        footer = QFrame()
        footer.setStyleSheet("background:#151515; border-top:1px solid #333333;")
        self.bottom_bar = QHBoxLayout(footer)
        self.bottom_bar.setContentsMargins(12, 8, 12, 8)
        self.bottom_bar.setSpacing(10)
        main_layout.addWidget(footer, 0)

        # --- Widget de imagem ---
        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT)
        self.image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Old inline control builders removed in favor of tabbed forms above


        # --- Botões (BARRA INFERIOR) ---
        self.btn_update = QPushButton("Atualizar Detecção (SPACE)")
        self.btn_update.clicked.connect(self._update_preview)
        self.bottom_bar.addWidget(self.btn_update)

        self.btn_export = QPushButton("Exportar Imagem (Ctrl+E)")
        self.btn_export.clicked.connect(self._export_annotated_image)
        self.bottom_bar.addWidget(self.btn_export)

        self.btn_save_params = QPushButton("Guardar Parâmetros (Ctrl+S)")
        self.btn_save_params.clicked.connect(self._save_current_params)
        self.bottom_bar.addWidget(self.btn_save_params)

        self.btn_reset = QPushButton("Reset Valores Padrão")
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        self.bottom_bar.addWidget(self.btn_reset)

        self.bottom_bar.addStretch(1)  # empurra combos para a direita

        # --- Comboboxes (BARRA INFERIOR) ---
        self.bottom_bar.addWidget(QLabel("Tipo de Visualização"))
        self.view_mode = QComboBox()
        self.view_mode.addItems([
            "Final", "Escuro", "Amarelo", "Azul", "Vermelho", "Gradiente", "Todos (colorido)",
            "DEBUG: Diff escuro (CLAHE)",
            "DEBUG: Diff escuro (sem CLAHE)"
        ])
        self.view_mode.currentIndexChanged.connect(self._update_preview)
        self.bottom_bar.addWidget(self.view_mode)

        self.bottom_bar.addSpacing(16)

        self.bottom_bar.addWidget(QLabel("Modo de Fundo"))
        self.display_mode = QComboBox()
        self.display_mode.addItems(["PB", "Colorida"])
        self.display_mode.currentIndexChanged.connect(self._update_preview)
        self.bottom_bar.addWidget(self.display_mode)

        # Heatmap como fundo (aplica a todos os modos)
        self.chk_heatmap_bg = QCheckBox("Heatmap fundo")
        self.chk_heatmap_bg.setChecked(bool(int(self.params.get("use_heatmap_bg", 0))))
        self.chk_heatmap_bg.setToolTip("Usa o mapa de calor do 'Diff Escuro (CLAHE)' como fundo para todas as visualizações.")
        self.chk_heatmap_bg.stateChanged.connect(lambda s: self._on_checkbox_change("use_heatmap_bg", s))
        self.bottom_bar.addWidget(self.chk_heatmap_bg)

        # Defect count label moved above tabs (outside tabs)

        # Timer de debounce (150ms)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self._debounced_update)

        # Atalhos
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._update_preview)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_current_params)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._export_annotated_image)

        # Primeira atualização
        self._update_preview()

    # ---------- UI helpers ----------
    def _sync_params_from_ui(self):
        """Read current widget states into self.params before saving."""
        try:
            for key, w in self.spin_boxes.items():
                if hasattr(w, "isChecked"):
                    self.params[key] = 1 if bool(w.isChecked()) else 0
                elif hasattr(w, "value"):
                    val = w.value()
                    if isinstance(val, float):
                        self.params[key] = float(val)
                    else:
                        self.params[key] = int(val)
            if hasattr(self, "color_metric_cb"):
                self.params["color_metric"] = str(self.color_metric_cb.currentText())
            if hasattr(self, "fusion_mode_cb"):
                self.params["fusion_mode"] = str(self.fusion_mode_cb.currentText())
        except Exception as e:
            print("[Tuner] _sync_params_from_ui warning:", e)

    def _create_spinbox(self, label_text, param_name, min_val, max_val, target_layout):
        label = QLabel(label_text)
        target_layout.addWidget(label)

        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(int(self.params[param_name]))
        spin.valueChanged.connect(lambda val, p=param_name: self._on_slider_change(p, val))
        target_layout.addWidget(spin)

        self.spin_boxes[param_name] = spin

    def _create_doublespinbox(self, label_text, param_name, min_val, max_val, target_layout, step=0.1, decimals=2):
        label = QLabel(label_text)
        target_layout.addWidget(label)

        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setRange(min_val, max_val)
        spin.setValue(float(self.params[param_name]))
        spin.valueChanged.connect(lambda val, p=param_name: self._on_slider_change_float(p, val))
        target_layout.addWidget(spin)

        self.spin_boxes[param_name] = spin

    def _create_checkbox(self, label_text, param_name, target_layout):
        cb = QCheckBox(label_text)
        cb.setChecked(bool(int(self.params[param_name])))
        cb.stateChanged.connect(lambda state, p=param_name: self._on_checkbox_change(p, state))
        target_layout.addWidget(cb)
        self.spin_boxes[param_name] = cb


    def _on_slider_change_float(self, param_name, value):
        self.params[param_name] = float(value)
        if not self._update_scheduled:
            self._update_scheduled = True
            self.timer.start()

    def _on_checkbox_change(self, param_name, state):
        self.params[param_name] = 1 if state == Qt.Checked else 0
        if not self._update_scheduled:
            self._update_scheduled = True
            self.timer.start()

    def _on_text_param_change(self, param_name, value):
        self.params[param_name] = str(value)
        if not self._update_scheduled:
            self._update_scheduled = True
            self.timer.start()

    def _load_saved_params(self, path="config/inspection_params.json"):
        """Carrega params guardados e faz merge com defaults.
        Aceita 'detect_area' (antigo) como alias de 'min_defect_area'."""
        try:
            if not os.path.isfile(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}

            # alias antigo -> novo
            if "detect_area" in data and "min_defect_area" not in data:
                data["min_defect_area"] = data["detect_area"]

            merged = self.params.copy()

            # chaves por tipo
            float_keys = {
                "msssim_percentile", "msssim_weight",
                "msssim_sigma_s1", "msssim_sigma_s2", "msssim_sigma_s3",
                # novos
                "th_top_percentile", "th_black_percentile",
                "color_percentile", "w_struct", "w_top", "w_black", "w_color",
                "fused_percentile"
            }
            bool_keys = {"use_ms_ssim", "use_morph_maps", "use_color_delta", "use_heatmap_bg", "ignore_overexposed"}
            string_keys = {"color_metric", "fusion_mode", "final_mode"}

            for k in list(merged.keys()):
                if k not in data:
                    continue
                v = data[k]
                try:
                    if k in bool_keys:
                        # robust boolean parsing: accepts 1/0, True/False, "true"/"false", "yes"/"no"
                        if isinstance(v, (int, np.integer)):
                            v = 1 if int(v) != 0 else 0
                        elif isinstance(v, bool):
                            v = 1 if v else 0
                        elif isinstance(v, str):
                            v_norm = v.strip().lower()
                            v = 1 if v_norm in {"1", "true", "yes", "on"} else 0
                        else:
                            v = 1 if bool(v) else 0
                    elif k in float_keys:
                        v = float(v)
                    elif k in string_keys:
                        v = str(v)
                    else:
                        v = int(v)
                except Exception:
                    continue

                # validações simples
                if "kernel" in k:
                    if v < 1: v = 1
                    if isinstance(v, (int, np.integer)) and v % 2 == 0:
                        v = v + 1
                # SEs devem ser ímpares >=1
                if k in {"se_top", "se_black"}:
                    v = max(1, int(v))
                    if v % 2 == 0:
                        v += 1
                if "iterations" in k:
                    v = max(0, int(v))
                if k.endswith("_threshold") or "threshold" in k:
                    if k not in float_keys:  # os threshold que tens são ints 0..255
                        v = max(0, min(255, int(v)))
                if k == "min_defect_area":
                    v = max(1, int(v))
                # percentis 0..100
                if k.endswith("percentile"):
                    try:
                        v = float(v)
                        if v < 0: v = 0.0
                        if v > 100: v = 100.0
                    except Exception:
                        pass
                # pesos 0..1
                if k in {"w_struct","w_top","w_black","w_color"}:
                    try:
                        v = float(v)
                        if v < 0: v = 0.0
                        if v > 1: v = 1.0
                    except Exception:
                        pass

                merged[k] = v

            self.params = merged
            print("[Tuner] Parâmetros carregados:", self.params)
        except Exception as e:
            print(f"[Tuner] Falha a carregar '{path}':", e)


    def _on_slider_change(self, param_name, value):
        # Para kernels, garantir ímpar (morfologia costuma beneficiar)
        if "kernel" in param_name and value % 2 == 0:
            value = value + 1
            self.spin_boxes[param_name].blockSignals(True)
            self.spin_boxes[param_name].setValue(value)
            self.spin_boxes[param_name].blockSignals(False)

        self.params[param_name] = int(value)
        # Debounce
        if not self._update_scheduled:
            self._update_scheduled = True
            self.timer.start()

    def _reset_to_defaults(self):
        defaults = {
            "dark_threshold": 30,
            "bright_threshold": 30,
            "blue_threshold": 25,
            "red_threshold": 25,
            "dark_morph_kernel_size": 3,
            "dark_morph_iterations": 1,
            "bright_morph_kernel_size": 3,
            "bright_morph_iterations": 1,
            "dark_gradient_threshold": 10,
            "min_defect_area": 1
        }
        self.params.update(defaults)
        for key, spin in self.spin_boxes.items():
            spin.blockSignals(True)
            spin.setValue(int(self.params[key]))
            spin.blockSignals(False)
        self._update_preview()

    def _debounced_update(self):
        self._update_scheduled = False
        self._update_preview()

    @staticmethod
    def _normalize_lab_to_template(tpl_bgr, img_bgr, mask):
        eps = 1e-6
        tpl_lab = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        img_lab = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2LAB).astype(np.float32)
        m = (mask > 0)
        norm = img_lab.copy()
        for c in range(3):
            mu_t = float(np.mean(tpl_lab[:,:,c][m])); sd_t = float(np.std(tpl_lab[:,:,c][m]) + eps)
            mu_i = float(np.mean(img_lab[:,:,c][m])); sd_i = float(np.std(img_lab[:,:,c][m]) + eps)
            gain = sd_t / sd_i
            bias = mu_t - gain * mu_i
            norm[:,:,c] = gain * img_lab[:,:,c] + bias
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _morph(mask_in, k, it):
        if mask_in is None or k <= 1 or it <= 0:
            return mask_in
        k_eff = k if k % 2 == 1 else (k + 1)
        kernel = np.ones((k_eff, k_eff), np.uint8)
        m = cv2.morphologyEx(mask_in, cv2.MORPH_OPEN, kernel, iterations=it)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=it)
        return m


    # ---------- Core ----------
    @staticmethod
    def _to_binary_mask(mask_like):
        """Garante máscara uint8 0/255 com shape 2D."""
        if mask_like is None:
            return None
        m = mask_like
        if len(m.shape) == 3:
            # se vier BGR, converter para cinza
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        m = (m > 0).astype(np.uint8) * 255
        return m

    def _update_preview(self):
        if self.tpl is None or self.aligned is None or self.mask is None:
            return

        # --- parâmetros atuais ---
        try:
            dark_th   = int(self.params["dark_threshold"])
            bright_th = int(self.params["bright_threshold"])
            blue_th   = int(self.params["blue_threshold"])
            red_th    = int(self.params["red_threshold"])
            dark_k    = int(self.params["dark_morph_kernel_size"])
            dark_it   = int(self.params["dark_morph_iterations"])
            color_k   = int(self.params["bright_morph_kernel_size"])
            color_it  = int(self.params["bright_morph_iterations"])
            dark_grad = int(self.params["dark_gradient_threshold"])
            min_area  = int(self.params["min_defect_area"])
        except Exception as e:
            print("Erro conversão parâmetros:", e)
            return

        # --- máscara binária 0/255 + aplica máscara UMA vez ---
        mask_bin = (self.mask > 0).astype(np.uint8) * 255
        tpl_m = cv2.bitwise_and(self.tpl,     self.tpl,     mask=mask_bin)
        ali_m = cv2.bitwise_and(self.aligned, self.aligned, mask=mask_bin)

        # --- diagnóstico: há diferenças mesmo? ---
        t_gray = cv2.cvtColor(tpl_m, cv2.COLOR_BGR2GRAY)
        a_gray = cv2.cvtColor(ali_m, cv2.COLOR_BGR2GRAY)
        diff_noeq = cv2.subtract(t_gray, a_gray)           # detectar “pontos pretos”: template - aligned
        nz_noeq   = cv2.countNonZero(diff_noeq)

        # se quiseres ver “o hotspot” mais diferente:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(diff_noeq)
        print(f"[DEBUG] nz_noeq={nz_noeq}, maxDiff={maxVal} @ {maxLoc}")

        # Se maxDiff==0, as imagens estão iguais na máscara → volta ao passo 1)

        # --- normalização fotométrica (condicional) ---
        tpl_mean = cv2.mean(cv2.cvtColor(tpl_m, cv2.COLOR_BGR2GRAY), mask=mask_bin)[0]
        ali_mean = cv2.mean(cv2.cvtColor(ali_m, cv2.COLOR_BGR2GRAY), mask=mask_bin)[0]
        if abs(tpl_mean - ali_mean) > 1.0:
            ali_m = self._normalize_lab_to_template(tpl_m, ali_m, mask_bin)

        # --- diffs para DEBUG ---
        # sem CLAHE
        t_gray_raw = cv2.cvtColor(tpl_m, cv2.COLOR_BGR2GRAY)
        a_gray_raw = cv2.cvtColor(ali_m, cv2.COLOR_BGR2GRAY)
        diff_raw_noeq = cv2.subtract(t_gray_raw, a_gray_raw)
        diff_raw_noeq = cv2.bitwise_and(diff_raw_noeq, diff_raw_noeq, mask=mask_bin)

        # com CLAHE (igual ao teu detector)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        t_gray_eq = clahe.apply(cv2.GaussianBlur(t_gray_raw, (3, 3), 0))
        a_gray_eq = clahe.apply(cv2.GaussianBlur(a_gray_raw, (3, 3), 0))
        diff_raw_eq = cv2.subtract(t_gray_eq, a_gray_eq)
        diff_raw_eq = cv2.bitwise_and(diff_raw_eq, diff_raw_eq, mask=mask_bin)

        # (opcional) marcar o hotspot no preview:
        debug_vis = ali_m.copy()
        cv2.circle(debug_vis, maxLoc, 40, (0,255,255), 2, cv2.LINE_AA)
        # depois mostrares o debug_vis no lugar do preview para ver se faz sentido

        # --- Short-circuit: Escuro mode fast path (skip heavy extras) ---
        mode_fast = self.view_mode.currentText()
        if mode_fast == "Escuro":
            use_heatmap_bg = bool(int(self.params.get("use_heatmap_bg", 0)))
            # Downscale for speed (process at <= 1200px on the long side)
            H0, W0 = t_gray_raw.shape
            target_max = 1200
            scale = min(1.0, float(target_max) / float(max(H0, W0)))
            inv_scale = 1.0 / scale

            if scale < 1.0:
                t_small = cv2.resize(t_gray_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                a_small = cv2.resize(a_gray_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                m_small = cv2.resize(mask_bin, (t_small.shape[1], t_small.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                t_small, a_small, m_small = t_gray_raw, a_gray_raw, mask_bin

            # Build dark score on blurred grayscale + micro blackhat, gated by gradient (small)
            t_blur_hm = cv2.GaussianBlur(t_small, (5, 5), 0)
            a_blur_hm = cv2.GaussianBlur(a_small, (5, 5), 0)
            hm_dark_s = cv2.subtract(t_blur_hm, a_blur_hm)
            # gradient gate (small)
            a_blur_grad = cv2.GaussianBlur(a_small, (5, 5), 0)
            hm_grad_s = cv2.morphologyEx(a_blur_grad, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
            if dark_grad <= 0:
                gradient_mask_dark_s = m_small.copy()
            else:
                _, gradient_mask_dark_s = cv2.threshold(hm_grad_s, int(dark_grad), 255, cv2.THRESH_BINARY)
            gradient_mask_dark_s = cv2.bitwise_and(gradient_mask_dark_s, m_small)
            # threshold main dark (small)
            _, dark_thr_s = cv2.threshold(hm_dark_s, int(dark_th), 255, cv2.THRESH_BINARY)
            dark_thr_s = cv2.bitwise_and(dark_thr_s, gradient_mask_dark_s)
            # micro dark via blackhat (small)
            bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            bh_tpl = cv2.morphologyEx(t_blur_hm, cv2.MORPH_BLACKHAT, bh_kernel)
            bh_cur = cv2.morphologyEx(a_blur_hm, cv2.MORPH_BLACKHAT, bh_kernel)
            hm_bh_s = cv2.subtract(bh_cur, bh_tpl)
            bh_th = max(6, min(20, int(max(dark_th, 0)) // 2 + 6))
            _, micro_dark_s = cv2.threshold(hm_bh_s, bh_th, 255, cv2.THRESH_BINARY)
            micro_dark_s = cv2.bitwise_and(micro_dark_s, m_small)
            dark_mask_s = cv2.bitwise_or(dark_thr_s, micro_dark_s)
            # morphology cleanup (small)
            escuro_clean_s = self._morph(dark_mask_s, dark_k, dark_it)

            # Base image at original scale
            if use_heatmap_bg:
                # Apply gradient gate only to grayscale diff; micro-blackhat remains un-gated (like detector)
                diff_gated_s = cv2.bitwise_and(hm_dark_s, gradient_mask_dark_s)
                bh_roi_s = cv2.bitwise_and(hm_bh_s, m_small)
                score_dark_s = cv2.max(diff_gated_s, bh_roi_s)
                score_dark_s = cv2.bitwise_and(score_dark_s, escuro_clean_s)
                score_dark = cv2.resize(score_dark_s, (W0, H0), interpolation=cv2.INTER_LINEAR)
                hm = cv2.normalize(score_dark, None, 0, 255, cv2.NORM_MINMAX)
                hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                preview = cv2.bitwise_and(hm, hm, mask=mask_bin)
            else:
                if self.display_mode.currentText() == "PB":
                    preview = cv2.cvtColor(ali_m, cv2.COLOR_BGR2GRAY)
                    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                else:
                    preview = ali_m.copy()

            # Draw larger white circles per contour, scaled to original size
            num_defeitos = 0
            cnts_s, _ = cv2.findContours(escuro_clean_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts_s:
                area_s = cv2.contourArea(cnt)
                area = area_s * (inv_scale * inv_scale)
                if area < min_area:
                    continue
                (cx_s, cy_s), r_s = cv2.minEnclosingCircle(cnt)
                cx = int(round(cx_s * inv_scale))
                cy = int(round(cy_s * inv_scale))
                r  = float(r_s) * inv_scale
                rad_draw = max(22, int(round(r * 1.4)) + 10)
                cv2.circle(preview, (cx, cy), rad_draw, (255, 255, 255), 3, cv2.LINE_AA)
                num_defeitos += 1
            # Update UI and return
            self.defect_count_label.setText(f"Total de defeitos: {num_defeitos}")
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            h, w, ch = preview_rgb.shape
            qt_image = QImage(preview_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            self.last_preview = preview.copy()
            return

        # --- detecção real (usa os MESMOS tensores que visualizas) ---
        # Final mode toggle: classic = union of 4 maps (no fusion extras); extended = with fusion extras
        _final_mode = str(self.params.get("final_mode", "extended")).strip().lower()
        _classic = (_final_mode == "classic")
        final_mask, _, dark_mask_filt, bright_mask_raw, blue_mask_raw, red_mask_raw, msssim_mask, fused_mask = detect_defects(
            tpl_m, ali_m, mask_bin,
            dark_th, bright_th,
            dark_k,  dark_it,
            color_k, color_it,
            min_area, dark_grad,
            blue_th, red_th,
            # ---- MS-SSIM ----
            use_ms_ssim=(False if _classic else bool(int(self.params["use_ms_ssim"]))),
            msssim_percentile=float(self.params["msssim_percentile"]),
            msssim_weight=(0.0 if _classic else float(self.params["msssim_weight"])) ,
            msssim_kernel_sizes=(
                int(self.params["msssim_kernel_size_s1"]),
                int(self.params["msssim_kernel_size_s2"]),
                int(self.params["msssim_kernel_size_s3"])
            ),
            msssim_sigmas=(
                float(self.params["msssim_sigma_s1"]),
                float(self.params["msssim_sigma_s2"]),
                float(self.params["msssim_sigma_s3"])
            ),
            msssim_morph_kernel_size=int(self.params["msssim_morph_kernel_size"]),
            msssim_morph_iterations=int(self.params["msssim_morph_iterations"]),
            # ---- Overexposure ----
            ignore_overexposed=bool(int(self.params.get("ignore_overexposed", 0))),
            # ---- Mapas adicionais + Fusão ----
            use_morph_maps=(False if _classic else bool(int(self.params["use_morph_maps"]))),
            th_top_percentile=float(self.params["th_top_percentile"]),
            th_black_percentile=float(self.params["th_black_percentile"]),
            se_top=int(self.params["se_top"]),
            se_black=int(self.params["se_black"]),
            use_color_delta=(False if _classic else bool(int(self.params["use_color_delta"]))),
            color_metric=str(self.params["color_metric"]),
            color_percentile=float(self.params["color_percentile"]),
            fusion_mode=str(self.params["fusion_mode"]),
            w_struct=float(self.params["w_struct"]),
            w_top=float(self.params["w_top"]),
            w_black=float(self.params["w_black"]),
            w_color=float(self.params["w_color"]),
            fused_percentile=float(self.params["fused_percentile"]),
            return_msssim=True,
            return_fusion=True,
        )


        # --- DEBUG: contagens de “sinal” ---
        print("[DEBUG] mask nz:", cv2.countNonZero(mask_bin),
            "| diff(eq) nz:", cv2.countNonZero(diff_raw_eq),
            "| diff(noeq) nz:", cv2.countNonZero(diff_raw_noeq),
            "| dark nz:", cv2.countNonZero(dark_mask_filt),
            "| final nz:", cv2.countNonZero(final_mask))

        # --- base de imagem ---
        use_heatmap_bg = bool(int(self.params.get("use_heatmap_bg", 0)))
        if use_heatmap_bg:
            heat_base = cv2.normalize(diff_raw_eq, None, 0, 255, cv2.NORM_MINMAX)
            heat_base = cv2.applyColorMap(heat_base, cv2.COLORMAP_JET)
            preview = cv2.bitwise_and(heat_base, heat_base, mask=mask_bin)
        else:
            if self.display_mode.currentText() == "PB":
                preview = cv2.cvtColor(ali_m, cv2.COLOR_BGR2GRAY)
                preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            else:
                preview = ali_m.copy()

        # --- desenho / modos ---
        num_defeitos = 0

        def draw_mask(mask, color):
            nonlocal num_defeitos, preview
            if mask is None: return
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if cv2.contourArea(cnt) < min_area:
                    continue
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                rad_draw = max(14, int(radius) + 6)
                cv2.circle(preview, (int(x), int(y)), rad_draw, color, 3, cv2.LINE_AA)
                num_defeitos += 1

        mode = self.view_mode.currentText()
        # Heatmap helpers per mode
        def make_heatmap(src_u8):
            hm = cv2.normalize(src_u8, None, 0, 255, cv2.NORM_MINMAX)
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            return cv2.bitwise_and(hm, hm, mask=mask_bin)
        t_blur_hm = cv2.GaussianBlur(t_gray_raw, (5, 5), 0)
        a_blur_hm = cv2.GaussianBlur(a_gray_raw, (5, 5), 0)
        hm_dark = cv2.subtract(t_blur_hm, a_blur_hm)
        tpl_lab_hm = cv2.cvtColor(tpl_m, cv2.COLOR_BGR2LAB)
        ali_lab_hm = cv2.cvtColor(ali_m,  cv2.COLOR_BGR2LAB)
        hm_yel = cv2.subtract(ali_lab_hm[:, :, 2], tpl_lab_hm[:, :, 2])
        hm_blue = cv2.subtract(tpl_lab_hm[:, :, 2], ali_lab_hm[:, :, 2])
        hm_red  = cv2.subtract(ali_lab_hm[:, :, 1], tpl_lab_hm[:, :, 1])
        a_blur_grad = cv2.GaussianBlur(a_gray_raw, (5, 5), 0)
        hm_grad = cv2.morphologyEx(a_blur_grad, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))

        # Enforce classic final mode as union of Escuro, Gradiente, Amarelo, Azul, Vermelho
        if _classic:
            if dark_grad <= 0:
                gradient_mask_dark = mask_bin.copy()
            else:
                _, gradient_mask_dark = cv2.threshold(hm_grad, int(dark_grad), 255, cv2.THRESH_BINARY)
            gradient_mask_dark = cv2.bitwise_and(gradient_mask_dark, mask_bin)

            escuro_clean_final   = self._morph(dark_mask_filt, dark_k, dark_it)
            amarelo_clean_final  = self._morph(bright_mask_raw, color_k, color_it)
            azul_clean_final     = self._morph(blue_mask_raw,   color_k, color_it)
            vermelho_clean_final = self._morph(red_mask_raw,    color_k, color_it)

            final_union = cv2.bitwise_or(escuro_clean_final, amarelo_clean_final)
            final_union = cv2.bitwise_or(final_union,        azul_clean_final)
            final_union = cv2.bitwise_or(final_union,        vermelho_clean_final)
            if bool(int(self.params.get("final_include_gradient", 1))):
                final_union = cv2.bitwise_or(final_union, gradient_mask_dark)
            final_mask = final_union

        if mode == "DEBUG: Diff escuro (CLAHE)":
            # Mapa de calor do diff escuro (com CLAHE) como fundo, se ativado
            if use_heatmap_bg:
                heat = cv2.normalize(diff_raw_eq, None, 0, 255, cv2.NORM_MINMAX)
                heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                preview = cv2.bitwise_and(heat, heat, mask=mask_bin)
            # sobrepõe o resultado com os parâmetros atuais para ver mudanças em tempo real
            draw_mask(final_mask, (0, 255, 0))

        elif mode == "DEBUG: Diff escuro (sem CLAHE)":
            # Mapa de calor do diff escuro (sem CLAHE) como fundo, se ativado
            if use_heatmap_bg:
                heat = cv2.normalize(diff_raw_noeq, None, 0, 255, cv2.NORM_MINMAX)
                heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                preview = cv2.bitwise_and(heat, heat, mask=mask_bin)
            # sobrepõe o resultado com os parâmetros atuais para ver mudanças em tempo real
            draw_mask(final_mask, (0, 255, 0))

        elif mode == "Escuro":
            # Build detector-like dark score (grayscale diff + blackhat), gate by gradient
            # Then mask by the actually detected dark mask (post-morph) for strict alignment
            escuro_clean = self._morph(dark_mask_filt, dark_k, dark_it)
            if use_heatmap_bg:
                bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                bh_tpl = cv2.morphologyEx(t_blur_hm, cv2.MORPH_BLACKHAT, bh_kernel)
                bh_cur = cv2.morphologyEx(a_blur_hm, cv2.MORPH_BLACKHAT, bh_kernel)
                hm_bh = cv2.subtract(bh_cur, bh_tpl)

                morph_grad = hm_grad  # gradiente de a_blur_hm
                if dark_grad <= 0:
                    gradient_mask_dark = mask_bin.copy()
                else:
                    _, gradient_mask_dark = cv2.threshold(morph_grad, int(dark_grad), 255, cv2.THRESH_BINARY)
                gradient_mask_dark = cv2.bitwise_and(gradient_mask_dark, mask_bin)

                # Gate only the gray diff; do not gate micro-blackhat
                diff_gated = cv2.bitwise_and(hm_dark, gradient_mask_dark)
                bh_roi = cv2.bitwise_and(hm_bh, mask_bin)
                score_dark = cv2.max(diff_gated, bh_roi)
                score_dark = cv2.bitwise_and(score_dark, escuro_clean)
                preview = make_heatmap(score_dark)

            # Draw exact outlines in black but clip strictly to the detected mask
            cnts, _ = cv2.findContours(escuro_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if cv2.contourArea(cnt) < min_area:
                    continue
                stroke = np.zeros(escuro_clean.shape, dtype=np.uint8)
                cv2.drawContours(stroke, [cnt], -1, 255, 3, cv2.LINE_AA)
                stroke = cv2.bitwise_and(stroke, escuro_clean)
                preview[stroke > 0] = (0, 0, 0)
                num_defeitos += 1

        elif mode == "Amarelo":
            if use_heatmap_bg:
                preview = make_heatmap(hm_yel)
            draw_mask(bright_mask_raw, (0, 255, 255))
        elif mode == "Azul":
            if use_heatmap_bg:
                preview = make_heatmap(hm_blue)
            draw_mask(blue_mask_raw,   (255, 255, 0))
        elif mode == "Vermelho":
            if use_heatmap_bg:
                preview = make_heatmap(hm_red)
            draw_mask(red_mask_raw,    (0, 0, 255))
        elif mode == "Gradiente":
            # Reconstroi o gate de gradiente em L (como no detector)
            morph_grad = hm_grad
            if use_heatmap_bg:
                preview = make_heatmap(morph_grad)
            if dark_grad <= 0:
                gradient_mask_dark = mask_bin.copy()
            else:
                _, gradient_mask_dark = cv2.threshold(morph_grad, int(dark_grad), 255, cv2.THRESH_BINARY)
            gradient_mask_dark = cv2.bitwise_and(gradient_mask_dark, mask_bin)
            draw_mask(gradient_mask_dark, (255, 0, 255))
        elif mode == "Todos (colorido)":
            if use_heatmap_bg:
                n1 = cv2.normalize(hm_dark, None, 0, 255, cv2.NORM_MINMAX)
                n2 = cv2.normalize(hm_yel,  None, 0, 255, cv2.NORM_MINMAX)
                n3 = cv2.normalize(hm_blue, None, 0, 255, cv2.NORM_MINMAX)
                n4 = cv2.normalize(hm_red,  None, 0, 255, cv2.NORM_MINMAX)
                n5 = cv2.normalize(hm_grad, None, 0, 255, cv2.NORM_MINMAX)
                combo = cv2.max(n1, cv2.max(n2, cv2.max(n3, cv2.max(n4, n5))))
                preview = make_heatmap(combo.astype(np.uint8))
            draw_mask(dark_mask_filt,  (255, 0, 0))
            draw_mask(bright_mask_raw, (0, 255, 255))
            draw_mask(blue_mask_raw,   (255, 255, 0))
            draw_mask(red_mask_raw,    (0, 0, 255))
            # também mostra o gradiente (magenta)
            morph_grad = hm_grad
            if dark_grad <= 0:
                gradient_mask_dark = mask_bin.copy()
            else:
                _, gradient_mask_dark = cv2.threshold(morph_grad, int(dark_grad), 255, cv2.THRESH_BINARY)
            gradient_mask_dark = cv2.bitwise_and(gradient_mask_dark, mask_bin)
            draw_mask(gradient_mask_dark, (255, 0, 255))
        else:  # Final
            if use_heatmap_bg:
                n1 = cv2.normalize(hm_dark, None, 0, 255, cv2.NORM_MINMAX)
                n2 = cv2.normalize(hm_yel,  None, 0, 255, cv2.NORM_MINMAX)
                n3 = cv2.normalize(hm_blue, None, 0, 255, cv2.NORM_MINMAX)
                n4 = cv2.normalize(hm_red,  None, 0, 255, cv2.NORM_MINMAX)
                combo = cv2.max(n1, cv2.max(n2, cv2.max(n3, n4)))
                preview = make_heatmap(combo.astype(np.uint8))
            draw_mask(final_mask, (0, 255, 0))

        # --- UI ---
        self.defect_count_label.setText(f"Total de defeitos: {num_defeitos}")

        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        h, w, ch = preview_rgb.shape
        qt_image = QImage(preview_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        self.last_preview = preview.copy()

    # ---------- Export / Save ----------
    def _export_annotated_image(self):
        if self.last_preview is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar Imagem", "", "PNG Files (*.png)")
        if path:
            cv2.imwrite(path, self.last_preview)
            print(f"Imagem exportada para {path}")

    def closeEvent(self, event):
        try:
            # auto-save params on close so user toggles persist without Ctrl+S
            self._save_current_params()
        except Exception as e:
            print("[Tuner] Auto-save on close failed:", e)
        event.accept()

    def _save_current_params(self):
        # Ensure we capture the latest UI state
        self._sync_params_from_ui()
        os.makedirs("config", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # preserva tipos
        params_to_save = {}
        float_keys = {
            "msssim_percentile", "msssim_weight",
            "msssim_sigma_s1", "msssim_sigma_s2", "msssim_sigma_s3",
            "th_top_percentile", "th_black_percentile",
            "color_percentile", "w_struct", "w_top", "w_black", "w_color",
            "fused_percentile"
        }
        bool_keys = {"use_ms_ssim", "use_morph_maps", "use_color_delta", "use_heatmap_bg", "ignore_overexposed", "final_include_gradient"}
        
        string_keys = {"color_metric", "fusion_mode", "final_mode"}
        for k, v in self.params.items():
            if k in bool_keys:
                params_to_save[k] = 1 if bool(int(v)) else 0
            elif k in float_keys:
                params_to_save[k] = float(v)
            elif k in string_keys:
                params_to_save[k] = str(v)
            else:
                params_to_save[k] = int(v)

        # alias para compatibilidade
        params_to_save["detect_area"] = int(self.params["min_defect_area"])

        with open("config/inspection_params.json", "w", encoding="utf-8") as f:
            json.dump(params_to_save, f, indent=4)

        # Log CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = "logs/param_history.csv"
        file_exists = os.path.isfile(log_path)

        field_order = ["timestamp", "user", "user_type"] + [
            # existentes
            "dark_threshold","bright_threshold","blue_threshold","red_threshold",
            "dark_morph_kernel_size","dark_morph_iterations",
            "bright_morph_kernel_size","bright_morph_iterations",
            "dark_gradient_threshold","min_defect_area","detect_area","ignore_overexposed","use_heatmap_bg",
            # novos MS-SSIM
            "use_ms_ssim","msssim_percentile","msssim_weight",
            "msssim_kernel_size_s1","msssim_kernel_size_s2","msssim_kernel_size_s3",
            "msssim_sigma_s1","msssim_sigma_s2","msssim_sigma_s3",
            "msssim_morph_kernel_size","msssim_morph_iterations",
            # novos mapas/fusão
            "use_morph_maps","th_top_percentile","th_black_percentile","se_top","se_black",
            "use_color_delta","color_metric","color_percentile",
            "fusion_mode","w_struct","w_top","w_black","w_color","fused_percentile"
        ]

        row_dict = {
            "timestamp": timestamp,
            "user": self.user_name,
            "user_type": self.user_type,
            **params_to_save
        }

        with open(log_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_order, delimiter=';')
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

        print("Parâmetros guardados com sucesso.")
