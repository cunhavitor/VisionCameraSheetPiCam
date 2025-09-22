import os
import json
import cv2
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QScrollArea
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QPolygon
from PySide6.QtCore import Qt, QPoint

from config.config import TEMPLATE_IMAGE_PATH


class ClickableImage(QLabel):
    """QLabel customizado que suporta clique/arrasto para selecionar área."""
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.start_pos = None
        self.end_pos = None
        self.rubber_band_active = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = self.start_pos
            self.rubber_band_active = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.rubber_band_active:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rubber_band_active:
            self.end_pos = event.pos()
            self.rubber_band_active = False
            self.update()
            self.parent_window.process_crop(self.start_pos, self.end_pos)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rubber_band_active and self.start_pos and self.end_pos:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            rect = (
                min(self.start_pos.x(), self.end_pos.x()),
                min(self.start_pos.y(), self.end_pos.y()),
                abs(self.start_pos.x() - self.end_pos.x()),
                abs(self.start_pos.y() - self.end_pos.y())
            )
            painter.drawRect(*rect)


class CriarFormaWindow(QDialog):
    def __init__(self, parent_window):
        super().__init__()
        self.setWindowTitle("Definir Forma Manual do Polígono")
        self.parent_window = parent_window
        self.setGeometry(100, 100, 1000, 750)

        # === Carregar imagem ===
        self.original_image = Image.open(TEMPLATE_IMAGE_PATH)
        self.img_width, self.img_height = self.original_image.size

        # Fator inicial de zoom
        self.scale_factor = 0.6
        self._update_scaled_image()

        # === Layout principal ===
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # === Área com scroll ===
        self.image_label = ClickableImage(self)
        self.image_label.setPixmap(self._get_scaled_pixmap())
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # === Botões ===
        btn_layout = QHBoxLayout()
        main_layout.addLayout(btn_layout)

        self.btn_undo = QPushButton("⏪ Remover Último")
        self.btn_guardar = QPushButton("✅ Guardar Forma")
        self.btn_cancelar = QPushButton("❌ Cancelar")
        self.btn_next_contour = QPushButton("Próximo Contorno")

        btn_layout.addWidget(self.btn_undo)
        btn_layout.addWidget(self.btn_guardar)
        btn_layout.addWidget(self.btn_cancelar)
        btn_layout.addWidget(self.btn_next_contour)

        # Ligações
        self.btn_guardar.clicked.connect(self.guardar_forma)
        self.btn_cancelar.clicked.connect(self.close)
        self.btn_next_contour.clicked.connect(self.mostrar_proximo_contorno)

        self.pontos = []
        self.contours_sorted = []
        self.current_contour_index = 0

    # =========================================================
    # FUNÇÕES AUXILIARES
    # =========================================================
    def _update_scaled_image(self):
        """Redimensiona a imagem PIL para o fator de escala atual."""
        self.scaled_w = int(self.img_width * self.scale_factor)
        self.scaled_h = int(self.img_height * self.scale_factor)
        self.img_resized = self.original_image.resize(
            (self.scaled_w, self.scaled_h),
            Image.Resampling.LANCZOS
        )

    def _get_scaled_pixmap(self):
        qim = self.pil2qimage(self.img_resized)
        return QPixmap.fromImage(qim)

    def pil2qimage(self, im):
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        return QImage(data, im2.width, im2.height, QImage.Format_RGBA8888)

    # =========================================================
    # EVENTOS
    # =========================================================
    def wheelEvent(self, event):
        """Zoom com a roda do rato."""
        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1

        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))
        self._update_scaled_image()
        self.image_label.setPixmap(self._get_scaled_pixmap())

    # =========================================================
    # PROCESSAMENTO DE CONTORNOS
    # =========================================================
    def process_crop(self, start_qpoint, end_qpoint):
        x1, y1 = start_qpoint.x(), start_qpoint.y()
        x2, y2 = end_qpoint.x(), end_qpoint.y()
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # converter para coords da imagem original
        x1_orig = int(x1 / self.scale_factor)
        y1_orig = int(y1 / self.scale_factor)
        x2_orig = int(x2 / self.scale_factor)
        y2_orig = int(y2 / self.scale_factor)

        img_crop = self.original_image.crop((x1_orig, y1_orig, x2_orig, y2_orig)).convert("L")
        img_np = np.array(img_crop)

        # === Downscale para reduzir ruído ===
        scale_down = 0.5
        img_small = cv2.resize(img_np, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)

        # === Pré-processamento ===
        img_inv = 255 - img_small
        img_blur = cv2.GaussianBlur(img_inv, (5, 5), 0)

        # Threshold Otsu
        _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morfologia
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Contornos
        contours, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Nenhum contorno encontrado")
            return

        # Reescala
        contours = [np.array(cnt / scale_down, dtype=np.float32) for cnt in contours]

        # Filtrar + ordenar por score
        candidatos = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000 or area > 3_000_000:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            circularity = 4 * np.pi * (area / (peri * peri))
            x,y,w,h = cv2.boundingRect(cnt)
            bbox_area = w*h

            # Score = circularidade * área + bounding box
            score = circularity * area + bbox_area * 0.5

            if 0.5 < circularity < 1.5:
                candidatos.append((score, cnt))


        if not candidatos:
            print("Nenhum contorno válido após filtragem")
            return

        # Ordena candidatos: melhor primeiro
        candidatos.sort(key=lambda c: c[0], reverse=True)
        self.contours_sorted = [c[1] for c in candidatos]

        self.current_contour_index = 0
        self.last_x_offset = x1_orig
        self.last_y_offset = y1_orig
        self.selection_rect = (x1_orig, y1_orig, x2_orig, y2_orig)

        self._desenhar_contorno_atual(self.last_x_offset, self.last_y_offset)


    def _desenhar_contorno_atual(self, x_offset, y_offset):
        cnt = self.contours_sorted[self.current_contour_index]
        area = cv2.contourArea(cnt)
        print(f"Mostrando contorno {self.current_contour_index + 1} com área {area:.2f}")

        contorno_scaled = (cnt + np.array([x_offset, y_offset])) * self.scale_factor
        contorno_scaled = contorno_scaled.astype(int)
        pontos_contorno = contorno_scaled.reshape(-1, 2).tolist()

        pixmap = self._get_scaled_pixmap()
        painter = QPainter(pixmap)

        # === Desenha quadrado em vermelho ===
        if hasattr(self, "selection_rect"):
            x1, y1, x2, y2 = self.selection_rect
            rect_scaled = (
                int(x1 * self.scale_factor),
                int(y1 * self.scale_factor),
                int((x2 - x1) * self.scale_factor),
                int((y2 - y1) * self.scale_factor),
            )
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            painter.drawRect(*rect_scaled)

        # === Desenha contorno em azul ===
        painter.setPen(QPen(Qt.blue, 2))
        polygon = QPolygon([QPoint(x, y) for x, y in pontos_contorno])
        painter.drawPolygon(polygon)

        painter.end()
        self.image_label.setPixmap(pixmap)

        # Guardar pontos normalizados
        self.pontos = [(int(x / self.scale_factor), int(y / self.scale_factor)) for x, y in pontos_contorno]

    def mostrar_proximo_contorno(self):
        if not self.contours_sorted:
            print("Nenhum contorno carregado")
            return

        self.current_contour_index += 1
        if self.current_contour_index >= len(self.contours_sorted):
            self.current_contour_index = 0

        self._desenhar_contorno_atual(self.last_x_offset, self.last_y_offset)


    def guardar_forma(self):
        if len(self.pontos) < 3:
            print("⚠️ Defina pelo menos 3 pontos.")
            return

        cx = sum(x for x, y in self.pontos) / len(self.pontos)
        cy = sum(y for x, y in self.pontos) / len(self.pontos)
        forma_normalizada = [(x - cx, y - cy) for x, y in self.pontos]

        os.makedirs("data/mask", exist_ok=True)
        caminho = "data/mask/forma_base.json"
        with open(caminho, "w") as f:
            json.dump(forma_normalizada, f)

        print(f"[INFO] Forma base salva em {caminho}")
        self.close()
