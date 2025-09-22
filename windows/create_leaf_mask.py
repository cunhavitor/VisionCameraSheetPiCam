import os
import cv2
import json
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QFrame
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from config.config import PREVIEW_WIDTH, PREVIEW_HEIGHT
from widgets.custom_widgets import ImageLabel, ButtonMain


class LeafMaskCreator(QDialog):
    def __init__(self, parent=None, image_path=None,
                 output_path="data/mask/leaf_mask.png",
                 coords_path="data/mask/mask_coords.txt"):
        super().__init__(parent)
        self.setWindowTitle("Selecionar Máscara da Folha")
        self.resize(1300, 800)

        # Centralizar janela
        screen = self.screen().availableGeometry()
        self.move((screen.width() - self.width()) // 2,
                  (screen.height() - self.height()) // 2)

        # Variáveis
        self.image_path = image_path
        self.output_path = output_path
        self.coords_path = coords_path
        self.points = []
        self.is_done = False
        self.scale = 1.0
        self.image = None
        self.clone = None

        # Edição de pontos
        self.dragging_point_index = None

        # UI
        self._setup_ui()
        self._load_and_prepare_image()

        # Se já existir coordenadas, carrega (permite edição)
        self._load_existing_coords()

    # ---------- UI ----------
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Lado esquerdo: botões
        left_layout = QVBoxLayout()
        self.btn_confirm = ButtonMain("Confirmar (Enter)")
        self.btn_confirm.clicked.connect(self._confirm)
        self.btn_confirm.setEnabled(False)

        self.btn_undo = ButtonMain("Desfazer (Z)")
        self.btn_undo.clicked.connect(self._undo)

        self.btn_reset = ButtonMain("Recomeçar")
        self.btn_reset.clicked.connect(self._reset_points)

        self.btn_cancel = ButtonMain("Cancelar (Esc)")
        self.btn_cancel.clicked.connect(self._cancel)

        left_layout.addWidget(self.btn_confirm)
        left_layout.addWidget(self.btn_undo)
        left_layout.addWidget(self.btn_reset)
        left_layout.addWidget(self.btn_cancel)
        left_layout.addStretch()

        # Lado direito: imagem + preview máscara
        right_layout = QVBoxLayout()
        self.image_label = ImageLabel()
        self.image_label.setFixedSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.image_label.mousePressEvent = self._on_mouse_press
        self.image_label.mouseMoveEvent = self._on_mouse_move
        self.image_label.mouseReleaseEvent = self._on_mouse_release
        self.image_label.mouseDoubleClickEvent = lambda e: self._confirm()
        self.image_label.leaveEvent = self._on_mouse_leave  # reset cursor ao sair
        self.image_label.setMouseTracking(True)             # <— IMPORTANTE!
        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

        # Também ativo no diálogo, por segurança (não estraga)
        self.setMouseTracking(True)

        self.mask_preview = QLabel("Preview da Máscara")
        self.mask_preview.setFixedSize(300, 200)
        self.mask_preview.setFrameShape(QFrame.Box)
        self.mask_preview.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        right_layout.addWidget(self.mask_preview, alignment=Qt.AlignCenter)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 4)

    # ---------- Load ----------
    def _load_and_prepare_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            QMessageBox.warning(self, "Erro", f"Erro ao carregar {self.image_path}")
            return
        display_img = self._resize_image(self.image)
        self.clone = display_img.copy()
        self._draw_polygon()

    def _resize_image(self, img):
        h, w = img.shape[:2]
        scale_w = PREVIEW_WIDTH / w
        scale_h = PREVIEW_HEIGHT / h
        self.scale = min(1.0, scale_w, scale_h)
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        return cv2.resize(img, (new_w, new_h))

    # ---------- Desenho ----------
    def _draw_polygon(self):
        img = self.clone.copy()

        # Desenhar linhas e pontos
        for i, (x, y) in enumerate(self.points):
            px, py = int(x * self.scale), int(y * self.scale)
            cv2.circle(img, (px, py), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(img, str(i+1), (px+6, py-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                        lineType=cv2.LINE_AA)
            if i > 0:
                px_prev, py_prev = int(self.points[i-1][0] * self.scale), int(self.points[i-1][1] * self.scale)
                cv2.line(img, (px_prev, py_prev), (px, py), (255, 0, 0), 1, lineType=cv2.LINE_AA)

        # Fechar polígono
        if self.is_done and len(self.points) >= 3:
            pts_scaled = np.array([(int(x*self.scale), int(y*self.scale)) for x,y in self.points])
            cv2.polylines(img, [pts_scaled], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts_scaled], (0, 255, 0))
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

        # Atualizar imagem
        qimg = self._cv2_to_qimage(img)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

        # Atualizar preview da máscara
        if self.is_done and len(self.points) >= 3:
            mask = self._create_mask_array()
            mask_disp = cv2.resize(mask, (300, 200))
            qmask = QImage(mask_disp.data, mask_disp.shape[1], mask_disp.shape[0],
                           mask_disp.strides[0], QImage.Format_Grayscale8)
            self.mask_preview.setPixmap(QPixmap.fromImage(qmask))
        else:
            self.mask_preview.clear()
            self.mask_preview.setText("Preview da Máscara")

    def _cv2_to_qimage(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

    # ---------- Eventos ----------
    def _hover_tol(self):
        """Tolerância em coordenadas da imagem original para ~8 px no ecrã."""
        return max(3, int(8 / self.scale))  # mínimo 3 px

    def _is_over_point(self, x, y):
        """Verifica se (x,y) está em cima de um ponto existente."""
        tol = self._hover_tol()
        for i, (px, py) in enumerate(self.points):
            if abs(x - px) <= tol and abs(y - py) <= tol:
                return i
        return None

    def _on_mouse_press(self, event):
        if self.is_done:
            return

        x, y = self._map_click_to_original(event.pos().x(), event.pos().y())
        idx = self._is_over_point(x, y)

        if idx is not None:  # clicou num ponto → começa arrastar
            self.dragging_point_index = idx
            self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # Se não clicou em ponto → adicionar novo
        self.points.append((x, y))
        if len(self.points) >= 3:
            self.btn_confirm.setEnabled(True)
        self._draw_polygon()

    def _on_mouse_move(self, event):
        x, y = self._map_click_to_original(event.pos().x(), event.pos().y())

        if self.dragging_point_index is not None:  # arrastando ponto
            self.points[self.dragging_point_index] = (x, y)
            self._draw_polygon()
        else:  # apenas hover
            if self._is_over_point(x, y) is not None:
                self.image_label.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_mouse_release(self, event):
        if self.dragging_point_index is not None:  # terminou arraste
            self.dragging_point_index = None

        # verifica posição final do rato para cursor correto
        x, y = self._map_click_to_original(event.pos().x(), event.pos().y())
        if self._is_over_point(x, y) is not None:
            self.image_label.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

        self._draw_polygon()

    def _on_mouse_leave(self, event):
        """Reset do cursor quando o rato sai do QLabel da imagem."""
        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

    def _map_click_to_original(self, click_x, click_y):
        orig_h, orig_w = self.image.shape[:2]
        label_w, label_h = self.image_label.width(), self.image_label.height()
        scale = min(label_w / orig_w, label_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        offset_x, offset_y = (label_w - disp_w) / 2, (label_h - disp_h) / 2
        x = int((click_x - offset_x) / scale)
        y = int((click_y - offset_y) / scale)
        return max(0, min(x, orig_w-1)), max(0, min(y, orig_h-1))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Z, Qt.Key_Backspace):
            self._undo()
        elif event.key() == Qt.Key_Return:
            self._confirm()
        elif event.key() == Qt.Key_Escape:
            self._cancel()

    # ---------- Ações ----------
    def _confirm(self):
        if len(self.points) < 3:
            QMessageBox.warning(self, "Erro", "⚠️ Pelo menos 3 pontos são necessários.")
            return
        self.is_done = True
        self._draw_polygon()
        self._save_mask_and_coords()
        QMessageBox.information(self, "Máscara Salva", f"✅ Máscara salva em:\n{self.output_path}")
        self.accept()

    def _undo(self):
        if self.points:
            self.points.pop()
            if len(self.points) < 3:
                self.btn_confirm.setEnabled(False)
            self._draw_polygon()

    def _reset_points(self):
        """Apaga todos os pontos e volta ao estado inicial."""
        self.points = []
        self.is_done = False
        self.btn_confirm.setEnabled(False)
        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        self._draw_polygon()

    def _cancel(self):
        QMessageBox.information(self, "Cancelado", "✖️ Criação da máscara cancelada.")
        self.reject()

    # ---------- Ficheiros ----------
    def _create_mask_array(self):
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        if len(self.points) >= 3:
            pts = np.array([self.points], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
        return mask

    def _save_mask_and_coords(self):
        mask = self._create_mask_array()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        cv2.imwrite(self.output_path, mask)
        with open(self.coords_path, "w") as f:
            json.dump(self.points, f)

    def _load_existing_coords(self):
        if os.path.exists(self.coords_path):
            try:
                with open(self.coords_path, "r") as f:
                    self.points = json.load(f)
                if len(self.points) >= 3:
                    # Não marcamos como "done", para permitir editar/arrastar/adicionar
                    self.btn_confirm.setEnabled(True)
                    self._draw_polygon()
            except Exception as e:
                print(f"[WARN] Falha ao carregar coords: {e}")
