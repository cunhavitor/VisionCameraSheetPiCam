import os
import json
import cv2
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QComboBox, QFrame, QApplication, QScrollArea
)
from PySide6.QtGui import QPixmap, QImage, QMouseEvent
from PySide6.QtCore import Qt, QSize
from ultralytics import YOLO
from config.config import TEMPLATE_IMAGE_PATH, INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT
from config.utils import center_window
from widgets.custom_widgets import ImageLabel, ButtonMain
from PySide6.QtWidgets import QMessageBox
from windows.create_form_can import CriarFormaWindow

class AdjustPositionsWindow(QDialog):
    def __init__(self, parent, template_path=None):
        super().__init__()
        self.setWindowTitle("Ajustar Máscara")
        self.setFixedSize(1300, 800)

        # Centralizar a janela
        screen = self.screen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        self.base_img_np = None
        self.annotated_img_np = None
        self.polygons_instances = []
        self.polygons = []

        self.preview_size = (INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT)
        self.original_image = cv2.imread(template_path)
        self.img_size = (self.original_image.shape[1], self.original_image.shape[0])
        self.line_entries = []

        self.model = YOLO("models/weights/best.pt")

        # Inicializa posições das linhas
        self.line_positions_x = []  # para linhas verticais
        self.line_positions_y = []  # para linhas horizontais

        self.num_latas_x_value = None
        self.num_latas_y_value = None

        self._build_ui()

    def _build_ui(self):
        # === Frames ===
        left_frame = QFrame(self)
        left_frame.setFixedWidth(350)
        left_frame.setFrameShape(QFrame.StyledPanel)

        # Layout do frame esquerdo
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)

        # === Preview Frame ===
        preview_frame = QFrame(self)
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_frame.setFixedSize(self.preview_size[0]+4, self.preview_size[1]+8)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        # Layout principal horizontal
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_frame)
        main_layout.addWidget(preview_frame)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        self.setLayout(main_layout)

        # === Botões e labels ===
        button_layout = QHBoxLayout()
        left_layout.addLayout(button_layout)
        self.detect_button = ButtonMain("Detectar Latas")
        self.detect_button.clicked.connect(self.run_detection)
        button_layout.addWidget(self.detect_button)

        self.create_form = ButtonMain("Criar Forma")
        self.create_form.clicked.connect(self.abrir_janela_criar_forma)
        button_layout.addWidget(self.create_form)

        self.create_form = ButtonMain(
            "Reset",
            bg_color="#ff4d4d",        # vermelho base
            hover_color="#ff6666",     # vermelho ao passar o mouse
            pressed_color="#cc0000",   # vermelho ao pressionar
            border_color="#990000"     # borda escura
        )
        self.create_form.clicked.connect(self.reset_lines)
        button_layout.addWidget(self.create_form)

        # Num latas X
        num_x_layout = QHBoxLayout()
        left_layout.addLayout(num_x_layout)
        num_x_label = QLabel("Número de latas em X:")
        num_x_layout.addWidget(num_x_label)
        self.num_latas_x_value = QLabel("0")
        num_x_layout.addWidget(self.num_latas_x_value)

        # Num latas Y
        num_y_layout = QHBoxLayout()
        left_layout.addLayout(num_y_layout)
        num_y_label = QLabel("Número de latas em Y:")
        num_y_layout.addWidget(num_y_label)
        self.num_latas_y_value = QLabel("0")
        num_y_layout.addWidget(self.num_latas_y_value)

        # Line entries container (scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.line_entries_widget = QWidget()
        self.line_entries_layout = QVBoxLayout(self.line_entries_widget)
        self.scroll_area.setWidget(self.line_entries_widget)
        left_layout.addWidget(self.scroll_area)

        # Botões abaixo das linhas
        self.enumerate_polygons_btn = ButtonMain("Numerar Latas")
        self.enumerate_polygons_btn.clicked.connect(self.number_polygons_on_lines)
        left_layout.addWidget(self.enumerate_polygons_btn)

        self.save_mask_btn = ButtonMain("Guardar Máscara")
        self.save_mask_btn.setEnabled(False)
        self.save_mask_btn.clicked.connect(self.on_salvar_mascara)
        left_layout.addWidget(self.save_mask_btn)

        # === Image preview ===
        image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        image = QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], image_rgb.strides[0], QImage.Format_RGB888)
        #pil_image = Image.fromarray(image_rgb)
        #qimage = self.pil2pixmap(pil_image)
        
        self.image_label = ImageLabel()
        self.image_label.set_image(QPixmap.fromImage(image))
        preview_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

    def reset_lines(self):
        print("[INFO] Resetando detecção e numeração...")

        # Reset valores de latas X/Y
        self.num_latas_x_value.setText("0")
        self.num_latas_y_value.setText("0")

        # Remove sliders e combos das linhas
        for i in reversed(range(self.line_entries_layout.count())):
            widget = self.line_entries_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.line_entries.clear()
        self.line_positions_x.clear()
        self.line_positions_y.clear()

        # Limpa polígonos e instâncias
        self.polygons.clear()
        self.polygons_instances.clear()
        self.polygons_numbered = []

        # Limpa imagem anotada
        self.annotated_img_np = None
        self.redraw_lines()

        # Desativa botão de salvar máscara
        self.save_mask_btn.setEnabled(False)

        print("[INFO] Reset completo.")


    def mask_image(self, image):
        """Aplica a máscara leaf_mask.png na imagem fornecida e retorna a imagem mascarada."""
        mask_path = "data/mask/leaf_mask.png"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[ERRO] Máscara não encontrada em {mask_path}")
            return image  # retorna original se máscara não existir

        # Aplica máscara
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        return masked_img

    def pil2pixmap(self, im):
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qim = QImage(data, im2.width, im2.height, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qim)
        return pix

    # =================== Eventos e funções ====================
    def update_line_entries(self, num_fila_y):
        # Remove widgets antigos
        for i in reversed(range(self.line_entries_layout.count())):
            widget = self.line_entries_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.line_entries.clear()

        try:
            num_linhas = int(num_fila_y)
        except ValueError:
            num_linhas = 0

        self.line_positions_y = []
        for i in range(num_linhas):
            frame = QFrame()
            layout = QHBoxLayout(frame)
            label = QLabel(f"Linha {i+1}")
            layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(self.img_size[1])
            slider.setValue(int(self.img_size[1]*(num_linhas-i)/(num_linhas+1)))
            slider.valueChanged.connect(lambda val, idx=i: self.update_line_position(idx, val))
            layout.addWidget(slider)

            combo = QComboBox()
            combo.addItems(["left->right", "right->left"])
            layout.addWidget(combo)

            self.line_entries_layout.addWidget(frame)
            self.line_entries.append((frame, slider, combo))
            self.line_positions_y.append(slider.value())

        self.redraw_lines()

    def update_line_position(self, index, new_y):
        if 0 <= index < len(self.line_positions_y):
            self.line_positions_y[index] = new_y
            self.redraw_lines()

    def redraw_lines(self):
        if self.annotated_img_np is not None:
            img = self.annotated_img_np.copy()
        else:
            img = self.original_image.copy()

        for y in self.line_positions_y:
            cv2.line(img, (0, int(y)), (self.img_size[0], int(y)), (0, 0, 255), 4)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pix = self.pil2pixmap(pil_img)

        # === Mantém tamanho original se for menor, redimensiona só se for maior ===
        if pix.width() > self.preview_size[0] or pix.height() > self.preview_size[1]:
            pix = pix.scaled(
                self.preview_size[0], self.preview_size[1],
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        self.image_label.setPixmap(pix)


    def run_detection(self):
        print("[INFO] Executando detecção...")
        image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Aplica máscara antes de enviar para YOLO
        masked_rgb = self.mask_image(image_rgb)

        results = self.model.predict(masked_rgb, verbose=False)
        if not results:
            print("Nenhum resultado do modelo.")
            return

        result = results[0]
        boxes = result.boxes

        if boxes is None or boxes.xywh is None or len(boxes.xywh) == 0:
            print("Nenhuma lata detectada.")
            return

        annotated = self.original_image.copy()
        base_shape = self.load_forma_base()
        for bbox in boxes.xywh.cpu().numpy():
            cx, cy, w, h = bbox
            x = int(cx - w/2)
            y = int(cy - h/2)
            self.draw_polygon_on_box(annotated, (x, y, int(w), int(h)), base_shape)

        self.annotated_img_np = annotated.copy()
        self.redraw_lines()
        # Atualiza número de latas em X/Y
        filas = self.detectar_filas_poligonos(self.polygons)
        num_filas = len(filas)
        poligonos_por_fila = [len(f) for f in filas]
        self.num_latas_x_value.setText(str(poligonos_por_fila[0]))
        self.num_latas_y_value.setText(str(num_filas))
        self.update_line_entries(num_filas)


    # ======= Funções auxiliares =======
    def load_forma_base(self, path="data/mask/forma_base.json"):
        with open(path, "r") as f:
            return json.load(f)

    def draw_polygon_on_box(self, image, box, base_shape):
        base_shape = np.array(base_shape, dtype=np.float32)
        x, y, w, h = box
        min_x, min_y = base_shape.min(axis=0)
        shape_norm = base_shape - [min_x, min_y]
        orig_width = np.ptp(shape_norm[:,0])
        orig_height = np.ptp(shape_norm[:,1])
        scale = min(w/orig_width, h/orig_height)
        shape_scaled = shape_norm*scale
        new_width = np.ptp(shape_scaled[:,0])
        new_height = np.ptp(shape_scaled[:,1])
        offset_x = x + (w-new_width)/2
        offset_y = y + (h-new_height)/2
        shape_translated = shape_scaled + [offset_x, offset_y]
        polygon = shape_translated.astype(np.int32)
        self.polygons.append(polygon.tolist())
        cx = int(x + w/2)
        cy = int(y + h/2)
        self.polygons_instances.append({"points":[cx,cy], "scale": scale})
        cv2.polylines(image, [polygon], isClosed=True, color=(0,255,0), thickness=4)

    def detectar_filas_poligonos(self, polygons, tolerancia_y=50):
        centros = []
        for poly in polygons:
            poly_np = np.array(poly)
            cx = int(np.mean(poly_np[:,0]))
            cy = int(np.mean(poly_np[:,1]))
            centros.append((cx,cy))
        filas = []
        for cx,cy in sorted(centros, key=lambda c:c[1]):
            encontrado=False
            for fila in filas:
                _, y_medio = np.mean(fila, axis=0)
                if abs(cy-y_medio)<tolerancia_y:
                    fila.append((cx,cy))
                    encontrado=True
                    break
            if not encontrado:
                filas.append([(cx,cy)])
        for fila in filas:
            fila.sort(key=lambda c:c[0])
        return filas

    def number_polygons_on_lines(self):
        print("[INFO] Numerando polígonos...")

        if not self.polygons or not self.line_positions_y:
            print("[AVISO] Nenhum polígono ou linhas definidas.")
            return

        # Cria lista de centros dos polígonos
        centros = []
        for poly in self.polygons:
            poly_np = np.array(poly)
            cx = int(np.mean(poly_np[:,0]))
            cy = int(np.mean(poly_np[:,1]))
            centros.append({"cx": cx, "cy": cy, "polygon": poly})

        # Agrupa polígonos por linha
        linhas_poligonos = [[] for _ in self.line_positions_y]
        for centro in centros:
            cy = centro["cy"]
            # Encontra a linha mais próxima
            idx_linha = min(range(len(self.line_positions_y)), key=lambda i: abs(cy - self.line_positions_y[i]))
            linhas_poligonos[idx_linha].append(centro)

        # Ordena cada linha de acordo com o combo
        for i, linha in enumerate(linhas_poligonos):
            if not linha:
                continue
            # Obtem direção do combo
            _, _, combo = self.line_entries[i]
            if combo.currentText() == "left->right":
                linha.sort(key=lambda c: c["cx"])
            else:
                linha.sort(key=lambda c: -c["cx"])

        # Numeração final
        numero = 1
        for linha in linhas_poligonos:
            for p in linha:
                p["numero"] = numero
                numero += 1

        # Guarda informação para referência
        self.polygons_numbered = linhas_poligonos

        # Atualiza visualização: desenha números
        annotated = self.annotated_img_np.copy() if self.annotated_img_np is not None else self.original_image.copy()
        for linha in linhas_poligonos:
            for p in linha:
                cx = p["cx"]
                cy = p["cy"]
                numero = p["numero"]

                # Determina tamanho do texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 5
                thickness = 10
                (text_w, text_h), baseline = cv2.getTextSize(str(numero), font, font_scale, thickness)

                # Centraliza horizontalmente sobre o centro do polígono
                text_x = cx - text_w // 2
                text_y = cy + text_h // 2  # Ajusta verticalmente para o centro

                cv2.putText(annotated, str(numero), (text_x, text_y), font, font_scale, (255,0,0), thickness)
            
        self.annotated_img_np = annotated
        self.redraw_lines()

        # Ativa botão de salvar máscara
        self.save_mask_btn.setEnabled(True)
        print(f"[INFO] Numeradas {numero-1} latas.")


    def on_salvar_mascara(self):
        try:
            mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)
            for polygon in self.polygons:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            os.makedirs("data/mask", exist_ok=True)
            path = "data/mask/leaf_mask.png"
            success = cv2.imwrite(path, mask)

            if success:
                QMessageBox.information(self, "Sucesso", f"Máscara salva em:\n{path}")
                print(f"[INFO] Máscara salva em {path}.")
            else:
                QMessageBox.critical(self, "Erro", "Falha ao guardar a máscara!")
                print("[ERRO] Falha ao guardar a máscara.")

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Ocorreu um erro:\n{e}")
            print(f"[ERRO] {e}")

    def abrir_janela_criar_forma(self):
        self.forma_window = CriarFormaWindow(self)
        self.forma_window.setWindowModality(Qt.ApplicationModal)   # bloqueia toda a app
        # ou: self.forma_window.setWindowModality(Qt.WindowModal) # bloqueia só a parent
        self.forma_window.show()

