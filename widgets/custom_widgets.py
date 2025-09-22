import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QWidget, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QApplication, QSizePolicy
from PySide6.QtGui import QFont, QPixmap, QImage, QPainter, QColor, QBrush
from PySide6.QtCore import Signal, Qt, QRect, QPropertyAnimation, QEasingCurve
import numpy as np
from config.config import TEMPLATE_IMAGE_PATH, INSPECTION_PREVIEW_WIDTH, INSPECTION_PREVIEW_HEIGHT

class TitleLabelMain(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;   /* cor de fundo */
                color: white;                 /* cor do texto */
                border: 2px solid #555555;   /* borda */
                border-radius: 5px;           /* cantos arredondados */
                padding: 5px;
                font-weight: bold;
                font-size: 16px;
                min-height: 40px;             /* altura mínima */
                max-height: 40px;             /* altura máxima */
            }
        """)

class ButtonMain(QPushButton):
    def __init__(
        self,
        text="",
        parent=None,
        font_size=14,
        bold=True,
        enable=True,
        bg_color="#448aff",
        hover_color="#5c9dff",
        pressed_color="#2d6fce",
        border_color="#4f5b62",
        disabled_bg="#3a3a3a",
        disabled_text="#aaaaaa",
        disabled_border="#2c2c2c"
    ):
        super().__init__(text, parent)

        font = QFont("Noto Color Emoji")
        font.setPointSize(font_size)
        font.setBold(bold)
        self.setFont(font)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: 2px solid {border_color};
                border-radius: 15px;
                padding: 8px 16px;
            }}
            QPushButton:hover:enabled {{
                background-color: {hover_color};
            }}
            QPushButton:pressed:enabled {{
                background-color: {pressed_color};
            }}
            QPushButton:disabled {{
                background-color: {disabled_bg};
                color: {disabled_text};
                border: 2px solid {disabled_border};
            }}
        """)

        self.setEnabled(enable)

class LabelNumeric(QWidget):
    valueChanged = Signal(float)  # sinal próprio para emitir mudanças

    def __init__(self, text="", value=0, step=1, minimum=-9999, maximum=9999, is_float=False, font_size=12, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # Label
        self.setObjectName("switchWidget")
        self.label = QLabel(text)
        self.label.setObjectName("switchLabel")
        font = QFont("", font_size)
        self.label.setFont(font)
        self.label.setStyleSheet(f"font-size: {font_size}pt;") 
        layout.addWidget(self.label)

        # SpinBox ou DoubleSpinBox
        if is_float:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(2)
        else:
            self.spinbox = QSpinBox()

        # Aplica fonte e aumenta tamanho do texto via stylesheet
        self.spinbox.setFont(QFont("", font_size))
        self.spinbox.setStyleSheet(f"QSpinBox, QDoubleSpinBox {{ font-size: {font_size}pt; }}")
        self.spinbox.setRange(minimum, maximum)
        self.spinbox.setValue(value)
        self.spinbox.setSingleStep(step)
        layout.addWidget(self.spinbox)

        # Guarda configurações
        self._step = step
        self._is_float = is_float

        # Conecta valueChanged
        self.spinbox.valueChanged.connect(lambda val: self.valueChanged.emit(val))

        # Ativar alteração com roda do rato
        self.spinbox.wheelEvent = self._wheelEvent

    def _wheelEvent(self, event):
        delta = event.angleDelta().y()
        step = self._step if delta > 0 else -self._step
        if self._is_float:
            self.spinbox.setValue(self.spinbox.value() + step)
        else:
            self.spinbox.setValue(int(self.spinbox.value() + step))

    def value(self):
        return self.spinbox.value()

    def setValue(self, val):
        self.spinbox.setValue(val)


    def _wheelEvent(self, event):
        """Incrementa ou decrementa com a roda do rato"""
        delta = event.angleDelta().y()
        step = self._step if delta > 0 else -self._step
        if self._is_float:
            self.spinbox.setValue(self.spinbox.value() + step)
        else:
            self.spinbox.setValue(int(self.spinbox.value() + step))

    def _wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.spinbox.setValue(self.spinbox.value() + self._step)
        else:
            self.spinbox.setValue(self.spinbox.value() - self._step)

    def value(self):
        """Retorna o valor atual"""
        return float(self.spinbox.value()) if self._is_float else int(self.spinbox.value())

    def setValue(self, val):
        """Define o valor"""
        self.spinbox.setValue(val)

    def valueChanged_connect(self, callback):
        """Liga uma função ao signal valueChanged do spinbox"""
        self.spinbox.valueChanged.connect(callback)

class ImageLabel(QLabel):
    def __init__(self, parent=None, width=INSPECTION_PREVIEW_WIDTH, height=INSPECTION_PREVIEW_HEIGHT, border_color="#D3D3D3", border_width=4, border_radius=5):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.border_color = border_color
        self.border_width = border_width
        self.border_radius = border_radius

        # Define tamanho inicial
        self.setFixedSize(width, height)

        self.setStyleSheet(f"""
            border: {self.border_width}px solid {self.border_color};
            border-radius: {self.border_radius}px;
        """)

    def update_style(self):
        self.setStyleSheet(f"""
            border: {self.border_width}px solid {self.border_color};
            border-radius: {self.border_radius}px;
        """)

    def set_border_color(self, color):
        self.border_color = color
        self.update_style()

    def set_image(self, img):
        """Recebe numpy array RGB ou QPixmap e ajusta para o tamanho da label (esticada)."""
        if isinstance(img, np.ndarray):
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
        elif isinstance(img, QPixmap):
            pixmap = img
        else:
            raise ValueError("img deve ser numpy array RGB ou QPixmap")

        pixmap = pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)

class SwitchButton(QWidget):
    stateChanged = Signal(bool)

    def __init__(self):
        super().__init__()
        self._checked = False
        self.setFixedSize(60, 28)

    def mousePressEvent(self, event):
        self._checked = not self._checked
        self.stateChanged.emit(self._checked)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Fundo
        painter.setBrush(QBrush(QColor("#00aa00") if self._checked else QColor("#cccccc")))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 14, 14)
        # Círculo
        circle_x = self.width() - 28 if self._checked else 0
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(circle_x, 0, 28, 28)

    def setChecked(self, value: bool):
        self._checked = value
        self.update()
        self.stateChanged.emit(self._checked)

    def isChecked(self):
        return self._checked

class Switch(QWidget):
    stateChanged = Signal(bool)

    def __init__(self, text="", font_size=14):
        super().__init__()

        self._checked = False

        # Layout horizontal: label e switch juntos
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)  # pequeno espaço entre label e switch

        # Label
        self.label = QLabel(text)
        # Respeita stylesheets globais; só aplica tamanho se explicitamente pedido
        if font_size is not None:
            # Usa stylesheet para permitir override fácil por QSS
            self.label.setStyleSheet(f"font-size: {int(font_size)}pt;")
        layout.addWidget(self.label, alignment=Qt.AlignVCenter)

        # Switch
        self._switch_width = 60
        self._switch_height = 28
        self.setFixedHeight(self._switch_height)  # só altura fixa
        layout.addStretch(0)  # evita expansão para direita

        # O switch vai ser desenhado dentro do widget
        # O paintEvent vai desenhar apenas a "área do switch" ao lado do texto

    def setFontSize(self, pt):
        """Opcional: define tamanho de fonte do label via stylesheet."""
        self.label.setStyleSheet(f"font-size: {int(pt)}pt;")

    def mousePressEvent(self, event):
        # Alterna estado se clicar na área do switch
        switch_x = self.label.width() + 5
        if switch_x <= event.position().x() <= switch_x + self._switch_width:
            self._checked = not self._checked
            self.stateChanged.emit(self._checked)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calcula posição do switch (logo após o label)
        switch_x = self.label.width() + 5
        switch_y = 0

        # Fundo
        painter.setBrush(QBrush(QColor("#00aa00") if self._checked else QColor("#cccccc")))
        painter.drawRoundedRect(switch_x, switch_y, self._switch_width, self._switch_height, 14, 14)

        # Círculo
        circle_x = switch_x + self._switch_width - 28 if self._checked else switch_x
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(circle_x, switch_y, 28, 28)

    def setChecked(self, value: bool):
        self._checked = value
        self.update()
        self.stateChanged.emit(self._checked)

    def isChecked(self):
        return self._checked

class LabeledValue(QWidget):
    def __init__(self, label_text="", value=0, value_width=100, font_size=14,
                 border_color="#D3D3D3", border_width=2, parent=None):
        super().__init__(parent)

        self.border_color = border_color
        self.border_width = border_width
        self.value_width = value_width
        self.font_size = font_size

        # Layout horizontal
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Label (bold)
        self.label = QLabel(label_text)
        self.label.setFixedWidth(200)
        self.label.setFixedHeight(20)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setStyleSheet(f"""
            font-size: {self.font_size}pt;
            font-weight: bold;
        """)
        layout.addWidget(self.label)

        # Value (não bold) com borda
        self.value_label = QLabel(str(value))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.update_style()
        layout.addWidget(self.value_label)
        layout.addWidget(QWidget())
        layout.addWidget(QWidget())

    def update_style(self):
        # Aplica o estilo com font-size, borda, padding e border-radius
        self.value_label.setStyleSheet(f"""
            font-size: {self.font_size}pt;
            font-weight: normal;
            border: {self.border_width}px solid {self.border_color};
            border-radius: 4px;
            padding: 2px;
        """)

    def set_value(self, value):
        """Atualiza o valor, converte int/float para str."""
        self.value_label.setText(str(value))

    def set_border_color(self, color):
        self.border_color = color
        self.update_style()

    def set_border_width(self, width):
        self.border_width = width
        self.update_style()

class LabeledText(QWidget):
    def __init__(self, label_text="", value="", value_width=100, font_size=15,
                 border_color="#D3D3D3", border_width=2, parent=None):
        super().__init__(parent)

        self.border_color = border_color
        self.border_width = border_width
        self.value_width = value_width
        self.font_size = font_size

        # Layout horizontal
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Label (bold)
        self.label = QLabel(label_text)
        self.label.setFixedWidth(200)
        self.label.setFixedHeight(20)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setStyleSheet(f"""
            font-size: {self.font_size}pt;
            font-weight: bold;
        """)
        layout.addWidget(self.label)

        # Value (não bold) com borda
        self.value_label = QLabel(str(value))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.update_style()
        layout.addWidget(self.value_label)
        layout.addWidget(QWidget())
        layout.addWidget(QWidget())

    def update_value(self, new_value):
        self.value_label.setText(str(new_value))

    def update_style(self):
        # Aplica o estilo com font-size, borda, padding e border-radius
        self.value_label.setStyleSheet(f"""
            font-size: {self.font_size}pt;
            font-weight: normal;
            border: {self.border_width}px solid {self.border_color};
            border-radius: 4px;
            padding: 2px;
        """)

    def set_value(self, value):
        """Atualiza o valor, converte int/float para str."""
        self.value_label.setText(str(value))

    def set_border_color(self, color):
        self.border_color = color
        self.update_style()

    def set_border_width(self, width):
        self.border_width = width
        self.update_style()

class LabeledIndicator(QWidget):
    def __init__(self, label_text="", state=False, font_size=15, parent=None):
        super().__init__(parent)

        self.font_size = font_size
        self.state = state

        # Layout horizontal
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Label (bold)
        self.label = QLabel(label_text)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setFixedWidth(200)
        self.label.setFixedHeight(20)
        self.label.setStyleSheet(f"""
            font-size: {self.font_size}pt;
            font-weight: bold;
        """)
        layout.addWidget(self.label)

        # Indicador de estado (círculo)
        self.state_indicator = QLabel()
        self.state_indicator.setFixedSize(self.font_size + 10, self.font_size + 10)
        self.state_indicator.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.state_indicator)
        layout.addWidget(QWidget())
        layout.addWidget(QWidget())

        # Aplica estilo inicial
        self.update_indicator()

    def update_indicator(self):
        """Atualiza a cor do indicador conforme o estado."""
        color = "#2ECC71" if self.state else "#555555"  # verde ou cinza
        size = self.font_size + 6
        self.state_indicator.setStyleSheet(f"""
            background-color: {color};
            border-radius: {size // 2}px;
        """)

    def set_state(self, state: bool):
        """Define o estado (True/False) e atualiza a cor."""
        self.state = state
        self.update_indicator()

class TitleLabel(QLabel):
    def __init__(self, text="", font_size=60, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setAlignment(Qt.AlignCenter)

        # Fonte bold e tamanho
        font = QFont()
        font.setPointSize(font_size)
        font.setBold(True)
        self.setFont(font)

        # Estilo visual
        self.setStyleSheet("""
            background-color: #A9A9A9;   /* cinzento */
            color: #FFFFFF;              /* branco */
            border: 2px solid #FFFFFF;   /* borda branca */
            border-radius: 5px;          /* cantos arredondados */
            padding: 5px;
        """)

class Indicator(QLabel):
    """Circular LED-style indicator.
    - True  => Green
    - False => Red
    """
    def __init__(self, state=False, diameter=16, parent=None):
        super().__init__(parent)
        self._state = bool(state)
        self._diameter = int(diameter)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(self._diameter, self._diameter)
        self._update_style()

    def _update_style(self):
        color = "#2ECC71" if self._state else "#E74C3C"
        radius = self._diameter // 2
        self.setStyleSheet(f"""
            background-color: {color};
            border: 1px solid #444444;
            border-radius: {radius}px;
        """)

    def set_state(self, state: bool):
        self._state = bool(state)
        self._update_style()

    def is_on(self) -> bool:
        return self._state

    def toggle(self):
        self._state = not self._state
        self._update_style()
