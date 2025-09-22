
# # Entrar na pasta do projeto
# cd ~/projects/VisionCameraSheetPiCam

# Criar o ambiente virtual chamado 'venv'
# python3 -m venv venv

# Ativar o ambiente virtual
# source venv/bin/activate

# cd ~/projects/VisionCameraSheet
# source venv/bin/activate
# python main.py

# token
# ***REMOVED***

# -*- coding: utf-8 -*-

import os
import sys
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
)
from PySide6.QtCore import Qt
from picamera2 import Picamera2
from qt_material import apply_stylesheet
from widgets.custom_widgets import ButtonMain, TitleLabelMain
from PySide6.QtWidgets import QSpacerItem, QSizePolicy
from windows.login_window import LoginWindow
from windows.create_users import NewUserWindow
from windows.manage_users_window import ManageUserWindow
from windows.capture_sheet import CaptureSheetWindow
from windows.params_cam_adjust_window import CameraAdjustParamsWindow
from windows.camera_adjust_positions import CameraAdjustPosition
from windows.inspection_window import InspectionWindow
from windows.create_leaf_mask import LeafMaskCreator
from windows.alignment_adjust import AlignmentWindow
from windows.adjust_positions import AdjustPositionsWindow

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detection Lito Errors")
        self.setFixedSize(1200, 700)

        try:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(4)  # Pi 5 tem 4 cores
        except Exception as e:
            print("OpenCV threading:", e)

        # Centralizar a janela
        screen = self.screen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        main_layout.setSpacing(50)  # distância horizontal entre colunas
        main_layout.setContentsMargins(50, 0, 50, 0)
        main_layout.setAlignment(Qt.AlignTop)

        # Parâmetros de espaçamento
        top_margin = 100
        title_to_button = 60
        button_spacing = 10

        # var picam
        self.picam2 = Picamera2()

        # Função utilitária para criar colunas
        def create_column(title_text, buttons):
            layout = QVBoxLayout()
            layout.setSpacing(10)  # espaçamento entre botões

            # Spacer fixo no topo da coluna
            layout.addSpacerItem(QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Fixed))

            # TitleLabel
            title = TitleLabelMain(title_text)
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)

            # Spacer entre TitleLabel e primeiro botão
            layout.addSpacerItem(QSpacerItem(20, 60, QSizePolicy.Minimum, QSizePolicy.Fixed))

            # Botões
            for btn in buttons:
                layout.addWidget(btn)
                layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))  # espaçamento entre botões

            # Spacer final para preencher a coluna
            layout.addStretch()
            return layout

        # Coluna esquerda
        self.login_button = ButtonMain("🔑 Login")
        self.new_user_button = ButtonMain("➕ Novo User")
        self.manage_users_button = ButtonMain("👥 Gerir Users")

        self.login_button.clicked.connect(self.open_login_window)
        self.new_user_button.clicked.connect(self.open_new_user_window)
        self.manage_users_button.clicked.connect(self.open_manage_users_window)

        main_layout.addLayout(create_column("USERS", [
            self.login_button,
            self.new_user_button,
            self.manage_users_button
        ]))

        # Coluna do meio
        self.capture_sheet_button = ButtonMain("📄 Adjust Cam and Template")
        self.adjust_positions_button = ButtonMain("✏️ Adjust Positions")
        self.mask_window_button = ButtonMain("🎭 Máscara")
        self.alignment_adjust_window_button = ButtonMain("🔧 Alignment Adjust")
        self.check_camera_position_button = ButtonMain("📷 Check Camera Positions")
        #self.check_camera_params_button = ButtonMain("⚙️ Check Camera Params")

        self.capture_sheet_button.clicked.connect(self.open_capture_sheet)
        self.adjust_positions_button.clicked.connect(self.open_adjust_positions)
        self.mask_window_button.clicked.connect(self.open_mask_window)
        self.alignment_adjust_window_button.clicked.connect(self.open_alignment_adjust_window)
        self.check_camera_position_button.clicked.connect(self.open_check_camera_position_window)
        #self.check_camera_params_button.clicked.connect(self.open_check_camera_params_window)

        main_layout.addLayout(create_column("SETTINGS", [
            self.capture_sheet_button,
            self.adjust_positions_button,
            self.mask_window_button,
            self.alignment_adjust_window_button,
            self.check_camera_position_button,
            #self.check_camera_params_button
        ]))

        # Coluna direita
        self.gallery_button = ButtonMain("🖼️ Ver Galeria")
        self.inspect_button = ButtonMain("🔍 Inspeção")

        self.gallery_button.clicked.connect(self.open_gallery)
        self.inspect_button.clicked.connect(self.open_inspection)

        main_layout.addLayout(create_column("INSPECTION", [
            self.gallery_button,
            self.inspect_button
        ]))
       
        # Tipo de usuário atual
        self.user_type = ""
        self.user = ""
        self.update_user_access() 

    # ----------------- Funções de abertura de janelas -----------------
    def open_login_window(self):
        def on_login(username, user_type):
            self.user = username
            self.user_type = user_type
            print(f"Usuário logado: {username}, tipo: {user_type}")
            self.update_user_access()  # habilita/desabilita botões conforme tipo

        login_dialog = LoginWindow(self, on_login_callback=on_login)
        login_dialog.exec()  # abre modal

    def open_new_user_window(self):
        new_user_dialog = NewUserWindow(self)
        new_user_dialog.exec()  # abre modal

    def open_manage_users_window(self):
        manage_users_dialog = ManageUserWindow(self)
        manage_users_dialog.exec()  # abre modal

    def open_capture_sheet(self):
        params_cam_window = CameraAdjustParamsWindow(self, self.picam2)
        params_cam_window.setWindowModality(Qt.NonModal) 
        params_cam_window.show()

    def open_adjust_positions(self):
        template_path = "data/raw/fba_template.jpg"
        adjust_positions_window = AdjustPositionsWindow(self, template_path)
        adjust_positions_window.exec()  # abre modal


    def open_mask_window(self):
        image_path = "data/raw/fba_template.jpg"
        create_mask_window = LeafMaskCreator(self, image_path)
        create_mask_window.exec()  # abre modal

    def open_alignment_adjust_window(self):
        create_mask_window = AlignmentWindow(self, self.picam2)
        create_mask_window.exec()  # abre modal

    def open_check_camera_position_window(self):
        aligment_window = AlignmentWindow(self, self.picam2)
        aligment_window.exec()  # abre moda


    def open_gallery(self):
        print("Abrir galeria")
        # TODO: abrir janela real de galeria

    def open_inspection(self):
        mask_path = "data/mask/leaf_mask.png"
        template_path = "data/raw/fba_template.jpg"
        inspection_window = InspectionWindow(
            parent=self,           
            picam2=self.picam2,
            template_path=template_path,
            mask_path=mask_path,
            user_type="User",
            user="Vitor"
        )
        inspection_window.exec()

    # ----------------- Função para atualizar permissões de usuário -----------------
    def update_user_access(self):
        buttons = {
            "new_user": self.new_user_button,
            "gallery": self.gallery_button,
            "adjust_positions": self.adjust_positions_button,
            "mask": self.mask_window_button,
            "alignment": self.alignment_adjust_window_button,
            "check_camera": self.check_camera_position_button,
            "manage_users": self.manage_users_button
        }

        if self.user_type == "User":
            for btn in buttons.values():
                btn.setEnabled(False)

        elif self.user_type == "Admin":
            for key, btn in buttons.items():
                btn.setEnabled(True)

        elif self.user_type == "SuperAdmin":
            for btn in buttons.values():
                btn.setEnabled(True)
        else:
            for btn in buttons.values():
                btn.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_blue.xml")
    window = App()
    window.show()
    sys.exit(app.exec())
