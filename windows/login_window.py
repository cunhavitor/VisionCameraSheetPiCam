import sys
import json
import os
from PySide6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QApplication
)
from PySide6.QtCore import Qt

USERS_FILE = "config/users.json"

from config.utils import center_window


class LoginWindow(QDialog):
    def __init__(self, parent=None, on_login_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        center_window(self, 400, 350)
        self.setFixedSize(400, 350)

        self.on_login_callback = on_login_callback

        # Variaveis
        self.username_var = ""
        self.password_var = ""

        # Layout principal
        layout = QVBoxLayout()

        title_label = QLabel("Autenticacao")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)

        # Usuario
        layout.addWidget(QLabel("Usuario:"))
        self.entry_username = QLineEdit(self)
        layout.addWidget(self.entry_username)

        # Senha
        layout.addWidget(QLabel("Senha:"))
        self.entry_password = QLineEdit()
        self.entry_password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.entry_password)

        # Label de erro
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        layout.addWidget(self.error_label)

        # Botao login
        self.btn_login = QPushButton("Entrar")
        self.btn_login.clicked.connect(self.tentar_login)
        layout.addWidget(self.btn_login)

        self.setLayout(layout)

        # Enter para submeter senha
        self.entry_password.returnPressed.connect(self.tentar_login)

        self.entry_username.setFocus()

    def tentar_login(self):
        username = self.entry_username.text().strip()
        password = self.entry_password.text().strip()

        users = self.carregar_usuarios()

        if username in users and users[username]["password"] == password:
            user_type = users[username]["type"]

            # ? chama callback no App
            if self.on_login_callback:
                self.on_login_callback(username, user_type)

            self.accept()  # fecha o dialog
        else:
            QMessageBox.warning(self, "Erro", "Usuario ou senha incorretos.")

    def carregar_usuarios(self):
        if not os.path.exists(USERS_FILE):
            return {}
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print("Erro ao ler users.json:", e)
            return {}

