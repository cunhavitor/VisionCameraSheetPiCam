import sys
import json
import os
from PySide6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QComboBox, QMessageBox
)
from PySide6.QtCore import Qt



class NewUserWindow(QDialog):
    def __init__(self, parent=None, users_file="users.json"):
        super().__init__(parent)
        self.setWindowTitle("Criar Novo Usuário")
        self.setFixedSize(400, 400)
        self.users_file = users_file

        layout = QVBoxLayout()

        title = QLabel("Novo Utilizador")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Username
        layout.addWidget(QLabel("Username:"))
        self.username_entry = QLineEdit()
        layout.addWidget(self.username_entry)

        # Password
        layout.addWidget(QLabel("Password:"))
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_entry)

        # Confirm password
        layout.addWidget(QLabel("Confirmar Password:"))
        self.confirm_entry = QLineEdit()
        self.confirm_entry.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.confirm_entry)

        # User type
        layout.addWidget(QLabel("Tipo de Utilizador:"))
        self.user_type_combo = QComboBox()
        self.user_type_combo.addItems(["User", "Admin", "SuperAdmin"])
        layout.addWidget(self.user_type_combo)

        # Create button
        create_button = QPushButton("Criar Usuário")
        create_button.clicked.connect(self._criar_usuario)
        layout.addWidget(create_button)

        self.setLayout(layout)

    def _criar_usuario(self):
        username = self.username_entry.text().strip()
        password = self.password_entry.text()
        confirm = self.confirm_entry.text()
        user_type = self.user_type_combo.currentText()

        if not username or not password:
            QMessageBox.critical(self, "Erro", "Preencha todos os campos.")
            return

        if password != confirm:
            QMessageBox.critical(self, "Erro", "As senhas não coincidem.")
            return

        # Carrega os usuários existentes
        if os.path.exists(self.users_file):
            with open(self.users_file, "r") as f:
                users = json.load(f)
        else:
            users = {}

        if username in users:
            QMessageBox.critical(self, "Erro", "Usuário já existe.")
            return

        # Salva novo usuário
        users[username] = {
            "password": password,
            "type": user_type
        }

        with open(self.users_file, "w") as f:
            json.dump(users, f, indent=4)

        QMessageBox.information(self, "Sucesso", f"Usuário '{username}' criado com sucesso!")
        self.accept()  # fecha a janela após criar

