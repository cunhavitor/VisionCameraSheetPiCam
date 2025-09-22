from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QPushButton, QMessageBox, QListWidgetItem, QScrollBar
)
from PySide6.QtCore import Qt
import json
import os

class ManageUserWindow(QDialog):
    def __init__(self, parent=None, users_file="config/users.json"):
        super().__init__(parent)
        self.users_file = users_file
        self.users_data = {}
        self.setWindowTitle("Gerenciar Usuários")
        self.setMinimumSize(600, 400)

        # Layout principal
        layout = QVBoxLayout(self)

        # Título
        title = QLabel("Usuários")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Lista de usuários
        self.user_list = QListWidget()
        self.user_list.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.user_list)

        # Botão eliminar
        delete_btn = QPushButton("Eliminar Selecionado")
        delete_btn.clicked.connect(self.delete_selected_users)
        layout.addWidget(delete_btn)

        self.load_users()

    def load_users(self):
        self.user_list.clear()

        if not os.path.exists(self.users_file):
            self.user_list.addItem("Arquivo users.json não encontrado.")
            return

        try:
            with open(self.users_file, "r") as f:
                self.users_data = json.load(f)
        except json.JSONDecodeError:
            self.user_list.addItem("Erro ao ler JSON.")
            return

        for username, info in self.users_data.items():
            user_type = info.get("type", "Desconhecido")
            item = QListWidgetItem(f"{username} ({user_type})")
            self.user_list.addItem(item)

    def delete_selected_users(self):
        selected_items = self.user_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            username = item.text().split(" ")[0]
            self.users_data.pop(username, None)

        # Atualiza ficheiro JSON
        with open(self.users_file, "w") as f:
            json.dump(self.users_data, f, indent=4)

        self.load_users()
