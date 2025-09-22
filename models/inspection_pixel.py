import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np


class PixelInspectorWindow(ctk.CTkToplevel):
    def __init__(self, master, tpl_img, aligned_img):
        super().__init__(master)
        self.title("Inspector de Pixels")

        # Converte para grayscale e equaliza
        self.t_gray_eq = cv2.equalizeHist(cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY))
        self.a_gray_eq = cv2.equalizeHist(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY))

        self.display_img = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(self.display_img)
        self.tk_img = ImageTk.PhotoImage(self.img_pil)

        self.canvas = ctk.CTkCanvas(self, width=self.img_pil.width, height=self.img_pil.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.tk_img, anchor='nw')

        self.label = ctk.CTkLabel(self, text="Passe o cursor sobre a imagem...")
        self.label.pack(pady=10)

        self.canvas.bind("<Motion>", self.on_mouse_move)

    def enable_pixel_inspection(self, tpl, aligned):
        import cv2
        import numpy as np

        t_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        a_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        t_gray_eq = cv2.equalizeHist(t_gray)
        a_gray_eq = cv2.equalizeHist(a_gray)

        # Salvar para uso no evento
        self._tpl_eq = t_gray_eq
        self._aligned_eq = a_gray_eq

        # Criar a label flutuante
        self.inspection_label = customtkinter.CTkLabel(self.canvas, text="", fg_color="white", text_color="black")
        self.inspection_label.place_forget()

        # Ativar tracking do rato no canvas
        self.canvas.bind("<Motion>", self._update_inspection_label)

    def _update_inspection_label(self, event):
        x, y = event.x, event.y

        # Verifica se está dentro dos limites
        if 0 <= x < self._tpl_eq.shape[1] and 0 <= y < self._tpl_eq.shape[0]:
            t_val = int(self._tpl_eq[y, x])
            a_val = int(self._aligned_eq[y, x])
            diff = abs(t_val - a_val)

            text = f"T: {t_val}  A: {a_val}  Δ: {diff}"
            self.inspection_label.configure(text=text)
            self.inspection_label.place(x=x + 15, y=y + 10)  # deslocada do cursor
        else:
            self.inspection_label.place_forget()
