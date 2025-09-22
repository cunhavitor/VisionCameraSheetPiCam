import cv2
import numpy as np
import json
import os


def detect_folha_bordas(aligned_img, config_path="config/config_detect_borda.json", save_path="mask_coords.txt"):
    # Carregar parâmetros do ficheiro JSON
    with open(config_path, "r") as f:
        config = json.load(f)

    blur_ksize = config.get("blur_ksize", 5)
    canny1 = config.get("canny_threshold1", 50)
    canny2 = config.get("canny_threshold2", 150)
    min_area = config.get("min_area", 100000)

    # Converter para escala de cinza
    gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

    # Aplicar blur
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Detecção de bordas
    edges = cv2.Canny(blur, canny1, canny2)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Nenhum contorno encontrado.")

    # Filtrar o maior contorno com área mínima
    folha_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(folha_contour) < min_area:
        raise ValueError("Área do contorno é demasiado pequena para ser a folha.")

    # Bounding box
    x, y, w, h = cv2.boundingRect(folha_contour)
    coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    # Salvar em mask_coords.txt
    with open(save_path, "w") as f:
        for px, py in coords:
            f.write(f"{px},{py}\n")

    print(f"Coordenadas da folha salvas em {save_path}: {coords}")
    return coords
