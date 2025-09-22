import time

import cv2
import numpy as np
import json

orb = cv2.ORB_create(nfeatures=1500)

def align_with_template(current_img, template_img, config_path="config/config_alignment.json", resize_scale=0.5):
    """
    Alinha a imagem atual com o template usando ORB + Homografia, redimensionando temporariamente para acelerar o processo.
    """
    start_time = time.perf_counter()

    # Carregar parâmetros
    with open(config_path, "r") as f:
        config = json.load(f)

    max_features = config.get("max_features", 1000)
    good_match_percent = config.get("good_match_percent", 0.2)

    # Função utilitária
    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Redimensionar para acelerar alinhamento
    template_small = cv2.resize(template_img, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current_img, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)

    template_gray = to_gray(template_small)
    current_gray = to_gray(current_small)

    # ORB + Matching
    #orb = cv2.ORB_create(nfeatures=max_features)
    kpts1, desc1 = orb.detectAndCompute(template_gray, None)
    kpts2, desc2 = orb.detectAndCompute(current_gray, None)

    if desc1 is None or desc2 is None:
        raise ValueError("Não foi possível extrair descritores ORB.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    if not matches:
        raise ValueError("Nenhum match encontrado.")

    matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(4, int(len(matches) * good_match_percent))
    good_matches = matches[:num_good]

    if len(good_matches) < 4:
        raise ValueError("Matches insuficientes para homografia.")

    # Pontos ajustados ao tamanho redimensionado
    pts1 = np.float32([kpts1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compensar escala nos pontos
    pts1 /= resize_scale
    pts2 /= resize_scale

    # Calcular homografia nos pontos originais
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None:
        raise ValueError("Homografia falhou.")

    # Aplicar na imagem em alta resolução
    h, w = to_gray(template_img).shape
    aligned = cv2.warpPerspective(current_img, H, (w, h))

    end_time = time.perf_counter()
    print(f"Align Image (com resize) demorou {end_time - start_time:.4f} segundos")

    return aligned, H

