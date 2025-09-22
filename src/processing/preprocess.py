import cv2
import os
import datetime

def prepare(image, save_dir="data/processed"):
    os.makedirs(save_dir, exist_ok=True)

    # Redimensionar (ex: para 512x512 px — ajusta conforme necessário)
    resized = cv2.resize(image, (512, 512))

    # Converter para escala de cinza
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Equalizar histograma (melhor contraste)
    equalized = cv2.equalizeHist(gray)

    # Guardar imagem processada
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"processed_{timestamp}.jpg")
    cv2.imwrite(filename, equalized)
    print(f"Imagem processada salva em: {filename}")

    return equalized
