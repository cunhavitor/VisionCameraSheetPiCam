import cv2
import datetime
import os

def capture_image(save_dir="data/raw"):
    # Garante que o diretório existe
    os.makedirs(save_dir, exist_ok=True)

    # Inicia a câmera (0 é o índice da câmera padrão)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Erro: não foi possível acessar a câmera.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Erro: não foi possível capturar a imagem.")

    # Gera nome do ficheiro com timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")

    # Salva a imagem capturada
    cv2.imwrite(filename, frame)
    print(f"Imagem capturada e salva em: {filename}")

    return frame
