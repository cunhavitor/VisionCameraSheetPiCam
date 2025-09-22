import json

def load_params(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def center_window(window, width, height):
    screen = window.screen().geometry()
    x = (screen.width() - width) // 2
    y = (screen.height() - height) // 2
    window.setGeometry(x, y, width, height)
