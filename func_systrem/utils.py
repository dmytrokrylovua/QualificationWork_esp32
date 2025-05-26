import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from func_systrem.constants import SIGN_NAMES 

def get_label(class_id):
    return SIGN_NAMES.get(class_id, f"Невідомий знак (Клас {class_id})")

def show_image(image, title='Image'):
    plt.figure(figsize=(10, 6))
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def cyrilc(img, text, position, font_size=32, color=(255, 255, 255), thickness=2, center=False):
    try:
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        if center:
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
            position = (position[0] - text_width // 2, position[1] - text_height // 2)
        
        draw.text(position, text, font=font, fill=color)
        
        return np.array(img_pil)
    except Exception as e:
        print(f"Помилка при відображенні тексту: {e}")
        return img

def test_stream_connection(url, timeout=3):
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    print(f"Успішне підключення до {url}")
                    return True, None
        return False, f"Статус: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Помилка запиту: {e}"
    except Exception as e:
        return False, f"Непередбачена помилка: {str(e)}"

def create_info_image(info_list, width=800, height=200, bg_color=(50, 50, 50), text_color=(255, 255, 255)):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = bg_color
    y = 20
    for line in info_list:
        img = cyrilc(img, line, (20, y), font_size=20, color=text_color)
        y += 30
    return img 