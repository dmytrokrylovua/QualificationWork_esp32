import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from classification import ImprovedCNN, SIZE, get_sign_name
from func_systrem.constants import SIGN_NAMES
import albumentations as A
import os
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import webbrowser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str, device: torch.device) -> ImprovedCNN:
    model = ImprovedCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess_image(image: np.ndarray, transform: A.Compose) -> torch.Tensor:
    transformed = transform(image=image)
    image = transformed['image']
    return torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)

def recognize_sign(model: nn.Module, image_path: str, device: torch.device):
    transform = A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Загрузка и предобработка изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Предобработка изображения
    input_tensor = preprocess_image(image, transform).to(device)
    
    # Получение предсказания
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, 1)[0]
        values, indices = torch.topk(probabilities, 3)
    
    predicted_class = indices[0].item()
    confidence = probabilities[predicted_class].item() * 100
    
    # Создание визуализации
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    
    # Добавление зеленого квадрата вокруг знака
    height, width = image.shape[:2]
    border_size = 10
    rect = plt.Rectangle((border_size, border_size), width - 2*border_size, height - 2*border_size, 
                         linewidth=5, edgecolor='green', facecolor='none')
    plt.gca().add_patch(rect)
    
    plt.axis('off')
    
    # Убираем информацию о распознавании с изображения
    plt.tight_layout()
    plt.savefig('recognized_sign.png')
    plt.close()
    
    sign_name = SIGN_NAMES.get(predicted_class, "Невідомий знак")
    
    return {
        'class_id': predicted_class,
        'sign_name': sign_name,
        'confidence': confidence
    }

def create_gui(model, device):
    # Основные цвета интерфейса (светлая тема)
    MAIN_BG = "#ffffff"
    SECONDARY_BG = "#ffffff"
    ACCENT_COLOR = "#5c5c5c"
    TEXT_COLOR = "#212121"
    BUTTON_BG = "#2196F3"
    BUTTON_ACTIVE_BG = "#000000"
    
    root = tk.Tk()
    root.title("Система розпізнавання дорожніх знаків")
    root.geometry("1000x700")
    root.configure(bg=MAIN_BG)
    
    # Настройка стилей для ttk
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TFrame', background=MAIN_BG)
    style.configure('TButton', 
                    background=BUTTON_BG, 
                    foreground="white", 
                    font=('Arial', 12, 'bold'),
                    borderwidth=0)
    style.map('TButton', 
              background=[('active', BUTTON_ACTIVE_BG)],
              foreground=[('active', "white")])
    style.configure('TLabel', 
                    background=MAIN_BG, 
                    foreground=TEXT_COLOR, 
                    font=('Arial', 12))
    style.configure('Header.TLabel', 
                    background=MAIN_BG, 
                    foreground=ACCENT_COLOR, 
                    font=('Arial', 18, 'bold'))
    style.configure('Result.TLabel', 
                    background=MAIN_BG, 
                    foreground=TEXT_COLOR, 
                    font=('Arial', 12),
                    padding=10)
    style.configure('Confidence.TLabel', 
                    background=MAIN_BG, 
                    foreground="#4CAF50", 
                    font=('Arial', 14, 'bold'))
    
    # Функция для выбора изображения
    def select_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Файли зображень", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            # Очистка предыдущих результатов
            status_label.config(text="Обробка зображення...")
            
            # Распознавание знака
            result = recognize_sign(model, file_path, device)
            
            # Отображение результата
            sign_name_label.config(text=f"{result['sign_name']}")
            confidence_label.config(text=f"Точність: {result['confidence']:.2f}%")
            
            # Отображение изображения
            img = Image.open('recognized_sign.png')
            img = img.resize((400, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_label.config(image=photo)
            image_label.image = photo
            
            # Обновление статуса
            status_label.config(text=f"Зображення успішно розпізнано: {os.path.basename(file_path)}")
            
            # Показать результаты
            result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Функция для открытия информации о знаке
    def open_sign_info():
        webbrowser.open("https://roadrules.com.ua/pdr/dorozhni-znaky/")
    
    # Создание верхней панели
    header_frame = ttk.Frame(root, style='TFrame')
    header_frame.pack(fill=tk.X, padx=20, pady=10)
    
    title_label = ttk.Label(header_frame, text="Система розпізнавання дорожніх знаків", 
                           style='Header.TLabel')
    title_label.pack(side=tk.LEFT, pady=10)
    
    # Создание основного контейнера
    main_frame = ttk.Frame(root, style='TFrame')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Левая панель с кнопками
    left_frame = ttk.Frame(main_frame, style='TFrame')
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
    
    select_button = ttk.Button(left_frame, text="Вибрати зображення", command=select_image)
    select_button.pack(fill=tk.X, pady=5)
    
    info_button = ttk.Button(left_frame, text="Інформація про знаки", command=open_sign_info)
    info_button.pack(fill=tk.X, pady=5)
    
    # Статус бар
    status_frame = ttk.Frame(root, style='TFrame')
    status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=10)
    
    status_label = ttk.Label(status_frame, text="Виберіть зображення для розпізнавання", 
                            style='TLabel')
    status_label.pack(side=tk.LEFT)
    
    # Область для отображения результатов
    result_frame = ttk.Frame(main_frame, style='TFrame')
    
    # Область для отображения изображения
    image_frame = ttk.Frame(result_frame, style='TFrame')
    image_frame.pack(side=tk.LEFT, padx=10)
    
    image_label = ttk.Label(image_frame, background=SECONDARY_BG)
    image_label.pack(padx=10, pady=10)
    
    # Область для отображения информации о знаке
    info_frame = ttk.Frame(result_frame, style='TFrame')
    info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
    
    result_header = ttk.Label(info_frame, text="Результат розпізнавання:", 
                             style='Header.TLabel')
    result_header.pack(anchor='w', pady=(0, 10))
    
    # Создаем рамку для результатов с фоном
    result_box = ttk.Frame(info_frame, style='TFrame')
    result_box.pack(fill=tk.BOTH, expand=True)
    result_box.configure(style='Result.TLabel')
    
    sign_name_label = ttk.Label(result_box, text="", style='Header.TLabel')
    sign_name_label.pack(anchor='w', pady=5)
    
    confidence_label = ttk.Label(result_box, text="", style='Confidence.TLabel')
    confidence_label.pack(anchor='w', pady=5)
    
    # Добавляем декоративную линию
    separator = ttk.Separator(root, orient='horizontal')
    separator.pack(fill=tk.X, padx=20, pady=5)
    
    # Информация о приложении
    footer_frame = ttk.Frame(root, style='TFrame')
    footer_frame.pack(fill=tk.X, padx=20, pady=5)
    
    footer_label = ttk.Label(footer_frame, 
                            text="© 2023 Система розпізнавання дорожніх знаків | Версія 1.0", 
                            style='TLabel', font=('Arial', 9))
    footer_label.pack(side=tk.RIGHT)
    
    root.mainloop()

def test_model(model: nn.Module, test_dir: str, device: torch.device) -> dict:
    transform = A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    results = {
        'total': 0,
        'correct': 0,
        'predictions': []
    }
    
    test_path = Path(test_dir)
    csv_path = test_path.parent / "Test.csv"
    
    df = pd.read_csv(csv_path)
    true_classes = dict(zip(df['Path'].apply(lambda x: x.split('/')[-1]), df['ClassId'].astype(int)))
    
    image_paths = sorted(list(test_path.glob('*.png')), key=lambda x: int(x.stem))
    
    ranges = [
        (1, 268, "Тестування знаків на відстані 100 метрів"),
        (269, 500, "Тестування знаків при поганому освітленні"),
        (501, 900, "Тестування знаків з сильним розмиттям"),
        (901, 1500, "Тестування знаків під кутом")
    ]
    
    for start, end, description in ranges:
        current_range_results = {
            'total': 0,
            'correct': 0,
            'predictions': []
        }
        
        logger.info(f"\n{description}")
        logger.info("=" * 50)
        
        current_images = [img for img in image_paths if start <= int(img.stem) <= end]
        
        for img_path in tqdm(current_images, desc=f"Тестування зображень {start}-{end}"):
            img_filename = img_path.name
            true_class = true_classes[img_filename]
            
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            input_tensor = preprocess_image(image, transform).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
            
            predicted_class = predicted.item()
            is_correct = int(predicted_class) == int(true_class)
            prediction_info = {
                'file': img_filename,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'correct': is_correct
            }
            
            current_range_results['predictions'].append(prediction_info)
            results['predictions'].append(prediction_info)
            
            current_range_results['total'] += 1
            results['total'] += 1
            
            if is_correct:
                current_range_results['correct'] += 1
                results['correct'] += 1
        
        if current_range_results['total'] > 0:
            accuracy = (current_range_results['correct'] / current_range_results['total']) * 100
            logger.info(f"Всього зображень: {current_range_results['total']}")
            logger.info(f"Правильно розпізнано: {current_range_results['correct']}")
            logger.info(f"Точність розпізнавання: {accuracy:.2f}%\n")
    
    return results

def print_results(results: dict):
    if results['total'] == 0:
        logger.info("Немає результатів для відображення")
        return
        
    accuracy = (results['correct'] / results['total']) * 100
    logger.info(f"\nЗагальні результати:")
    logger.info(f"Всього зображень: {results['total']}")
    logger.info(f"Правильно розпізнано: {results['correct']}")
    logger.info(f"Точність розпізнавання: {accuracy:.2f}%")
    
    errors = [p for p in results['predictions'] if not p['correct']]
    if errors:
        logger.info(f"\nПомилки розпізнавання:")
        for error in errors[:10]:
            logger.info(f"Файл: {error['file']}, Справжній клас: {error['true_class']}, " +
                       f"Передбачений клас: {error['predicted_class']}")
        if len(errors) > 10:
            logger.info(f"... та ще {len(errors) - 10} помилок")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Використовується пристрій: {device}")
    
    model = load_model("best_traffic_sign_model.pth", device)
    logger.info("Модель успішно завантажено")
    
    # Запуск графического интерфейса для распознавания отдельных изображений
    create_gui(model, device)

if __name__ == "__main__":
    main()