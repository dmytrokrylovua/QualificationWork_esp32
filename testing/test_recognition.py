import torch
import os
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, average_precision_score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from func_systrem.constants import SIGN_NAMES  # Исправлено имя модуля
from classification import ImprovedCNN  # Импортируем архитектуру модели
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_image(image_path):
    # Загрузка и предобработка изображения
    img = cv2.imread(image_path)
    
    # Используем трансформации, аналогичные тем, что в classification.py
    transform = A.Compose([
        A.Resize(256, 256),  # Соответствует размеру в модели
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transformed = transform(image=img)
    img = transformed['image']
    img = torch.FloatTensor(img).permute(2, 0, 1)
    
    return img

def calculate_metrics(y_true, y_pred, y_scores):
    # Расчет Precision и Recall для каждого класса
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Расчет AP для каждого класса
    ap_per_class = {}
    for class_id in range(43):
        # Создаем бинарные метки для текущего класса
        y_true_binary = (y_true == class_id).astype(int)
        y_scores_binary = y_scores[:, class_id]
        
        try:
            ap = average_precision_score(y_true_binary, y_scores_binary)
            ap_per_class[class_id] = ap
        except:
            ap_per_class[class_id] = 0
    
    return precisions, recalls, ap_per_class

def test_recognition():
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Загрузка модели
    try:
        model = ImprovedCNN().to(device)
        model.load_state_dict(torch.load('best_traffic_sign_model.pth', map_location=device))
        model.eval()
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка: Не удалось загрузить модель: {str(e)}")
        return

    # Загрузка CSV файла с метками
    try:
        df = pd.read_csv('dataset/Train.csv')
        print(f"Загружено {len(df)} записей из CSV")
    except:
        print("Ошибка: Не удалось загрузить Train.csv")
        return

    # Списки для хранения истинных меток и предсказаний
    y_true = []
    y_pred = []
    y_scores = []

    # Проход по всем изображениям
    with torch.no_grad():  # Отключаем вычисление градиентов
        processed_images = 0
        skipped_images = 0
        for index, row in df.iterrows():
            image_path = os.path.join('dataset', str(row['Path']))
            true_class = row['ClassId']

            try:
                if not os.path.exists(image_path):
                    print(f"Файл не найден: {image_path}")
                    skipped_images += 1
                    continue

                # Предобработка изображения
                img = preprocess_image(image_path)
                img = img.unsqueeze(0).to(device)  # Добавляем размерность батча и переносим на устройство

                # Предсказание
                outputs = model(img)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()

                # Сохранение результатов
                y_true.append(true_class)
                y_pred.append(predicted_class)
                y_scores.append(probabilities[0].cpu().numpy())

                processed_images += 1
            except Exception as e:
                print(f"Ошибка при обработке изображения {image_path}: {str(e)}")
                skipped_images += 1

            if (index + 1) % 1000 == 0:
                print(f"Обработано {index + 1} записей. Успешно обработано: {processed_images}, пропущено: {skipped_images}")

        print(f"\nВсего записей: {len(df)}")
        print(f"Успешно обработано изображений: {processed_images}")
        print(f"Пропущено изображений: {skipped_images}")

    # Проверка, есть ли обработанные изображения
    if not y_scores:
        print("КРИТИЧЕСКАЯ ОШИБКА: Не обработано ни одно изображение!")
        return

    # Преобразование списков в массивы numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.stack(y_scores, axis=0)  # Правильное объединение массивов вероятностей

    # Расчет метрик
    precisions, recalls, ap_per_class = calculate_metrics(y_true, y_pred, y_scores)

    # Вывод результатов
    print("\nРезультаты оценки по классам:")
    print("-" * 140)
    print(f"{'Класс':<5} {'Название знака':<50} {'Precision':<10} {'Recall':<10} {'AP@0.5':<10}")
    print("-" * 140)

    mean_ap = 0
    valid_classes = 0

    for class_id in range(43):
        if class_id in ap_per_class:
            precision = precisions[class_id]
            recall = recalls[class_id]
            ap = ap_per_class[class_id]
            
            if not np.isnan(ap):
                mean_ap += ap
                valid_classes += 1
                
            print(f"{class_id:<5} {SIGN_NAMES[class_id]:<50} {precision:.4f}    {recall:.4f}    {ap:.4f}")

    # Вывод средних метрик
    print("-" * 140)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_ap = mean_ap / valid_classes if valid_classes > 0 else 0

    print(f"\nСредние значения метрик:")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"mAP@0.5: {mean_ap:.4f}")
    print(f"\nВсего обработано изображений: {len(y_true)}")

    # Вывод результатов в виде таблицы
    results_df = pd.DataFrame({
        'Класс': range(43),
        'Название знака': [SIGN_NAMES[class_id] for class_id in range(43)],
        'Precision': precisions,
        'Recall': recalls,
        'AP@0.5': [ap_per_class.get(class_id, 0) for class_id in range(43)]
    })
    print("\nТаблица результатов:")
    print(results_df.to_string(index=False))

    # Создание диаграммы
    plt.figure(figsize=(12, 8))
    plt.bar(results_df['Назва знаку'], results_df['Precision'], color='skyblue')
    plt.xlabel('Назва знаку')
    plt.ylabel('Точність')
    plt.title('Точність по класам')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Отображение диаграммы
    plt.show()

    # Создание тепловой карты
    plt.figure(figsize=(10, 8))
    heatmap_data = results_df.pivot('Класс', 'Назва знаку', 'Precision')
    sns.heatmap(heatmap_data, annot=True, cmap='Blues', cbar_kws={'label': 'Точність'})
    plt.title('Тепловая карта точности по классам')
    plt.xlabel('Назва знаку')
    plt.ylabel('Класс')
    plt.tight_layout()
    
    # Отображение тепловой карты
    plt.show()

if __name__ == "__main__":
    test_recognition() 