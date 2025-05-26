import os
import sys
import time
import queue
import threading
import logging
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt
import func_systrem.streaming as streaming
import func_systrem.tts as tts
import requests

from classification import ImprovedCNN, SIZE
from func_systrem.utils import cyrilc, test_stream_connection, create_info_image
from func_systrem.detection import detect_traffic_sign
from func_systrem.streaming import stream_receiver
from func_systrem.tts import speak
from func_systrem.image_processing import stabilize_sign_image
from func_systrem.ui import (
    UI_ACCENT_COLOR,
    UI_TEXT_COLOR,
    UI_BACKGROUND,
    UI_WARNING_COLOR,
    UI_SUCCESS_COLOR,
    UI_DANGER_COLOR
)
from func_systrem.constants import (
    SIGN_PRIORITIES,
    DEFAULT_PRIORITY,
    frame_queue,
    results_queue,
    stop_threads,
    min_consecutive_detections,
    max_no_detection_frames,
    DISTANCE_THRESHOLDS,
    prediction_cooldown,
    reset_distances_cooldown,
    no_sign_detection_count,
    announced_distances,
    last_sign_id,
    last_prediction,
    last_prediction_time,
    last_distance,
    last_confidence,
    sign_detection_count,
    global_announced_cache,
    session_start_time,
    last_distance_reset_time,
    fallback_tts_engine,
    voice_notification_cooldown,
    last_voice_notification_time
)
from func_systrem.autopilot import (
    activate_autopilot,
    autopilot_loop,
    deactivate_autopilot,
    robot
)
from func_systrem.sign_data import (
    get_label,
    get_current_sign_id,
    get_current_confidence, 
    get_current_sign_count,
    get_current_distance,
    update_sign_count, 
    reset_sign_count,
    update_sign_data
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_url = 'http://192.168.4.1:81/stream'

frame_queue                 = queue.Queue(maxsize=2)
results_queue               = queue.Queue(maxsize=2)
stop_threads                = False

last_sign_id                = None
last_confidence             = None
last_prediction             = None
last_prediction_time        = None
last_distance               = None

last_distance_reset_time    = datetime.now()

sign_detection_count        = {}
global_announced_cache      = {}

announced_distances         = set()
no_sign_detection_count     = 0

DISTANCE_THRESHOLDS = [25]  # Изменяем на одно значение для озвучивания

def main():
    global last_sign_id, last_confidence, last_prediction,                      \
           last_prediction_time, last_distance, last_distance_reset_time,       \
           announced_distances, sign_detection_count, no_sign_detection_count,  \
           global_announced_cache, fallback_tts_engine,                         \
           voice_notification_cooldown, last_voice_notification_time
    

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    has_gui = True

    from func_systrem.tts import initialize_tts_engine
    fallback_tts_engine = initialize_tts_engine()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = ImprovedCNN()
    if os.path.exists('best_traffic_sign_model.pth'):
        model.load_state_dict(torch.load('best_traffic_sign_model.pth', map_location=device))
    elif os.path.exists('best_traffic_sign_model_checkpoint.pth'):
        checkpoint = torch.load('best_traffic_sign_model_checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    stream_url = base_url.replace('@', '')

    test_urls = [
        stream_url,
        base_url.replace('/stream', '/video'),
        'http://192.168.4.1:81/stream',
    ]

    cv2.namedWindow('Тест', cv2.WINDOW_NORMAL)
    cv2.destroyWindow('Тест')
    
    if has_gui:
        cv2.namedWindow('Розпізнавання', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Інформація', cv2.WINDOW_NORMAL)
        
        cv2.moveWindow('Розпізнавання', 20, 20)
        cv2.moveWindow('Інформація', 350, 520)
        
        cv2.resizeWindow('Розпізнавання', 640, 480)
        cv2.resizeWindow('Інформація', 800, 200)

    working_url = None
    info = [
        "Перевірка доступних URL",
    ]
    
    info_img = create_info_image(info)
    if has_gui:
        cv2.imshow('Інформація', info_img)
        cv2.waitKey(1)
    else:
        plt.imshow(cv2.cvtColor(info_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    for url in test_urls:
        connection_ok, error_msg = test_stream_connection(url)
        if connection_ok:
            working_url = url
            break
        else:
            info = [
                f"Тестування URL: {url}",
                f"Причина: {error_msg}",
            ]
            info_img = create_info_image(info)
            
            if has_gui:
                cv2.imshow('Інформація', info_img)
                cv2.waitKey(1)
            else:
                plt.imshow(cv2.cvtColor(info_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
    
    if working_url is None:
        info = [
            "Натисніть 'q' для виходу"
        ]

        info_img = create_info_image(info)

        if has_gui:
            cv2.imshow('Інформація', info_img)
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
        else:
            plt.imshow(cv2.cvtColor(info_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            return
    else:
        stream_url = working_url
    
    info = [
        f"URL: {stream_url}",
        f"Модель: ({device})",
    ]

    info_img = create_info_image(info)

    if has_gui:
        cv2.imshow('Інформація', info_img)
        cv2.waitKey(1)
    else:
        plt.imshow(cv2.cvtColor(info_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    streaming.is_receiving = True 
    receiver_thread = threading.Thread(target=stream_receiver, args=(stream_url,))
    receiver_thread.daemon = True
    receiver_thread.start()
    time.sleep(2)
    activate_autopilot()
    print("Автопілот активовано успішно")
    
    last_prediction             = None
    last_prediction_time        = None
    last_distance               = None
    last_confidence             = None
    last_sign_id                = None
    
    last_distance_reset_time    = datetime.now()
    announced_distances         = set()
    
    global_announced_cache      = {}
    sign_detection_count        = {}
    no_sign_detection_count     = 0
    
    try:
        while True:
            current_time = datetime.now()
            elapsed_since_reset = (current_time - last_distance_reset_time).total_seconds()
            if elapsed_since_reset > reset_distances_cooldown:
                if last_sign_id is not None:
                    global_announced_cache[last_sign_id] = set()
                announced_distances = set()
                last_distance_reset_time = current_time
                
            autopilot_loop()
            
            info = [
                f"URL: {stream_url}",
                f"Модель: ({device})",
                f"Отримано кадрів: {streaming.frame_count}"
            ]
            
            if streaming.last_frame is not None:
                info.append(f"Останній кадр: Доступний ({streaming.last_frame.shape})")
                
                if streaming.last_frame_time:
                    elapsed = (datetime.now() - streaming.last_frame_time).total_seconds()
                    info.append(f"Останнє оновлення: {elapsed:.2f} сек тому")
            else:
                info.append("Останній кадр: Немає")
            
            if last_prediction is not None:
                info.append(f"Розпізнано знак: {get_label(last_prediction)}")
                if last_confidence is not None:
                    info.append(f"Впевненість: {last_confidence:.1f}%")
                if last_distance is not None:
                    info.append(f"Відстань: {last_distance:.1f} см")
            
            debug_img = create_info_image(info)
            if has_gui:
                cv2.imshow('Інформація', debug_img)
            else:
                plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
            
            if streaming.last_frame is not None:
                frame = streaming.last_frame.copy()
                display_frame = frame.copy()
                
                current_time = datetime.now()
                should_predict = (last_prediction_time is None or 
                                (current_time - last_prediction_time).total_seconds() > prediction_cooldown)
                
                if should_predict:
                    found, signs = detect_traffic_sign(frame)
                    
                    if found and signs:
                        recognized_signs = []
                        
                        for sign_info in signs:
                            sign = sign_info['sign']
                            try:
                                sign = stabilize_sign_image(sign, target_size=(SIZE, SIZE))
                                
                                distance = sign_info['distance']
                                confidence_threshold = 97.0
                                diff_threshold = 30.0
                                
                                if distance > 70:  # Верхний порог остается
                                    continue  # Пропускаем знаки слишком далеко
                                elif distance < 15:  # Нижний порог остается
                                    continue  # Пропускаем знаки слишком близко
                                elif distance > 30 or distance < 20:  # Добавляем проверку диапазона озвучивания
                                    confidence_threshold = 85.0
                                    diff_threshold = 15.0
                                else:  # В диапазоне 20-30 см используем максимальные пороги
                                    confidence_threshold = 97.0
                                    diff_threshold = 30.0
                                
                                if len(sign.shape) == 2:
                                    sign_rgb = cv2.cvtColor(sign, cv2.COLOR_GRAY2RGB)
                                else:
                                    sign_rgb = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                                
                                if distance > 150:
                                    lab = cv2.cvtColor(sign_rgb, cv2.COLOR_RGB2LAB)
                                    l, a, b = cv2.split(lab)
                                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                                    cl = clahe.apply(l)
                                    enhanced_lab = cv2.merge((cl, a, b))
                                    sign_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                                
                                transform = A.Compose([
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    A.GaussianBlur(blur_limit=(3, 3), p=0.5),
                                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                                ])
                                
                                transformed = transform(image=sign_rgb)
                                sign_tensor = torch.FloatTensor(transformed['image']).permute(2, 0, 1).unsqueeze(0)
                                sign_tensor = sign_tensor.to(device)
                                
                                with torch.no_grad():
                                    outputs = model(sign_tensor)
                                
                                probabilities = F.softmax(outputs, dim=1)[0]
                                top_p, top_class = probabilities.topk(2)
                                
                                prediction = top_class[0].item()
                                confidence = top_p[0].item() * 100
                                
                                confidence_diff = (top_p[0] - top_p[1]).item() * 100
                                
                                if confidence > confidence_threshold and confidence_diff > diff_threshold:
                                    recognized_signs.append({
                                        'class_id': prediction,
                                        'confidence': confidence,
                                        'confidence_diff': confidence_diff,
                                        'distance': distance,
                                        'bbox': sign_info['bbox'],
                                        'sign': sign,
                                        'priority': SIGN_PRIORITIES.get(prediction, DEFAULT_PRIORITY)
                                    })
                            except Exception as e:
                                cv2.putText(display_frame, f"Помилка: {str(e)}", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                no_sign_detection_count += 1
                    
                        if recognized_signs:
                            recognized_signs.sort(key=lambda x: (x['priority'], -x['confidence']))
                            best_sign = recognized_signs[0]
                            current_sign_id = best_sign['class_id']
                            
                            for i, sign_data in enumerate(recognized_signs):
                                x, y, w, h = sign_data['bbox']
                                color = (0, 0, 255) if i == 0 else (0, 255, 0)
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
                                label = f"{sign_data['class_id']}: {sign_data['confidence']:.1f}%"
                                cv2.putText(display_frame, label, (x, y-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            if current_sign_id in sign_detection_count:
                                sign_detection_count[current_sign_id] += 1
                            else:
                                old_counts = sign_detection_count.copy()
                                sign_detection_count = {current_sign_id: 1}
                                for sign_id, count in old_counts.items():
                                    if sign_id != current_sign_id and count >= min_consecutive_detections:
                                        sign_detection_count[sign_id] = count
                                if current_sign_id not in global_announced_cache:
                                    global_announced_cache[current_sign_id] = set()
                            
                            no_sign_detection_count = 0
                            
                            if sign_detection_count[current_sign_id] >= min_consecutive_detections:
                                if current_sign_id not in global_announced_cache:
                                    global_announced_cache[current_sign_id] = set()
                                
                                if last_sign_id != current_sign_id:
                                    announced_distances = set()

                                update_sign_data(
                                    sign_id=current_sign_id,
                                    confidence=best_sign['confidence'],
                                    distance=best_sign['distance']
                                )

                                last_prediction = current_sign_id
                                last_prediction_time = current_time
                                
                                # Перемещаем логику озвучивания сюда
                                current_distance = best_sign['distance']
                                if (20 <= current_distance <= 30 and  # Проверяем диапазон расстояния
                                    best_sign['confidence'] >= 97.0 and  # Проверяем уверенность
                                    current_sign_id not in global_announced_cache):  # Проверяем, не был ли знак озвучен
                                    
                                    info_text = f"Знак {get_label(current_sign_id)}"
                                    priority_level = best_sign['priority']
                                    if priority_level <= 2:
                                        info_text += " (високий пріоритет)"
                                    elif priority_level <= 4:
                                        info_text += " (середній пріоритет)"
                                        
                                    speech_text = f"{info_text} на відстані {int(current_distance)} сантиметрів"
                                    speak(speech_text)
                                    
                                    global_announced_cache[current_sign_id] = {current_distance}
                                    print(f"Озвучено знак {get_label(current_sign_id)} на відстані {int(current_distance)} см.")
                                
                                print(f"Впевненість: {best_sign['confidence']}, Відстань: {best_sign['distance']}")
                            else:
                                update_sign_data(
                                    sign_id=current_sign_id,
                                    confidence=best_sign['confidence'],
                                    distance=best_sign['distance']
                                )
                                print(f"Впевненість: {best_sign['confidence']}, Відстань: {best_sign['distance']}")
                            autopilot_loop()
                            
                            info_text = f"Знак {get_label(current_sign_id)}"
                            priority_level = best_sign['priority']
                            priority_text = ""
                            if priority_level <= 2:
                                priority_text = " (високий пріоритет)"
                            elif priority_level <= 4:
                                priority_text = " (середній пріоритет)"
                            info_text += priority_text
                            
                            for threshold in DISTANCE_THRESHOLDS:
                                is_in_local_set = threshold in announced_distances
                                is_in_global_cache = (current_sign_id in global_announced_cache and 
                                                    threshold in global_announced_cache[current_sign_id])
                                
                                # Изменяем логику проверки расстояния для озвучивания
                                if (last_distance is not None and  # Добавляем проверку на None
                                    20 <= last_distance <= 30 and  # Проверяем, что знак в нужном диапазоне
                                    not is_in_local_set and 
                                    not is_in_global_cache and
                                    best_sign['confidence'] >= 97.0):  # Проверяем высокую уверенность
                                    
                                    speech_text = f"{info_text} на відстані {int(last_distance)} сантиметрів"
                                    speak(speech_text)
                                    announced_distances.add(threshold)
                                    if current_sign_id not in global_announced_cache:
                                        global_announced_cache[current_sign_id] = set()
                                    global_announced_cache[current_sign_id].add(threshold)
                                    print(f"Озвучено знак {get_label(current_sign_id)} на відстані {int(last_distance)} см.")
                                    break
                            
                            if len(recognized_signs) > 1:
                                print(f"Виявлено {len(recognized_signs)} знаків:\n {get_label(current_sign_id)}")
                                for i, sign_data in enumerate(recognized_signs):
                                    print(f"  {i+1}. {get_label(sign_data['class_id'])} - "
                                          f"Пріоритет: {sign_data['priority']}, "
                                          f"Впевненість: {sign_data['confidence']:.1f}%, "
                                          f"Відстань: {sign_data['distance']:.1f} см")
                        else:
                            no_sign_detection_count += 1
                    else:
                        no_sign_detection_count += 1
                
                if no_sign_detection_count >= max_no_detection_frames:
                    reset_sign_count()
                    last_prediction = None
                    last_distance = None
                    last_confidence = None
                    no_sign_detection_count = 0
                
                if last_prediction is not None:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                
                if has_gui:
                    cv2.imshow('Розпізнавання', display_frame)
                else:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(121)
                    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    plt.title('Відеопотік')
                    plt.axis('off')
                    plt.subplot(122)
                    plt.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    plt.title('Розпізнавання')
                    plt.axis('off')
                    plt.show()
            else:
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                error_img = cyrilc(error_img, "Немає відеопотоку", (180, 240), font_size=32)
                if has_gui:
                    cv2.imshow('Розпізнавання', error_img)
                else:
                    plt.imshow(cv2.cvtColor(error_img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
            
            if has_gui:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):  
                    if robot.active:
                        deactivate_autopilot()
                        print("Автопілот деактивовано")
                    else:
                        activate_autopilot()
                        print("Автопілот активовано")
            else:
                plt.pause(0.03)
                
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("\nПрограма зупинена користувачем")
    finally:
        stop_url = "http://192.168.4.1/stop"
        response = requests.get(stop_url, timeout=1)
        if response.status_code == 200:
            print("Модель зупинена")
        else:
            print(f"Помилка {response.status_code}")
            
        streaming.is_receiving = False 
        if receiver_thread.is_alive():
            receiver_thread.join(timeout=1)
        if has_gui:
            cv2.destroyAllWindows()
        plt.close('all')
        
if __name__ == "__main__":
    main()
