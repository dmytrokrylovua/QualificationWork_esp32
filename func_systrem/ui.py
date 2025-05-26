import cv2
import numpy as np
from datetime import datetime
from .utils import cyrilc # Используем относительный импорт
from .constants import (
    UI_ACCENT_COLOR,
    UI_TEXT_COLOR,
    UI_BACKGROUND,
    UI_WARNING_COLOR,
    UI_SUCCESS_COLOR,
    UI_DANGER_COLOR
)

def create_modern_button(width, height, text, is_active=True, alpha=0.9):
    button = np.zeros((height, width, 3), dtype=np.uint8)
    
    if is_active:
        cv2.rectangle(button, (0, 0), (width, height), UI_ACCENT_COLOR, -1)
    else:
        cv2.rectangle(button, (0, 0), (width, height), (50, 50, 50), -1)
    
    text_img = np.zeros((height, width, 3), dtype=np.uint8)
    font_size = height // 2
    text_img = cyrilc(text_img, text, (width//2 - len(text)*font_size//4, height//2 + font_size//3), 
                                font_size=font_size, color=UI_TEXT_COLOR)
    
    button = cv2.addWeighted(button, alpha, text_img, 1.0, 0)
    
    return button

def create_info_image(info_list):
    base_height = 200
    additional_height_per_info = 35
    height = base_height + len(info_list) * additional_height_per_info
    width = 1000

    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        color_value = max(20, min(40, int(20 + (y / height) * 25)))
        img[y, :] = (color_value, color_value, color_value)
    
    header_height = 60
    for y in range(header_height):
        alpha = y / header_height
        color = tuple(int(c1 * (1-alpha) + c2 * alpha) for c1, c2 in zip(UI_ACCENT_COLOR, (0, 80, 150)))
        cv2.line(img, (0, y), (width, y), color, 1)
    
    img = cyrilc(img, "СИСТЕМА РОЗПІЗНАВАННЯ ДОРОЖНІХ ЗНАКІВ", 
                         (width//2, header_height//2), font_size=28, color=UI_TEXT_COLOR, center=True)
    
    cv2.line(img, (0, header_height), (width, header_height), (100, 100, 100), 2)
    cv2.line(img, (0, header_height+2), (width, header_height+2), (50, 50, 50), 1)
    
    logo_center = (50, header_height//2)
    cv2.circle(img, logo_center, 20, (0, 0, 150), -1)
    cv2.circle(img, logo_center, 20, (255, 255, 255), 2)
    cv2.line(img, (logo_center[0]-10, logo_center[1]-10), (logo_center[0]+10, logo_center[1]+10), (255, 255, 255), 2)
    cv2.line(img, (logo_center[0]+10, logo_center[1]-10), (logo_center[0]-10, logo_center[1]+10), (255, 255, 255), 2)
    
    has_sign_info = False
    sign_info = ""
    distance_info = ""
    
    for text in info_list:
        if "Розпізнано знак:" in text:
            has_sign_info = True
            sign_info = text.split(": ")[1]
        elif "Відстань:" in text:
            distance_info = text.split(": ")[1]
    
    y_offset = header_height + 25
    
    if has_sign_info:
        sign_block_height = 90
        for y in range(y_offset, y_offset+sign_block_height):
            alpha = (y - y_offset) / sign_block_height
            color = (int(40 + alpha * 10), int(40 + alpha * 5), int(40 + alpha * 15))
            cv2.rectangle(img, (20, y), (width-20, y), color, -1)
        
        cv2.rectangle(img, (20, y_offset), (width-20, y_offset+sign_block_height), (60, 60, 70), 1)
        
        cv2.rectangle(img, (20, y_offset), (30, y_offset+sign_block_height), UI_SUCCESS_COLOR, -1)
        
        shadow_offset = 1
        img = cyrilc(img, "ВИЯВЛЕНО ДОРОЖНІЙ ЗНАК", 
                             (50+shadow_offset, y_offset+25+shadow_offset), font_size=22, color=(0, 0, 0))
        img = cyrilc(img, "ВИЯВЛЕНО ДОРОЖНІЙ ЗНАК", 
                             (50, y_offset+25), font_size=22, color=(100, 200, 255))
        
        img = cyrilc(img, sign_info, 
                             (50+shadow_offset, y_offset+60+shadow_offset), font_size=26, color=(0, 0, 0))
        img = cyrilc(img, sign_info, 
                             (50, y_offset+60), font_size=26, color=UI_TEXT_COLOR)
        
        if distance_info:
            distance_block_width = 250
            cv2.rectangle(img, (width-distance_block_width-30, y_offset+20), 
                         (width-30, y_offset+70), (0, 70, 120), -1)
            cv2.rectangle(img, (width-distance_block_width-30, y_offset+20), 
                         (width-30, y_offset+70), (0, 100, 170), 1)
            img = cyrilc(img, f"Відстань: {distance_info}", 
                                 (width-distance_block_width-15, y_offset+45), 
                                 font_size=22, color=UI_WARNING_COLOR)
        
        y_offset += sign_block_height + 25
    
    info_count = 0
    for text in info_list:
        if "Розпізнано знак:" in text or "Відстань:" in text:
            continue
            
        color = UI_TEXT_COLOR
        block_color = (40, 40, 45)
        border_color = (70, 70, 75)
        
        if "URL:" in text:
            color = (180, 180, 180)
            block_color = (35, 35, 45)
            border_color = (60, 60, 80)
        elif "Останній кадр:" in text:
            color = (100, 200, 255)
            block_color = (30, 40, 50)
            border_color = (50, 70, 90)
        elif "Останнє оновлення:" in text:
            color = (100, 200, 100)
            block_color = (30, 50, 30)
            border_color = (50, 90, 50)
        
        block_height = 30
        cv2.rectangle(img, (40, y_offset), (width-40, y_offset+block_height), block_color, -1)
        cv2.rectangle(img, (40, y_offset), (width-40, y_offset+block_height), border_color, 1)
        
        shadow_offset = 1
        img = cyrilc(img, text, 
                             (60+shadow_offset, y_offset+20+shadow_offset), font_size=18, color=(0, 0, 0))
        img = cyrilc(img, text, 
                             (60, y_offset+20), font_size=18, color=color)
        
        y_offset += block_height + 5
        info_count += 1
    
    footer_height = 40
    current_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    
    for y in range(height-footer_height, height):
        alpha = (y - (height-footer_height)) / footer_height
        color = tuple(int(c1 * (1-alpha) + c2 * alpha) for c1, c2 in zip((30, 30, 40), (50, 50, 60)))
        cv2.line(img, (0, y), (width, y), color, 1)
    
    img = cyrilc(img, f"Поточний час: {current_time}", 
                         (width-300, height-footer_height+25), font_size=18, color=(180, 180, 180))
    
    cv2.circle(img, (30, height-footer_height//2), 10, UI_ACCENT_COLOR, -1)
    cv2.circle(img, (30, height-footer_height//2), 10, (100, 150, 200), 1)
    
    return img 