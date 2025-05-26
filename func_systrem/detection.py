import cv2
import numpy as np
import traceback
from func_systrem.constants import SIGN_SIZES

# Матриця камери (відкалібрована для нашої камери)
CAMERA_MATRIX = np.array([
    [320, 0, 320],  # fx, 0, cx
    [0, 320, 240],  # 0, fy, cy
    [0, 0, 1]       # 0, 0, 1
], dtype=np.float32)

# Коефіцієнти дисторсії (спотворення)
DIST_COEFFS = np.zeros((4,1))

def calculate_distance(bbox_width, bbox_height, sign_class, shape_type):
    real_size = SIGN_SIZES.get(sign_class, 70)
    shape_coefficients = {
        0: 1.0,    # Круглий
        1: 0.866,  # Трикутний (cos(30°))
        2: 1.0,    # Квадратний
        3: 0.924   # Восьмикутний (cos(22.5°))
    }
    
    real_size *= shape_coefficients.get(shape_type, 1.0)
    
    if shape_type == 1:
        pixel_size = bbox_height * 0.8 + bbox_width * 0.2
    else:
        pixel_size = (bbox_width + bbox_height) / 2
    
    focal_length = CAMERA_MATRIX[0,0]
    distance = (real_size * focal_length) / pixel_size        
    distance *= 0.12
    distance = np.clip(distance, 15, 500)
    
    return distance
    

def detect_traffic_sign(image):
    try:
        img_copy = image.copy()
        
        img_norm = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX)
        hsv = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 100])
        upper_red2 = np.array([180, 255, 255])
        
        lower_blue = np.array([100, 120, 100])
        upper_blue = np.array([140, 255, 255])
        
        lower_yellow = np.array([20, 120, 100])
        upper_yellow = np.array([35, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        mask_red = cv2.addWeighted(mask_red1, 1.0, mask_red2, 1.0, 0)
        mask = cv2.addWeighted(cv2.addWeighted(mask_red, 1.0, mask_blue, 0.7, 0), 1.0, mask_yellow, 0.5, 0)
        
        kernel_open = np.ones((3,3), np.uint8)
        kernel_close = np.ones((7,7), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        
        valid_signs = []
        min_area = 600
        img_area = image.shape[0] * image.shape[1]
        max_area = img_area * 0.25
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                vertices = len(approx)
                
                if vertices >= 3:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    
                    if 0.7 <= aspect_ratio <= 1.3:
                        if vertices == 3:
                            shape_type = 1
                        elif vertices == 4:
                            shape_type = 2
                        elif vertices >= 8:
                            shape_type = 3 
                        else:
                            shape_type = 0
                        
                        mask_roi = np.zeros_like(mask)
                        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
                        
                        if cv2.bitwise_and(mask_red, mask_roi).any():
                            color_type = 0 
                        elif cv2.bitwise_and(mask_blue, mask_roi).any():
                            color_type = 1
                        elif cv2.bitwise_and(mask_yellow, mask_roi).any():
                            color_type = 2 
                        else:
                            color_type = 3
                        
                        padding = 10
                        y1 = max(0, y - padding)
                        y2 = min(image.shape[0], y + h + padding)
                        x1 = max(0, x - padding)
                        x2 = min(image.shape[1], x + w + padding)
                        
                        sign = image[y1:y2, x1:x2]
                        
                        if sign.size > 0:
                            distance = calculate_distance(w, h, color_type, shape_type)
                            
                            if distance is not None:
                                valid_signs.append({
                                    'sign': sign,
                                    'distance': distance,
                                    'bbox': (x, y, w, h),
                                    'area': area,
                                    'vertices': vertices,
                                    'color_type': color_type,
                                    'shape_type': shape_type
                                })
        
        valid_signs.sort(key=lambda x: x['area'], reverse=True)
        
        return len(valid_signs) > 0, valid_signs
        
    except Exception as e:
        return False, [] 