import cv2
import numpy as np
from typing import Tuple, Optional

def analyze_brightness(image: np.ndarray) -> Tuple[float, float, float]:
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = hsv[..., 2]
    else:
        brightness = image
        
    mean_brightness = np.mean(brightness)
    min_brightness  = np.min(brightness)
    max_brightness  = np.max(brightness)
    
    return mean_brightness, min_brightness, max_brightness

def adaptive_brightness_correction(image: np.ndarray, 
                                target_brightness: float = 127.0,
                                min_brightness: float = 40.0,
                                max_brightness: float = 215.0) -> np.ndarray:
    mean_bright, min_bright, max_bright = analyze_brightness(image)
    
    if min_bright >= min_brightness and max_bright <= max_brightness and \
       abs(mean_bright - target_brightness) < 20:
        return image
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[..., 0]
    
    correction_factor = target_brightness / (mean_bright + 1e-6)
    corrected_l = cv2.convertScaleAbs(l_channel, alpha=correction_factor, beta=0)
    corrected_l = np.clip(corrected_l, min_brightness, max_brightness)
    
    lab[..., 0] = corrected_l
    corrected_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return corrected_image

def enhance_sign_visibility(image: np.ndarray, 
                          is_dark: bool = None,
                          contrast_clip_limit: float = 3.5,
                          tile_size: int = 8) -> np.ndarray:
    if is_dark is None:
        mean_bright, _, _ = analyze_brightness(image)
        is_dark = mean_bright < 85
    
    if is_dark:
        contrast_clip_limit = 4.5
        tile_size = 4
    
    clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, 
                           tileGridSize=(tile_size, tile_size))
    
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[..., 0]
        enhanced_l = clahe.apply(l_channel)
        
        if is_dark:
            enhanced_l = cv2.convertScaleAbs(enhanced_l, alpha=1.4, beta=15)
        
        lab[..., 0] = enhanced_l
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        if is_dark:
            enhanced = adaptive_brightness_correction(
                enhanced,
                target_brightness=170.0,
                min_brightness=70.0,
                max_brightness=245.0
            )
            
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.25, beta=8)
    else:
        enhanced = clahe.apply(image)
        if is_dark:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.9, beta=45)
    
    return enhanced

def stabilize_sign_image(image: np.ndarray,
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    mean_bright, min_bright, max_bright = analyze_brightness(image)
    
    is_dark = mean_bright < 85
    needs_brightness_correction = mean_bright < 85 or mean_bright > 200 or \
                                min_bright < 15 or max_bright > 240
    
    enhanced = enhance_sign_visibility(
        image,
        is_dark=is_dark,
        contrast_clip_limit=3.0 if is_dark else 3.5
    )
    
    if needs_brightness_correction:
        target_brightness = 170.0 if is_dark else 127.0
        enhanced = adaptive_brightness_correction(
            enhanced,
            target_brightness=target_brightness,
            min_brightness=50.0 if is_dark else 30.0,
            max_brightness=245.0
        )
    
    if target_size is not None:
        enhanced = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
    
    return enhanced 