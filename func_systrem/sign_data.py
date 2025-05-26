from func_systrem.constants import (
    min_consecutive_detections,
    last_sign_id,
    last_confidence,
    last_distance,
    sign_detection_count,
    SIGN_NAMES
)

def format_output(title, content, color='white'):
    colors = {
        'white': '\033[0m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'cyan': '\033[96m'
    }
    
    color_code = colors.get(color.lower(), colors['white'])
    reset = colors['white']
    
    width = 80
    border = '─' * width
    
    output = f"\n{color_code}┌{border}┐{reset}\n"
    output += f"{color_code}│{title.center(width)}│{reset}\n"
    output += f"{color_code}├{border}┤{reset}\n"
    
    for line in content:
        output += f"{color_code}│ {line.ljust(width-2)}│{reset}\n"
    
    output += f"{color_code}└{border}┘{reset}\n"
    return output

def get_label(class_id):
    return SIGN_NAMES.get(class_id, f"Невідомий знак (Клас {class_id})")


def get_current_sign_id():
    global last_sign_id
    return last_sign_id


def get_current_confidence():
    global last_confidence
    return last_confidence


def get_current_sign_count():
    global sign_detection_count
    return sign_detection_count.copy()


def get_current_distance():
    global last_distance
    return last_distance


def update_sign_count(sign_id):
    global sign_detection_count
    if sign_id is not None:
        if sign_id in sign_detection_count:
            sign_detection_count[sign_id] += 1
        else:
            old_counts = sign_detection_count.copy()
            sign_detection_count = {sign_id: 1}
            for old_id, count in old_counts.items():
                if old_id != sign_id and count >= min_consecutive_detections:
                    sign_detection_count[old_id] = count
        
        content = [
            f"Знак: {get_label(sign_id)}",
            f"Кількість виявлень: {sign_detection_count[sign_id]}"
        ]
        print(format_output("ОНОВЛЕННЯ ЛІЧИЛЬНИКА", content, 'cyan'))
    return sign_detection_count


def reset_sign_count():
    global sign_detection_count
    sign_detection_count = {}


def update_sign_data(sign_id=None, confidence=None, distance=None):
    global last_sign_id, last_confidence, last_distance
    
    old_sign_id = last_sign_id
    old_confidence = last_confidence
    old_distance = last_distance
    
    content = [
        "Старі значення:",
        f"  • ID знака: {old_sign_id} ({get_label(old_sign_id) if old_sign_id is not None else 'Немає'})",
        f"  • Впевненість: {old_confidence:.1f}%" if old_confidence is not None else "  • Впевненість: Немає",
        f"  • Відстань: {old_distance:.1f} см" if old_distance is not None else "  • Відстань: Немає",
        "",
        "Нові значення:"
    ]
    
    changes = []
    if sign_id is not None:
        last_sign_id = sign_id
        changes.append(f"  • ID знака: {sign_id} ({get_label(sign_id)})")
        update_sign_count(sign_id)
    
    if confidence is not None:
        last_confidence = confidence
        changes.append(f"  • Впевненість: {confidence:.1f}%")
    
    if distance is not None:
        last_distance = distance
        changes.append(f"  • Відстань: {distance:.1f} см")
    
    content.extend(changes)
    
    color = 'green' if changes else 'yellow'
    
    print(format_output("ОНОВЛЕННЯ ДАНИХ ПРО ЗНАК", content, color)) 