import time
import requests
import urllib3
urllib3.disable_warnings()
import logging
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

from func_systrem.constants import (
    min_consecutive_detections,
    SERVER_IP,
    SMOOTH_STOP_TIME,
    WAIT_AFTER_STOP,
    ACCELERATION_TIME,
    TURN_WAIT_TIME,
    PROHIBITED_WAIT,
    FULL_TURN_TIME
)
from func_systrem.sign_data import (
    get_label, get_current_sign_id,
    get_current_confidence,
    get_current_sign_count,
    get_current_distance
)
from func_systrem.ui import create_info_image

info_messages           = []
current_info_image      = None
current_command         = "Очікування"

danger_signs = [
    "Інша небезпека",
    "Небезпечний поворот ліворуч",
    "Небезпечний поворот праворуч", 
    "Небезпечні повороти",
    "Нерівна дорога",
    "Слизька дорога",
    "Звуження дороги",
    "Дорожні роботи",
    "Світлофорне регулювання",
    "Пішохідний перехід",
    "Діти",
    "Виїзд велосипедистів",
    "Залізничний переїзд зі шлагбаумом",
    "Залізничний переїзд без шлагбаума"
]

stop_signs = [
    "Проїзд без зупинки заборонено",
    "Пішохідний перехід",
    "Залізничний переїзд зі шлагбаумом",
    "Залізничний переїзд без шлагбаума",
    "Дати дорогу"
]

prohibited_signs = [
    "В'їзд заборонено",
    "Рух заборонено",
    "Зупинку заборонено",
    "Стоянку заборонено"
]

priority_signs = [
    "Дорога з пріоритетом",
    "Головна дорога"
]

movement_signs = [
    "Рух ліворуч",
    "Рух праворуч"
]

def update_info_display(message, message_type='info'):
    global info_messages, current_info_image, current_command
    timestamp = time.strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    
    command_message = f"Поточна команда: {current_command}"
    info_messages = [command_message] + [m for m in info_messages if not m.startswith("Поточна команда:")]    
    info_messages.insert(1, formatted_message)

    if len(info_messages) > 10:
        info_messages = info_messages[:10]
    
    current_info_image = create_info_image(info_messages)
    print(formatted_message)

class RobotController:
    def __init__(self):
        self.cached_data = {
            'sign_id': None,
            'confidence': None,
            'distance': None,
            'count': 0,
            'last_update': 0
        }
        self.active             = False
        self.last_command       = None
        self.last_sign          = None
        self.processed_signs    = set()
        self.last_action_time   = time.time()
        self.last_sign_time     = {}
        self.command_cooldown   = 2.0
        self.sign_cooldown      = 15.0
        update_info_display("Система автопілота ініціалізована")

    def send_command(self, command):
        global current_command
        current_time = time.time()
        if command == self.last_command and current_time - self.last_action_time < self.command_cooldown:
            update_info_display(f"Команда {command} уже была отправлена недавно")
            return True
        try:
            if command not in ["forward", "left", "right", "stop"]:
                update_info_display(f"Спроба відправити некоректну команду: {command}")
                return False

            url = f"http://{SERVER_IP}:80/{command}"
            
            if command == "forward":
                current_command = "Рух вперед"
            elif command == "left":
                current_command = "Поворот ліворуч"
            elif command == "right":
                current_command = "Поворот праворуч"
            elif command == "stop":
                current_command = "Зупинка"
            else:
                current_command = command
                
            update_info_display(f"Відправка команди: {command}")
            
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=1.0)
                    if response.status_code == 200:
                        self.last_command = command
                        self.last_action_time = current_time
                        return True
                    elif response.status_code == 503:
                        if attempt == 2:
                            update_info_display("Сервер тимчасово недоступний")
                        time.sleep(1)
                    else:
                        update_info_display(f"Помилка: {response.status_code}")
                        time.sleep(1)
                except requests.exceptions.RequestException as e:
                    if attempt == 2:
                        update_info_display("Немає зв'язку з роботом")
                    time.sleep(1)
            return False
        except Exception as e:
            update_info_display("Помилка підключення до робота")
            return False

    def move_forward(self):
        return self.send_command("forward")

    def turn_left(self):
        update_info_display("Поворот ліворуч")
        return self.send_command("left")

    def turn_right(self):
        update_info_display("Поворот праворуч")
        return self.send_command("right")

    def stop_movement(self):
        update_info_display("Зупинка")
        return self.send_command("stop")

    def handle_danger_sign(self, sign_label):
        if sign_label in self.processed_signs:
            return True
        
        update_info_display(f"Знак небезпеки: {sign_label}")
        self.processed_signs.add(sign_label)
        
        return self.move_forward()

    def handle_stop_sign(self, sign_label):
        if sign_label in self.processed_signs:
            return True
        
        update_info_display(f"Знак зупинки: {sign_label}")
        self.stop_movement()
        time.sleep(5.0)

        
        if sign_label == "Дати дорогу":
            self.stop_movement()
            time.sleep(5.0)
        elif "переїзд" in sign_label:
            self.stop_movement()
            time.sleep(10.0)

        self.processed_signs.add(sign_label)
        return self.move_forward()

    def handle_prohibited_sign(self, sign_label):
        if sign_label in self.processed_signs:
            return True
        
        update_info_display(f"Заборонний знак: {sign_label}")
        
        if sign_label == "В'їзд заборонено":
            self.stop_movement()
            time.sleep(3.0)
            self.turn_left()
            time.sleep(0.40)
            self.stop_movement()
            time.sleep(2.0)
            return self.move_forward()
            
        elif sign_label == "Рух заборонено":
            self.stop_movement()
            time.sleep(3.0)
            self.turn_left()
            time.sleep(0.50)
            self.stop_movement()
            time.sleep(5.0)
            self.processed_signs.add(sign_label)
            return self.move_forward()
            
        elif sign_label in ["Зупинку заборонено", "Стоянку заборонено"]:
            self.processed_signs.add(sign_label)
            return self.move_forward()
        return True

    def handle_priority_sign(self, sign_label):
        if sign_label in self.processed_signs:
            return True
        
        update_info_display(f"Знак пріоритету: {sign_label}")
        
        self.processed_signs.add(sign_label)
        return self.move_forward()

    def handle_movement_sign(self, sign_label):
        if sign_label in self.processed_signs:
            return True
        
        update_info_display(f"Знак напрямку: {sign_label}")
        
        if "ліворуч" in sign_label or sign_label == "Рух ліворуч":
            self.stop_movement()
            time.sleep(2.0)
            self.move_forward()
            time.sleep(0.5)
            self.stop_movement()
            time.sleep(2.0)
            self.turn_left()
            time.sleep(0.06)
            self.stop_movement()
            time.sleep(2.0)
            success = self.move_forward()

        elif "праворуч" in sign_label or sign_label == "Рух праворуч":
            self.stop_movement()
            time.sleep(5.0)
            self.move_forward()
            time.sleep(0.65)
            self.stop_movement()
            time.sleep(5.0)
            self.turn_right()
            time.sleep(0.12)
            self.stop_movement()
            time.sleep(5.0)
            success = self.move_forward()
            time.sleep(5.0)
            self.stop_movement()
            success = True
        else:
            self.stop_movement()
            success = False
            
        if success:
            self.processed_signs.add(sign_label)
        return success

    def handle_sign(self, sign_label):
        update_info_display(f"Обробка знаку: {sign_label}")
        handlers = {
            lambda s: s in danger_signs: self.handle_danger_sign,
            lambda s: s in stop_signs: self.handle_stop_sign,
            lambda s: s in prohibited_signs: self.handle_prohibited_sign,
            lambda s: s in priority_signs: self.handle_priority_sign,
            lambda s: s in movement_signs or "Рух" in s: self.handle_movement_sign
        }
        for condition, handler in handlers.items():
            if condition(sign_label):
                success = handler(sign_label)
                if success:
                    self.last_sign = sign_label
                    self.last_sign_time[sign_label] = time.time()
                return success
        update_info_display(f"Невідомий тип знаку: {sign_label}")
        return False

    def _update_cache(self):
        self.cached_data.update({
            'sign_id': get_current_sign_id(),
            'confidence': get_current_confidence(),
            'distance': get_current_distance(),
            'count': get_current_sign_count().get(get_current_sign_id(), 0) if get_current_sign_id() is not None else 0
        })

    def _can_process_sign(self, sign_label):
        if sign_label in self.last_sign_time:
            time_passed = time.time() - self.last_sign_time[sign_label]
            if time_passed < self.sign_cooldown:
                update_info_display(f"Знак {sign_label} був оброблений нещодавно")
                return False
        return True

    def process_signs(self):
        if not self.active:
            return False
        self._update_cache()
        data = self.cached_data
        if data['sign_id'] is None or data['confidence'] is None:
            if self.last_sign is not None:
                self.last_sign = None
                self.processed_signs.clear()
            if time.time() - self.last_action_time >= self.command_cooldown:
                self.move_forward()
            return False
        if data['count'] >= min_consecutive_detections and data['confidence'] > 85.0:
            sign_label = get_label(data['sign_id'])
            if data['distance'] is not None and data['distance'] <= 55:
                if sign_label != self.last_sign and self._can_process_sign(sign_label):
                    return self.handle_sign(sign_label)
        return False

    def activate(self):
        update_info_display("Активація автопілота", "success")
        self.active = True
        self.move_forward()

    def deactivate(self):
        update_info_display("Деактивація автопілота", "success")
        self.active = False
        self.stop_movement()

robot = RobotController()

def activate_autopilot():
    robot.activate()

def deactivate_autopilot():
    robot.deactivate()

def autopilot_loop():
    robot.process_signs()

def get_current_info_image():
    return current_info_image
