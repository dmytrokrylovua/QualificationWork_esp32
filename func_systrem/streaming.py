import requests
import time
import cv2
import numpy as np
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

last_frame = None
is_receiving = False
frame_count = 0
last_frame_time = None

def create_session():
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def stream_receiver(url):
    global last_frame, is_receiving, frame_count, last_frame_time
    
    print(f"Потік отримання кадрів запущений для {url}")
    is_receiving = True
    
    try:
        session = create_session()
        response = session.get(url, stream=True, timeout=(10, 15))  # (час очікування з'єднання, час очікування читання)
        
        if response.status_code == 200:
            print("З'єднання встановлено")
            
            bytes_data = bytes()
            last_successful_read = time.time()
            
            while is_receiving:
                try:
                    chunk = None
                    for chunk in response.iter_content(chunk_size=1024):
                        if not is_receiving:
                            break
                        if not chunk:
                            continue
                            
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        
                        if a != -1 and b != -1 and a < b:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            
                            if frame is not None and frame.size > 0:
                                last_frame = frame.copy()
                                frame_count += 1
                                last_frame_time = datetime.now()
                                last_successful_read = time.time()
                        
                        if time.time() - last_successful_read > 15:
                            print("Таймаут читання кадрів, перепідключення...")
                            raise TimeoutError("Таймаут читання кадрів")
                    
                    if chunk is None:
                        print("З'єднання закрито сервером")
                        break
                        
                except TimeoutError as te:
                    print(f"Таймаут: {te}")
                    time.sleep(2)
                    break
                except Exception as e:
                    print(f"Помилка читання потоку: {e}")
                    time.sleep(2)
                    break
            
            if is_receiving:
                print("Спроба перепідключення...")
                time.sleep(2)
                stream_receiver(url)
                
    except requests.exceptions.ConnectTimeout:
        print("Таймаут підключення до сервера")
        time.sleep(3)
        if is_receiving:
            stream_receiver(url)
    except Exception as e:
        print(f"Помилка в потоці отримання: {e}")
        time.sleep(3)
        if is_receiving:
            stream_receiver(url)
    
    print("Потік отримання кадрів зупинено")