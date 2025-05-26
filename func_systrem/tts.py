import pyttsx3
import io
from gtts import gTTS
import pygame
import threading
from datetime import datetime
from func_systrem.constants import last_voice_notification_time, voice_notification_cooldown, fallback_tts_engine

def speak_google_tts(text, lang='uk'):
    global last_voice_notification_time
    
    current_time = datetime.now()
    
    if last_voice_notification_time is not None:
        elapsed = (current_time - last_voice_notification_time).total_seconds()
        if elapsed < voice_notification_cooldown:
            print(f"Занадто часті голосові сповіщення, ігноруємо ({elapsed:.1f} сек < {voice_notification_cooldown} сек)")
            return False
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        
        def play_audio_thread():
            try:
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"Помилка при відтворенні аудіо: {e}")
        
        thread = threading.Thread(target=play_audio_thread)
        thread.daemon = True
        thread.start()
        
        last_voice_notification_time = current_time
        print(f"Голосове сповіщення: {text}")
        return True
        
    except Exception as e:
        print(f"Помилка при створенні/відтворенні аудіо: {e}")
        return False

def initialize_tts_engine():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        ukrainian_voice = None
        for voice in voices:
            if 'UKR' in voice.id.upper() or 'UK' in voice.id.upper():
                ukrainian_voice = voice.id
                break
        
        if ukrainian_voice:
            engine.setProperty('voice', ukrainian_voice)
        
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 0.8)
        
        return engine
    except Exception as e:
        print(f"Помилка ініціалізації синтезатора мови: {e}")
        return None

def speak_fallback(engine, text):
    global last_voice_notification_time
    
    if engine is None:
        print(f"Неможливо озвучити текст: двигун не ініціалізований")
        return False
    
    current_time = datetime.now()
    
    if last_voice_notification_time is not None:
        elapsed = (current_time - last_voice_notification_time).total_seconds()
        if elapsed < voice_notification_cooldown:
            print(f"Занадто часті голосові сповіщення, ігноруємо ({elapsed:.1f} сек < {voice_notification_cooldown} сек)")
            return False
    
    try:
        def speak_thread():
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Помилка при озвученні: {e}")
        
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True
        thread.start()
        
        last_voice_notification_time = current_time
        print(f"Голосове сповіщення (fallback): {text}")
        return True
    except Exception as e:
        print(f"Помилка при запуску озвучення: {e}")
        return False

def speak(text):
    success = speak_google_tts(text)
    
    if not success and fallback_tts_engine:
        success = speak_fallback(fallback_tts_engine, text)
    
    return success

def initialize_tts_engine():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        ukrainian_voice = None
        for voice in voices:
            if 'UKR' in voice.id.upper() or 'UK' in voice.id.upper():
                ukrainian_voice = voice.id
                break
        
        if ukrainian_voice:
            engine.setProperty('voice', ukrainian_voice)
        
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 0.8)
        
        return engine
    except Exception as e:
        print(f"Помилка ініціалізації синтезатора мови: {e}")
        return None 