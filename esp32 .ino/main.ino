#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <EEPROM.h>
#include <esp_wifi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "soc/rtc_cntl_reg.h" 
#include "soc/soc.h"
#include "esp_http_server.h"

const char* ap_ssid = "ESP32-Robot";
const char* ap_password = "12345678";

#define EEPROM_SIZE 64
#define EEPROM_INVERT_LEFT 0
#define EEPROM_INVERT_RIGHT 1
#define EEPROM_INITIALIZED 2

const byte DNS_PORT = 53;
DNSServer dnsServer;

WebServer server(80);

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define LED_FLASH_PIN 4

const int leftMotorPin1 = 12;
const int leftMotorPin2 = 13;
const int enableLeft = 27;

const int rightMotorPin1 = 2;
const int rightMotorPin2 = 14;
const int enableRight = 26;

const int freq = 30000;
const int pwmChannel1 = 0;
const int pwmChannel2 = 1;
const int pwmResolution = 8;
int dutyCycle = 200;

bool invertLeftMotor = false;
bool invertRightMotor = true;

String valueString = String(0);

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";
httpd_handle_t stream_httpd = NULL;

void controlMotor(int pin1, int pin2, bool forward, bool invert) {
  bool actualDirection = forward;
  if (invert) {
    actualDirection = !forward;
  }
  if (actualDirection) {
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, LOW);
  } else {
    digitalWrite(pin1, LOW);
    digitalWrite(pin2, HIGH);
  }
}

void saveSettingsToEEPROM() {
  EEPROM.write(EEPROM_INVERT_LEFT, invertLeftMotor ? 1 : 0);
  EEPROM.write(EEPROM_INVERT_RIGHT, invertRightMotor ? 1 : 0);
  EEPROM.write(EEPROM_INITIALIZED, 1);
  EEPROM.commit();
  Serial.println("Налаштування збережено в EEPROM");
}

void loadSettingsFromEEPROM() {
  if (EEPROM.read(EEPROM_INITIALIZED) == 1) {
    invertLeftMotor = EEPROM.read(EEPROM_INVERT_LEFT) == 1;
    invertRightMotor = EEPROM.read(EEPROM_INVERT_RIGHT) == 1;
    Serial.println("Налаштування завантажено з EEPROM");
    Serial.print("Інверсія лівого мотора: ");
    Serial.println(invertLeftMotor ? "УВІМК" : "ВИМК");
    Serial.print("Інверсія правого мотора: ");
    Serial.println(invertRightMotor ? "УВІМК" : "ВИМК");
  } else {
    saveSettingsToEEPROM();
  }
}

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL;
  char *part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Помилка захоплення кадру");
      res = ESP_FAIL;
    } else {
      if (fb->width > 400) {
        if (fb->format != PIXFORMAT_JPEG) {
          bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
          esp_camera_fb_return(fb);
          fb = NULL;
          if (!jpeg_converted) {
            Serial.println("Помилка конвертації JPEG");
            res = ESP_FAIL;
          }
        } else {
          _jpg_buf_len = fb->len;
          _jpg_buf = fb->buf;
        }
      }
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (_jpg_buf) {
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    if (res != ESP_OK) {
      break;
    }
    delay(10);
  }
  return res;
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 10;
    config.fb_count = 1;
  }
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Помилка ініціалізації камери: 0x%x", err);
    return false;
  }
  return true;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("Сервер потокової передачі відео запущено на порту 81");
  }
}

void setCorsHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
}

void handleRoot() {
  setCorsHeaders();
  server.send(200, "application/json", "{\"message\":\"ESP32 Robot API\",\"status\":\"ok\"}");
}

void handleStatus() {
  setCorsHeaders();
  String json = "{";
  json += "\"ip\":\"" + WiFi.softAPIP().toString() + "\",";
  json += "\"leftInvert\":" + String(invertLeftMotor ? "true" : "false") + ",";
  json += "\"rightInvert\":" + String(invertRightMotor ? "true" : "false") + ",";
  json += "\"speed\":" + String(map(dutyCycle, 200, 255, 25, 100)) + ",";
  json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"connected_clients\":" + String(WiFi.softAPgetStationNum()) + ",";
  json += "\"frame_size\":\"" + String(psramFound() ? "VGA" : "SVGA") + "\"";
  json += "}";
  server.send(200, "application/json", json);
}

void handleSpeed() {
  setCorsHeaders();
  if (server.hasArg("value")) {
    valueString = server.arg("value");
    int value = valueString.toInt();
    if (value >= 0 && value <= 100) {
      if (value == 0) {
        analogWrite(enableLeft, 0);
        analogWrite(enableRight, 0);
        digitalWrite(leftMotorPin1, LOW);
        digitalWrite(leftMotorPin2, LOW);
        digitalWrite(rightMotorPin1, LOW);
        digitalWrite(rightMotorPin2, LOW);
      } else {
        dutyCycle = map(value, 25, 100, 200, 255);
        analogWrite(enableLeft, dutyCycle);
        analogWrite(enableRight, dutyCycle);
      }
      server.send(200, "application/json", "{\"speed\":" + String(value) + "}");
    } else {
      server.send(400, "application/json", "{\"error\":\"Значення швидкості має бути від 0 до 100\"}");
    }
  } else {
    server.send(400, "application/json", "{\"error\":\"Не вказано значення швидкості\"}");
  }
}

void handleForward() {
  setCorsHeaders();
  Serial.println("Вперед");
  controlMotor(leftMotorPin1, leftMotorPin2, true, invertLeftMotor);
  controlMotor(rightMotorPin1, rightMotorPin2, true, invertRightMotor);
  analogWrite(enableLeft, dutyCycle);
  analogWrite(enableRight, dutyCycle);
  server.send(200, "application/json", "{\"command\":\"forward\",\"status\":\"ok\"}");
}

void handleLeft() {
  setCorsHeaders();
  Serial.println("Вліво");
  controlMotor(leftMotorPin1, leftMotorPin2, true, invertLeftMotor);
  controlMotor(rightMotorPin1, rightMotorPin2, true, invertRightMotor);
  int leftSpeed = dutyCycle / 2;
  int rightSpeed = dutyCycle;
  analogWrite(enableLeft, leftSpeed);
  analogWrite(enableRight, rightSpeed);
  server.send(200, "application/json", "{\"command\":\"left\",\"status\":\"ok\"}");
}

void handleStop() {
  setCorsHeaders();
  Serial.println("Стоп");
  digitalWrite(leftMotorPin1, LOW);
  digitalWrite(leftMotorPin2, LOW);
  digitalWrite(rightMotorPin1, LOW);
  digitalWrite(rightMotorPin2, LOW);
  analogWrite(enableLeft, 0);
  analogWrite(enableRight, 0);
  server.send(200, "application/json", "{\"command\":\"stop\",\"status\":\"ok\"}");
}

void handleRight() {
  setCorsHeaders();
  Serial.println("Вправо");
  controlMotor(leftMotorPin1, leftMotorPin2, true, invertLeftMotor);
  controlMotor(rightMotorPin1, rightMotorPin2, false, invertRightMotor);
  int leftSpeed = dutyCycle;
  int rightSpeed = dutyCycle / 2;
  analogWrite(enableLeft, leftSpeed);
  analogWrite(enableRight, rightSpeed);
  server.send(200, "application/json", "{\"command\":\"right\",\"status\":\"ok\"}");
}

void handleReverse() {
  setCorsHeaders();
  Serial.println("Назад");
  controlMotor(leftMotorPin1, leftMotorPin2, false, invertLeftMotor);
  controlMotor(rightMotorPin1, rightMotorPin2, false, invertRightMotor);
  analogWrite(enableLeft, dutyCycle);
  analogWrite(enableRight, dutyCycle);
  server.send(200, "application/json", "{\"command\":\"reverse\",\"status\":\"ok\"}");
}

void handleTestLeftMotor() {
  setCorsHeaders();
  Serial.println("Тест лівого мотора");
  digitalWrite(rightMotorPin1, LOW);
  digitalWrite(rightMotorPin2, LOW);
  analogWrite(enableRight, 0);
  controlMotor(leftMotorPin1, leftMotorPin2, true, invertLeftMotor);
  analogWrite(enableLeft, dutyCycle);
  server.send(200, "application/json", "{\"command\":\"test_left\",\"status\":\"ok\"}");
  delay(2000);
  digitalWrite(leftMotorPin1, LOW);
  digitalWrite(leftMotorPin2, LOW);
  analogWrite(enableLeft, 0);
}

void handleTestRightMotor() {
  setCorsHeaders();
  Serial.println("Тест правого мотора");
  digitalWrite(leftMotorPin1, LOW);
  digitalWrite(leftMotorPin2, LOW);
  analogWrite(enableLeft, 0);
  controlMotor(rightMotorPin1, rightMotorPin2, true, invertRightMotor);
  analogWrite(enableRight, dutyCycle);
  server.send(200, "application/json", "{\"command\":\"test_right\",\"status\":\"ok\"}");
  delay(2000);
  digitalWrite(rightMotorPin1, LOW);
  digitalWrite(rightMotorPin2, LOW);
  analogWrite(enableRight, 0);
}

void handleInvertLeft() {
  setCorsHeaders();
  invertLeftMotor = !invertLeftMotor;
  Serial.print("Інверсія лівого мотора: ");
  Serial.println(invertLeftMotor ? "УВІМК" : "ВИМК");
  server.send(200, "application/json", "{\"command\":\"invert_left\",\"status\":\"ok\",\"value\":" + String(invertLeftMotor ? "true" : "false") + "}");
}

void handleInvertRight() {
  setCorsHeaders();
  invertRightMotor = !invertRightMotor;
  Serial.print("Інверсія правого мотора: ");
  Serial.println(invertRightMotor ? "УВІМК" : "ВИМК");
  server.send(200, "application/json", "{\"command\":\"invert_right\",\"status\":\"ok\",\"value\":" + String(invertRightMotor ? "true" : "false") + "}");
}

void handleSaveSettings() {
  setCorsHeaders();
  saveSettingsToEEPROM();
  server.send(200, "application/json", "{\"command\":\"save_settings\",\"status\":\"ok\"}");
}

void handleNotFound() {
  setCorsHeaders();
  String message = "{\"error\":\"not_found\",\"details\":{";
  message += "\"uri\":\"" + server.uri() + "\",";
  message += "\"method\":\"" + String((server.method() == HTTP_GET) ? "GET" : "POST") + "\",";
  message += "\"args\":[";
  for (uint8_t i = 0; i < server.args(); i++) {
    if (i > 0) message += ",";
    message += "{\"" + server.argName(i) + "\":\"" + server.arg(i) + "\"}";
  }
  message += "]}}";
  server.send(404, "application/json", message);
}

void setupAP() {
  WiFi.mode(WIFI_AP);
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.softAP(ap_ssid, ap_password);
  dnsServer.start(DNS_PORT, "*", WiFi.softAPIP());
  Serial.println("Режим точки доступу активовано");
  Serial.print("SSID: ");
  Serial.println(ap_ssid);
  Serial.print("Пароль: ");
  Serial.println(ap_password);
  Serial.print("IP адреса точки доступу: ");
  Serial.println(WiFi.softAPIP());
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n\nЗапуск ESP32 робота з камерою...");
  EEPROM.begin(EEPROM_SIZE);
  loadSettingsFromEEPROM();
  pinMode(leftMotorPin1, OUTPUT);
  pinMode(leftMotorPin2, OUTPUT);
  pinMode(rightMotorPin1, OUTPUT);
  pinMode(rightMotorPin2, OUTPUT);
  pinMode(enableLeft, OUTPUT);
  pinMode(enableRight, OUTPUT);
  digitalWrite(leftMotorPin1, LOW);
  digitalWrite(leftMotorPin2, LOW);
  digitalWrite(rightMotorPin1, LOW);
  digitalWrite(rightMotorPin2, LOW);
  pinMode(LED_FLASH_PIN, OUTPUT);  
  setupAP();
  if (initCamera()) {
    Serial.println("Камеру ініціалізовано успішно");
    startCameraServer();
  } else {
    Serial.println("Помилка ініціалізації камери");
  }
  server.on("/", HTTP_OPTIONS, []() {
    setCorsHeaders();
    server.send(200);
  });
  server.on("/", handleRoot);
  server.on("/forward", handleForward);
  server.on("/left", handleLeft);
  server.on("/stop", handleStop);
  server.on("/right", handleRight);
  server.on("/reverse", handleReverse);
  server.on("/speed", handleSpeed);
  server.on("/test_left", handleTestLeftMotor);
  server.on("/test_right", handleTestRightMotor);
  server.on("/invert_left", handleInvertLeft);
  server.on("/invert_right", handleInvertRight);
  server.on("/status", handleStatus);
  server.on("/save_settings", handleSaveSettings);
  server.on("/stream", [](){ server.sendHeader("Location", "http://" + WiFi.softAPIP().toString() + ":81/stream", true); server.send(302, "text/plain", ""); });
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("HTTP сервер запущено");
  pinMode(LED_FLASH_PIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_FLASH_PIN, HIGH);
  dnsServer.processNextRequest();
  server.handleClient();
  delay(2);
}
