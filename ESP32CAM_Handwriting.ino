/*
 * ============================================================
 *  ESP32-CAM Handwriting Detection - Arduino IDE
 *  Based on: ESP32-Work/Text-Recognition-ESP32-CAM
 *  Adapted for: Handwriting OCR with enhanced preprocessing
 * ============================================================
 *
 *  HARDWARE:
 *    - AI-Thinker ESP32-CAM (OV2640)
 *    - FTDI programmer (or USB-TTL adapter) for flashing
 *      → IO0 to GND during upload, remove after
 *
 *  LIBRARIES NEEDED (Arduino IDE → Sketch → Manage Libraries):
 *    - "esp32cam" by Yoursunny  (search: esp32cam)
 *    Board package: esp32 by Espressif (>=2.0.0)
 *    Board: AI Thinker ESP32-CAM (under ESP32 Arduino)
 *
 *  ENDPOINTS served on port 80:
 *    /          → status page with IP
 *    /cam-lo    → low-res  JPEG snapshot (320x240)  - fast
 *    /cam-hi    → high-res JPEG snapshot (800x600)  - for OCR
 *    /cam-mid   → mid-res  JPEG snapshot (640x480)
 *    /stream    → MJPEG stream (continuous)
 *
 *  HOW IT WORKS:
 *    1. ESP32-CAM connects to your WiFi and starts HTTP server.
 *    2. Python script fetches /cam-hi, preprocesses the image,
 *       runs EasyOCR/Tesseract for handwriting recognition,
 *       and displays results in real time.
 * ============================================================
 */

#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>
#include <esp_camera.h>

// ── WiFi credentials ─────────────────────────────────────────
const char* WIFI_SSID = "YOUR_SSID";       // ← change this
const char* WIFI_PASS = "YOUR_PASSWORD";   // ← change this
// ─────────────────────────────────────────────────────────────

WebServer server(80);

// Resolutions
static auto loRes  = esp32cam::Resolution::find(320, 240);
static auto midRes = esp32cam::Resolution::find(640, 480);
static auto hiRes  = esp32cam::Resolution::find(800, 600);

// ── Capture & send one JPEG frame ────────────────────────────
void serveJpg() {
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("[CAM] Capture failed");
    server.send(503, "text/plain", "Camera capture failed");
    return;
  }
  Serial.printf("[CAM] Captured %dx%d  %uB\n",
                frame->getWidth(), frame->getHeight(),
                (unsigned)frame->size());

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

// ── Route handlers ────────────────────────────────────────────
void handleJpgLo() {
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("[CAM] Low-res change failed");
  }
  serveJpg();
}

void handleJpgMid() {
  if (!esp32cam::Camera.changeResolution(midRes)) {
    Serial.println("[CAM] Mid-res change failed");
  }
  serveJpg();
}

void handleJpgHi() {
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("[CAM] High-res change failed");
  }
  serveJpg();
}

void handleCapture() {
  handleJpgHi();
}

// ── MJPEG stream handler ──────────────────────────────────────
void handleStream() {
  if (!esp32cam::Camera.changeResolution(midRes)) {
    Serial.println("[CAM] Stream res change failed");
  }

  WiFiClient client = server.client();
  String boundary = "frame";
  String response = "HTTP/1.1 200 OK\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=" + boundary + "\r\n"
                    "Connection: keep-alive\r\n\r\n";
  client.print(response);

  while (client.connected()) {
    auto frame = esp32cam::capture();
    if (frame == nullptr) {
      Serial.println("[CAM] Stream capture failed");
      delay(100);
      continue;
    }

    client.printf("--%s\r\n"
                  "Content-Type: image/jpeg\r\n"
                  "Content-Length: %u\r\n\r\n",
                  boundary.c_str(), (unsigned)frame->size());
    frame->writeTo(client);
    client.print("\r\n");
    delay(80); // ~12 fps
  }
  Serial.println("[CAM] Stream client disconnected");
}

// ── Status / root page ────────────────────────────────────────
void handleRoot() {
  String ip = WiFi.localIP().toString();
  String html = "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>ESP32-CAM Handwriting Server</title>"
                "<style>body{font-family:monospace;background:#111;color:#eee;max-width:980px;margin:20px auto;padding:16px}"
                "h2{color:#7CFC98} .card{background:#1c1c1c;border:1px solid #333;border-radius:12px;padding:16px;margin:16px 0}"
                "a{color:#8fd3ff} code,pre{background:#0b0b0b;padding:10px;border-radius:8px;display:block;overflow:auto}"
                "img{width:100%;max-width:900px;border:2px solid #444;border-radius:12px}</style></head><body>"
                "<h2>ESP32-CAM Handwriting Server</h2>"
                "<div class='card'><p>Press <b>RST</b> on the ESP32-CAM to restart the sketch and bring the stream up.</p>"
                "<p>IP address to paste into Python: <b>" + ip + "</b></p>"
                "<pre>python3 handwriting_ocr.py --ip " + ip + " --headless --duration 300</pre></div>"
                "<div class='card'><p>Live feed:</p><img src='/stream' alt='Live stream'></div>"
                "<div class='card'><p>Endpoints:</p><ul>"
                "<li><a href='/capture'>/capture</a> high-res snapshot for OCR</li>"
                "<li><a href='/cam-hi'>/cam-hi</a> high-res snapshot for OCR</li>"
                "<li><a href='/cam-mid'>/cam-mid</a> mid-res snapshot</li>"
                "<li><a href='/cam-lo'>/cam-lo</a> low-res snapshot</li>"
                "<li><a href='/stream'>/stream</a> MJPEG live feed</li>"
                "</ul></div>"
                "</body></html>";
  server.send(200, "text/html", html);
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("\n[BOOT] ESP32-CAM Handwriting Detection");
  Serial.println("[BOOT] After flashing, press RST to start the camera server.");

  // Camera config for AI-Thinker board
  esp32cam::Config cfg;
  cfg.setPins(esp32cam::pins::AiThinker);
  cfg.setResolution(hiRes);
  cfg.setBufferCount(2);
  cfg.setJpeg(80); // JPEG quality 0-100 (higher = better OCR, larger size)

  bool ok = esp32cam::Camera.begin(cfg);
  if (!ok) {
    Serial.println("[ERROR] Camera init failed! Check wiring.");
    // Blink LED to signal error
    pinMode(4, OUTPUT); // Flash LED pin
    while (true) {
      digitalWrite(4, HIGH); delay(200);
      digitalWrite(4, LOW);  delay(200);
    }
  }
  Serial.println("[CAM] Camera OK");

  // Fix mirrored captures so text is readable in the browser and OCR pipeline.
  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_hmirror(sensor, 1);
    sensor->set_vflip(sensor, 0);
    Serial.println("[CAM] Image orientation corrected (hmirror=1, vflip=0)");
  } else {
    Serial.println("[WARN] Could not access camera sensor for orientation fix");
  }

  // Connect WiFi
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());

  // Register routes
  server.on("/",        handleRoot);
  server.on("/cam-lo",  handleJpgLo);
  server.on("/cam-mid", handleJpgMid);
  server.on("/cam-hi",  handleJpgHi);
  server.on("/capture",  handleCapture);
  server.on("/stream",  handleStream);

  server.begin();
  Serial.println("[HTTP] Server started");
  Serial.printf("[HTTP] Open http://%s/ in your browser\n", WiFi.localIP().toString().c_str());
}

// ── Loop ──────────────────────────────────────────────────────
void loop() {
  server.handleClient();
}
