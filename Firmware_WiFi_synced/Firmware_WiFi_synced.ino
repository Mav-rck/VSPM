/*
 * V.S.P.M. - Vibration Sensing and Predictive Maintenance
 * ESP32 dual-core FreeRTOS firmware - 100-batch run, Wi-Fi (HTTP POST) transmission
 *
 * Hardware (unchanged)
 *   ADXL345 accelerometer : I2C bus 0, SDA = 21, SCL = 22
 *   SSD1306 128x64 OLED   : I2C bus 1, SDA = 32, SCL = 33
 *   Trigger button        : GPIO 13 (INPUT_PULLUP, active LOW)
 *
 * State machine (owner in brackets)
 *
 *   READY --button--> RUNNING (batch 1..100) --100th batch POSTed--> COMPLETE --hold--> READY
 *   UI=Ready          UI=Sensing... / Batch n/100                    UI=100 Batch complete   UI=Ready
 *
 *   READY -> RUNNING   : samplingTask (Core 1), the moment it wakes from the ISR notification
 *   RUNNING -> COMPLETE: transmitTask (Core 0), right after the server accepted batch 100
 *   COMPLETE -> READY  : transmitTask (Core 0), after COMPLETE_HOLD_MS
 *   RUNNING -> READY   : transmitTask (Core 0) after a sensor fault ("I2C Error") or after a
 *                        batch could not be delivered even after retries ("Send Error")
 *   Button presses are ignored unless the state is READY.
 *
 * Core layout
 *   Core 1 (APP_CPU) : samplingTask  - ONLY touches the ADXL345 (I2C bus 0)
 *   Core 0 (PRO_CPU) : transmitTask  - ONLY touches Wi-Fi/HTTP, the OLED and (debug) Serial
 *   The Wi-Fi driver itself also lives on Core 0, so network work stays away from the sampler.
 *
 * Buffer handoff (ping-pong, zero-copy) - same scheme as the serial version
 *   freeQ : indices of EMPTY sample buffers  (Core 0 -> Core 1)
 *   fullQ : messages about FILLED buffers / run events (Core 1 -> Core 0)
 *   A buffer index lives in exactly one place at a time, so a buffer is never read and written
 *   at the same time. Core 0 turns a batch into JSON in a separate staging buffer, returns the
 *   sample buffer to freeQ BEFORE the slow network call, then POSTs. If the network is slower
 *   than sampling, Core 1 blocks on freeQ (backpressure): nothing is overwritten or dropped.
 *
 * Failure handling
 *   A POST is retried (HTTP_MAX_ATTEMPTS total). If it still fails, Core 0 raises stopRequested;
 *   Core 1 sees it at its next batch boundary and ends the run; Core 0 discards whatever is
 *   still queued, shows "Send Error", and returns to READY. A partial run is never reported
 *   as complete.
 *
 * Server contract (matches main.py POST /predict)
 *   POST SERVER_URL      Content-Type: application/json
 *   {"batch_index":1..100,"total_batches":100,"run_id":"r<8 hex>","remove_dc":false,
 *    "is_final":true only on batch 100,"x":[800 ints],"y":[800 ints],"z":[800 ints]}
 *   raw ADXL345 counts, 1 kHz. run_id is new for every button press; batch_index lets the
 *   server ignore a retried duplicate and start a fresh run instead of merging with an
 *   aborted one. Any 2xx status = accepted. 4xx = rejected (not retried). Anything else
 *   (incl. 503 "model not loaded") is retried.
 *
 * [CHANGED vs. previous version - all marked "CHANGED" below]
 *   1. JSON now carries batch_index / total_batches / run_id / remove_dc / is_final.
 *   2. Steady-state HTTP timeout is 1500 ms (spec). Batch 1 keeps a long allowance for a
 *      sleeping Render server; a retry after a dropped connection gets a longer connect time
 *      (a fresh TLS handshake can take more than 1.5 s on an ESP32).
 *   3. setup() pings GET /health once so a sleeping Render server wakes before the first run.
 *   4. OLED texts aligned with the spec (Connecting To Wi-Fi / Connected / Batch Finished).
 *
 * Serial is now ONLY a debug log (DEBUG_LOG). After the tasks start, only Core 0 prints.
 *
 * Libraries: "Adafruit SSD1306", "Adafruit GFX Library" (Library Manager).
 * WiFi, HTTPClient and WiFiClientSecure come with the ESP32 Arduino core.
 */

#include <Arduino.h>
#include <esp_system.h>      // esp_random() for run_id
#include <Wire.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// --------------------------------------------------------------------------
// >>> EDIT THESE <<<  Wi-Fi + server settings
// --------------------------------------------------------------------------
#define WIFI_SSID      "F15 5G"
#define WIFI_PASSWORD  "88888888"
#define SERVER_URL     "https://your-app.onrender.com/predict"   // http:// also works (local test)

// TLS: 1 = encrypt but do NOT verify the server certificate (fine for a demo, NOT for
// production). Set to 0 and fill SERVER_ROOT_CA below to verify the server properly.
#define TLS_INSECURE   1
#if !TLS_INSECURE
static const char SERVER_ROOT_CA[] =
  "-----BEGIN CERTIFICATE-----\n"
  "...paste the root CA of your server here...\n"
  "-----END CERTIFICATE-----\n";
#endif

// --------------------------------------------------------------------------
// Pin / device configuration (unchanged)
// --------------------------------------------------------------------------
#define PIN_ADXL_SDA   21
#define PIN_ADXL_SCL   22
#define PIN_OLED_SDA   32
#define PIN_OLED_SCL   33
#define PIN_BUTTON     13

#define OLED_WIDTH     128
#define OLED_HEIGHT    64
#define OLED_ADDR      0x3C      // some modules use 0x3D
#define I2C_FREQ_HZ    400000    // fast mode on both buses

#define ADXL_ADDR      0x53      // ALT ADDRESS pin low (default)
#define ADXL_REG_DEVID     0x00
#define ADXL_REG_BW_RATE   0x2C
#define ADXL_REG_POWER_CTL 0x2D
#define ADXL_REG_DATA_FMT  0x31
#define ADXL_REG_DATAX0    0x32
#define ADXL_DEVID_VALUE   0xE5

#define ADXL_BW_RATE_1600HZ 0x0E   // ODR 1600 Hz; we read at 1 kHz
#define ADXL_DATA_FMT       0x08   // FULL_RES, +/-2 g (256 LSB/g)
#define ADXL_POWER_MEASURE  0x08

// --------------------------------------------------------------------------
// Sampling / run configuration
// --------------------------------------------------------------------------
#define SAMPLE_RATE_HZ      1000
#define SAMPLE_PERIOD_US    (1000000UL / SAMPLE_RATE_HZ)   // 1000 us
#define BATCH_SIZE          800                            // 0.8 s of data
#define BATCHES_PER_RUN     100                            // batches per button press

#define NUM_BUFFERS         2      // ping-pong
#define TRIGGER_DEBOUNCE_MS 250    // ignore edges closer together than this
#define TRIGGER_SETTLE_MS   0      // optional wait after the press before batch 1
#define COMPLETE_HOLD_MS    1500   // how long "100 Batch complete" stays on screen
#define ERROR_HOLD_MS       1500   // how long error screens stay
#define SHOW_PROGRESS       1      // 1 = show "Batch n/100" on the OLED (~25 ms per batch)
#define DEBUG_LOG           1      // 1 = per-batch log lines on Serial (Core 0 only)

// Task priorities. The sampler sits ABOVE lwIP (18) and the Arduino event task (19) so
// network traffic can never pre-empt the 1 kHz loop on Core 1.
#define SAMPLER_PRIORITY    20
#define TRANSMIT_PRIORITY   2

// Network timing
#define WIFI_CONNECT_TIMEOUT_MS  15000   // per attempt in setup()
#define WIFI_RECONNECT_MS         8000   // how long a POST waits for Wi-Fi to come back
// CHANGED: 1500 ms is the spec'd timeout for normal batches (reused keep-alive connection).
// NOTE: on arduino-esp32 core 2.x HTTPClient rounds the read timeout to whole seconds (~2 s).
#define HTTP_CONNECT_TIMEOUT_MS   1500   // normal batches (connection is reused)
#define HTTP_TIMEOUT_MS           1500   // normal batches (spec: 1500 ms)
#define HTTP_RECONNECT_TIMEOUT_MS 6000   // CHANGED: fresh connection + TLS handshake (batch 1 / retries)
#define HTTP_FIRST_TIMEOUT_MS    60000   // batch 1 of a run: allows a sleeping Render server to wake up
#define HTTP_MAX_ATTEMPTS            3   // 1 try + 2 retries
#define HTTP_RETRY_DELAY_MS        300

// CHANGED: value of "remove_dc" sent to the server. MUST equal what vspm_node.py used when it
// built vspm_training_features.csv (0 = raw counts, 1 = per-axis mean removed).
#define REMOVE_DC_FLAG               0

// JSON staging buffer. Worst case per value: "-32768," = 7 bytes, 2400 values per batch.
// CHANGED: margin 64 -> 192 bytes for the metadata fields in front of the arrays.
#define TX_TEXT_MAX (BATCH_SIZE * 3 * 7 + 192)

#if DEBUG_LOG
  #define LOGF(...) Serial.printf(__VA_ARGS__)
#else
  #define LOGF(...) do {} while (0)
#endif

// One batch of raw samples (800 * 3 * 2 = 4800 bytes).
struct SampleBatch {
  int16_t x[BATCH_SIZE];
  int16_t y[BATCH_SIZE];
  int16_t z[BATCH_SIZE];
};

// Messages from Core 1 to Core 0 on fullQ.
enum MsgType : uint8_t {
  MSG_RUN_START   = 0,
  MSG_BATCH       = 1,
  MSG_RUN_ABORT   = 2,   // sensor (I2C) fault on Core 1
  MSG_RUN_STOPPED = 3    // Core 1 obeyed stopRequested (Core 0 asked for the stop)
};
struct BatchMsg {
  uint8_t  type;
  uint8_t  buf;        // which ping-pong buffer holds the data (MSG_BATCH only)
  uint8_t  number;     // 1..BATCHES_PER_RUN (MSG_BATCH only)
  uint16_t lateUs;     // worst sampling-slot lateness inside this batch (timing diagnostics)
};

enum RunState : uint8_t { STATE_READY = 0, STATE_RUNNING = 1, STATE_COMPLETE = 2 };

// --------------------------------------------------------------------------
// Globals
// --------------------------------------------------------------------------
TwoWire I2C_SENSOR = TwoWire(0);   // I2C bus 0 -> ADXL345
TwoWire I2C_DISPLAY = TwoWire(1);  // I2C bus 1 -> SSD1306

Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &I2C_DISPLAY, -1,
                         I2C_FREQ_HZ, I2C_FREQ_HZ);

volatile RunState runState = STATE_READY;
volatile bool stopRequested = false;           // Core 0 -> Core 1: "end the run at the next batch boundary"
volatile TaskHandle_t samplingTaskHandle = NULL;

QueueHandle_t freeQ = NULL;   // uint8_t buffer indices, Core 0 -> Core 1
QueueHandle_t fullQ = NULL;   // BatchMsg,               Core 1 -> Core 0

static SampleBatch buffers[NUM_BUFFERS];   // ping-pong sample buffers
static char txText[TX_TEXT_MAX];           // JSON staging buffer, owned by Core 0 only

static WiFiClient       plainClient;       // used for http://
static WiFiClientSecure secureClient;      // used for https://
static bool             useTls = false;
static char             lastResp[96];      // start of the last server reply (debug only)
static char             runId[24] = "r0";  // CHANGED: new id per button press, sent as "run_id" (Core 0 only)

// --------------------------------------------------------------------------
// ISR: trigger on GPIO 13 (FALLING edge = button pressed, active LOW)
// --------------------------------------------------------------------------
void IRAM_ATTR onTrigger() {
  static uint32_t lastAcceptedMs = 0;

  // Only READY accepts a trigger; presses during a run / result screen are ignored
  // (this also swallows release bounce).
  if (runState != STATE_READY) return;

  // Debounce: ignore edges too soon after the last ACCEPTED press.
  uint32_t now = millis();
  if ((uint32_t)(now - lastAcceptedMs) < TRIGGER_DEBOUNCE_MS) return;
  lastAcceptedMs = now;

  // Wake samplingTask (sleeps with zero CPU use while idle).
  BaseType_t woken = pdFALSE;
  vTaskNotifyGiveFromISR(samplingTaskHandle, &woken);
  if (woken) portYIELD_FROM_ISR();
}

// --------------------------------------------------------------------------
// OLED helper: one message centered horizontally AND vertically
// (setup() and then Core 0 only)
// --------------------------------------------------------------------------
void showMessage(const char *text) {
  int16_t x1, y1;
  uint16_t w, h;

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextWrap(false);

  display.setTextSize(2);
  display.getTextBounds(text, 0, 0, &x1, &y1, &w, &h);
  if (w > OLED_WIDTH) {
    display.setTextSize(1);
    display.getTextBounds(text, 0, 0, &x1, &y1, &w, &h);
  }

  display.setCursor((OLED_WIDTH - w) / 2 - x1, (OLED_HEIGHT - h) / 2 - y1);
  display.print(text);
  display.display();
}

// --------------------------------------------------------------------------
// ADXL345 low-level I2C helpers (bus 0) - unchanged
// --------------------------------------------------------------------------
bool adxlWrite(uint8_t reg, uint8_t value) {
  I2C_SENSOR.beginTransmission(ADXL_ADDR);
  I2C_SENSOR.write(reg);
  I2C_SENSOR.write(value);
  return I2C_SENSOR.endTransmission() == 0;
}

bool adxlRead(uint8_t reg, uint8_t &value) {
  I2C_SENSOR.beginTransmission(ADXL_ADDR);
  I2C_SENSOR.write(reg);
  if (I2C_SENSOR.endTransmission(false) != 0) return false;   // repeated start
  if (I2C_SENSOR.requestFrom((uint8_t)ADXL_ADDR, (uint8_t)1) != 1) return false;
  value = I2C_SENSOR.read();
  return true;
}

// Burst-read X, Y, Z (6 bytes, little-endian) in a single transaction.
bool adxlReadXYZ(int16_t &x, int16_t &y, int16_t &z) {
  I2C_SENSOR.beginTransmission(ADXL_ADDR);
  I2C_SENSOR.write(ADXL_REG_DATAX0);
  if (I2C_SENSOR.endTransmission(false) != 0) return false;
  if (I2C_SENSOR.requestFrom((uint8_t)ADXL_ADDR, (uint8_t)6) != 6) return false;
  uint8_t b[6];
  for (uint8_t i = 0; i < 6; i++) b[i] = I2C_SENSOR.read();
  x = (int16_t)((b[1] << 8) | b[0]);
  y = (int16_t)((b[3] << 8) | b[2]);
  z = (int16_t)((b[5] << 8) | b[4]);
  return true;
}

bool adxlInit() {
  uint8_t id = 0;
  if (!adxlRead(ADXL_REG_DEVID, id)) return false;     // no ACK on the bus
  if (id != ADXL_DEVID_VALUE) return false;            // wrong device
  if (!adxlWrite(ADXL_REG_POWER_CTL, 0x00)) return false;   // standby while configuring
  if (!adxlWrite(ADXL_REG_BW_RATE, ADXL_BW_RATE_1600HZ)) return false;
  if (!adxlWrite(ADXL_REG_DATA_FMT, ADXL_DATA_FMT)) return false;
  if (!adxlWrite(ADXL_REG_POWER_CTL, ADXL_POWER_MEASURE)) return false;
  delay(10);
  return true;
}

// Unrecoverable boot error. Only used from setup(), before the tasks exist.
void haltWithError(const char *serialMsg, bool oledAvailable) {
  if (oledAvailable) showMessage("I2C Error");
  while (true) {
    Serial.println(serialMsg);
    delay(2000);
  }
}

// --------------------------------------------------------------------------
// Wi-Fi bring-up (setup() only, before the tasks exist). Retries until connected.
// After that, the Wi-Fi stack's auto-reconnect handles drops.
// --------------------------------------------------------------------------
static void connectWifiBlocking() {
  showMessage("Connecting To Wi-Fi");
  WiFi.mode(WIFI_STA);
  WiFi.persistent(false);        // don't wear the flash with credentials on every boot
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);          // no modem sleep: lower and steadier latency per POST

  while (true) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - t0) < WIFI_CONNECT_TIMEOUT_MS) {
      delay(250);
    }
    if (WiFi.status() == WL_CONNECTED) break;

    Serial.println("WiFi connect failed, retrying...");
    showMessage("WiFi Error");
    WiFi.disconnect();
    delay(2000);
    showMessage("Connecting To Wi-Fi");
  }
  Serial.print("WiFi connected, IP ");
  Serial.println(WiFi.localIP());
  showMessage("Connected");
  delay(600);
}

// --------------------------------------------------------------------------
// Collect exactly BATCH_SIZE samples at 1 kHz. Returns false on an I2C failure.
// Fixed-period schedule (nextTick += period), so timing errors never accumulate.
// Also records the worst lateness of any sample slot (diagnostic for Wi-Fi interference).
// Runs on Core 1 only.
// --------------------------------------------------------------------------
bool collectBatch(SampleBatch &batch, uint16_t &maxLateUs) {
  uint32_t nextTick = micros();
  uint32_t maxLate = 0;

  for (int i = 0; i < BATCH_SIZE; i++) {
    int32_t late;
    while ((late = (int32_t)(micros() - nextTick)) < 0) { /* wait for slot */ }
    if ((uint32_t)late > maxLate) maxLate = (uint32_t)late;
    nextTick += SAMPLE_PERIOD_US;

    if (!adxlReadXYZ(batch.x[i], batch.y[i], batch.z[i])) {
      return false;   // sensor disappeared / bus fault
    }
  }
  maxLateUs = (maxLate > 65535) ? 65535 : (uint16_t)maxLate;
  return true;
}

static void sendCtrl(uint8_t type) {
  BatchMsg m = { type, 0, 0, 0 };
  xQueueSend(fullQ, &m, portMAX_DELAY);
}

// --------------------------------------------------------------------------
// Core 1 task (producer): wait for trigger, then sample BATCHES_PER_RUN batches
// --------------------------------------------------------------------------
void samplingTask(void *param) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);   // sleep until the ISR notifies

    runState = STATE_RUNNING;                  // READY -> RUNNING (ISR now ignores presses)
    sendCtrl(MSG_RUN_START);                   // Core 0 shows "Sensing..."

    if (TRIGGER_SETTLE_MS > 0) vTaskDelay(pdMS_TO_TICKS(TRIGGER_SETTLE_MS));

    bool faulted = false;
    bool stopped = false;

    for (uint8_t n = 1; n <= BATCHES_PER_RUN; n++) {
      // Core 0 gave up on the network: end the run here, before producing more data.
      if (stopRequested) { stopped = true; break; }

      // 1) Get an EMPTY buffer. Blocks here (backpressure) if Core 0 is behind.
      uint8_t idx;
      xQueueReceive(freeQ, &idx, portMAX_DELAY);

      if (stopRequested) {                     // may have been raised while we were blocked
        xQueueSend(freeQ, &idx, portMAX_DELAY);
        stopped = true;
        break;
      }

      // 2) Fill it. We exclusively own buffers[idx] until we hand it over.
      uint16_t lateUs = 0;
      if (!collectBatch(buffers[idx], lateUs)) {
        xQueueSend(freeQ, &idx, portMAX_DELAY);   // give the buffer back, nothing leaks
        faulted = true;
        break;
      }

      // 3) Hand it over. After this send we must not touch buffers[idx] again.
      BatchMsg m = { MSG_BATCH, idx, n, lateUs };
      xQueueSend(fullQ, &m, portMAX_DELAY);

      // Give lower-priority tasks on this core (Wi-Fi event handling) a 1-tick window
      // between batches. Without this, a sampler that never blocks would starve them.
      vTaskDelay(1);
    }

    // Exactly one terminal message in the abnormal cases; none after a normal batch 100
    // (Core 0 recognises the end of a normal run by batch number == BATCHES_PER_RUN).
    if (faulted)      sendCtrl(MSG_RUN_ABORT);
    else if (stopped) sendCtrl(MSG_RUN_STOPPED);
  }
}

// --------------------------------------------------------------------------
// JSON staging (Core 0 only)
// --------------------------------------------------------------------------
static char *appendAxis(char *p, char *const end, const int16_t *a) {
  for (int i = 0; i < BATCH_SIZE; i++) {
    if (i == 0) p += snprintf(p, end - p, "%d", a[i]);
    else        p += snprintf(p, end - p, ",%d", a[i]);
  }
  return p;
}

// Builds the /predict body (field names match main.py):
//   {"batch_index":n,"total_batches":100,"run_id":"r<8 hex>","remove_dc":false,
//    "is_final":false,"x":[800],"y":[800],"z":[800]}
// Change this ONE function if the server expects a different JSON shape.
static size_t formatBatchJson(const SampleBatch &b, uint8_t number) {
  char *p = txText;
  char *const end = txText + sizeof(txText);

  p += snprintf(p, end - p,
                "{\"batch_index\":%u,\"total_batches\":%d,\"run_id\":\"%s\","
                "\"remove_dc\":%s,\"is_final\":%s,\"x\":[",
                (unsigned)number, BATCHES_PER_RUN, runId,
                REMOVE_DC_FLAG ? "true" : "false",
                (number == BATCHES_PER_RUN) ? "true" : "false");
  p = appendAxis(p, end, b.x);
  p += snprintf(p, end - p, "],\"y\":[");
  p = appendAxis(p, end, b.y);
  p += snprintf(p, end - p, "],\"z\":[");
  p = appendAxis(p, end, b.z);
  p += snprintf(p, end - p, "]}");
  return (size_t)(p - txText);
}

// --------------------------------------------------------------------------
// Network (Core 0 only)
// --------------------------------------------------------------------------
static WiFiClient &activeClient() {
  return useTls ? (WiFiClient &)secureClient : plainClient;
}

// Wait (without blocking other tasks) for Wi-Fi to be up; auto-reconnect does the work.
static bool ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return true;
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - t0) < WIFI_RECONNECT_MS) {
    vTaskDelay(pdMS_TO_TICKS(100));
  }
  return WiFi.status() == WL_CONNECTED;
}

// One POST of the JSON currently in txText. Returns the HTTP status, or a negative error.
static int postBatch(size_t len, uint32_t connectMs, uint32_t readMs) {
  if (!ensureWifi()) return -100;

  HTTPClient http;
  http.setReuse(true);                        // keep the TCP/TLS connection open between batches
  http.setConnectTimeout(connectMs);
  http.setTimeout(readMs);

  bool began = useTls ? http.begin(secureClient, SERVER_URL)
                      : http.begin(plainClient, SERVER_URL);
  if (!began) return -101;

  http.addHeader("Content-Type", "application/json");
  int code = http.POST((uint8_t *)txText, len);

  lastResp[0] = '\0';
  if (code > 0) {
    // Always read the reply so the keep-alive connection is clean for the next POST.
    String body = http.getString();
    strlcpy(lastResp, body.c_str(), sizeof(lastResp));
  }
  http.end();
  return code;
}

// POST with retries. 2xx = success; 4xx = server rejected the data (no point retrying,
// except 408/429 which are transient). Retries are safe: the server dedupes on
// (run_id, batch_index).
// CHANGED: timeouts per attempt:
//   batch 1 of a run : long connect + read (sleeping Render server, fresh TLS handshake)
//   normal batch     : 1500 ms on the reused connection (spec)
//   any retry        : connection was dropped -> longer connect time for a fresh TLS handshake
static bool sendWithRetry(size_t len, bool firstOfRun, int &code, uint8_t &tries) {
  for (uint8_t attempt = 1; attempt <= HTTP_MAX_ATTEMPTS; attempt++) {
    tries = attempt;

    uint32_t connectMs, readMs;
    if (firstOfRun)        { connectMs = HTTP_FIRST_TIMEOUT_MS;     readMs = HTTP_FIRST_TIMEOUT_MS; }
    else if (attempt > 1)  { connectMs = HTTP_RECONNECT_TIMEOUT_MS; readMs = HTTP_TIMEOUT_MS; }
    else                   { connectMs = HTTP_CONNECT_TIMEOUT_MS;   readMs = HTTP_TIMEOUT_MS; }

    code = postBatch(len, connectMs, readMs);
    if (code >= 200 && code < 300) return true;
    if (code >= 400 && code < 500 && code != 408 && code != 429) return false;
    activeClient().stop();                    // drop a possibly dead keep-alive connection
    vTaskDelay(pdMS_TO_TICKS(HTTP_RETRY_DELAY_MS));
  }
  return false;
}

// CHANGED: best-effort wake-up of a sleeping Render server (setup() only, before the tasks
// exist). Replaces the trailing "/predict" of SERVER_URL with "/health". Failure is not fatal.
static void warmServer() {
  char url[160];
  strlcpy(url, SERVER_URL, sizeof(url));
  char *slash = strrchr(url, '/');
  if (slash == NULL) return;
  strlcpy(slash, "/health", sizeof(url) - (size_t)(slash - url));

  HTTPClient http;
  http.setReuse(true);                        // leave the TLS connection open for batch 1
  http.setConnectTimeout(HTTP_FIRST_TIMEOUT_MS);
  http.setTimeout(HTTP_FIRST_TIMEOUT_MS);
  bool began = useTls ? http.begin(secureClient, url) : http.begin(plainClient, url);
  if (!began) return;
  int code = http.GET();
  Serial.printf("warm-up GET /health -> %d\n", code);
  http.end();
}

// --------------------------------------------------------------------------
// End an unsuccessful run (Core 0): show the error, then go back to READY.
// --------------------------------------------------------------------------
static void endRunWithError(const char *text) {
  showMessage(text);
  vTaskDelay(pdMS_TO_TICKS(ERROR_HOLD_MS));
  showMessage("Ready");
  stopRequested = false;       // reset BEFORE allowing a new trigger
  runState = STATE_READY;
}

// --------------------------------------------------------------------------
// Core 0 task (consumer): JSON + Wi-Fi POST + OLED + owner of the end-of-run states
// --------------------------------------------------------------------------
void transmitTask(void *param) {
  bool failing = false;   // true after a delivery failure: drain the rest of the run silently

  for (;;) {
    BatchMsg m;
    if (xQueueReceive(fullQ, &m, portMAX_DELAY) != pdTRUE) continue;

    switch (m.type) {

      case MSG_RUN_START:
        failing = false;
        snprintf(runId, sizeof(runId), "r%08lx", (unsigned long)esp_random());   // CHANGED: new id per press
        showMessage("Sensing...");
        break;

      case MSG_BATCH: {
        const bool last = (m.number == BATCHES_PER_RUN);

        if (failing) {
          // The run is already lost: just give the buffer back. Batch 100 is the
          // last message the sampler can send in this case.
          xQueueSend(freeQ, &m.buf, portMAX_DELAY);
          if (last) { failing = false; endRunWithError("Send Error"); }
          break;
        }

#if SHOW_PROGRESS
        char prog[24];
        snprintf(prog, sizeof(prog), "Batch %u/%d", m.number, BATCHES_PER_RUN);
        showMessage(prog);
#endif

        // Copy-out as JSON, then immediately return the sample buffer so Core 1
        // can refill it while the network call below is in progress.
        size_t len = formatBatchJson(buffers[m.buf], m.number);
        xQueueSend(freeQ, &m.buf, portMAX_DELAY);

        int code = 0;
        uint8_t tries = 0;
        uint32_t t0 = millis();
        bool ok = sendWithRetry(len, m.number == 1, code, tries);
        LOGF("batch %u/%d -> HTTP %d  %lu ms  tries=%u  late=%uus  %s\n",
             m.number, BATCHES_PER_RUN, code, (unsigned long)(millis() - t0),
             tries, m.lateUs, lastResp);

        if (!ok) {
          failing = true;
          stopRequested = true;               // ask Core 1 to end the run at its next boundary
          if (last) { failing = false; endRunWithError("Send Error"); }
          break;
        }

        if (last) {
          // 100th batch accepted by the server -> RUNNING -> COMPLETE
          runState = STATE_COMPLETE;
          showMessage("Batch Finished");
          vTaskDelay(pdMS_TO_TICKS(COMPLETE_HOLD_MS));

          // UI first, THEN allow a new trigger.
          showMessage("Ready");
          runState = STATE_READY;
        }
        break;
      }

      case MSG_RUN_ABORT:        // sensor fault on Core 1
        LOGF("run aborted: ADXL345 read failed\n");
        failing = false;
        endRunWithError("I2C Error");
        break;

      case MSG_RUN_STOPPED:      // Core 1 obeyed our stop request after a send failure
        LOGF("run stopped after a delivery failure\n");
        failing = false;
        endRunWithError("Send Error");
        break;
    }
  }
}

// --------------------------------------------------------------------------
// Arduino entry points
// --------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);          // debug log only
  delay(200);

  // ---- Bus 1 + OLED first, so later errors can be shown on screen ----
  I2C_DISPLAY.begin(PIN_OLED_SDA, PIN_OLED_SCL, I2C_FREQ_HZ);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    haltWithError("ERROR: SSD1306 OLED init failed (check wiring on SDA=32, SCL=33, address 0x3C).",
                  false);
  }
  showMessage("Booting");

  // ---- Bus 0 + ADXL345 ----
  I2C_SENSOR.begin(PIN_ADXL_SDA, PIN_ADXL_SCL, I2C_FREQ_HZ);
  if (!adxlInit()) {
    haltWithError("ERROR: ADXL345 init failed (check wiring on SDA=21, SCL=22, address 0x53).",
                  true);
  }

  // ---- Wi-Fi + TLS ----
  connectWifiBlocking();
  useTls = (strncmp(SERVER_URL, "https", 5) == 0);
  if (useTls) {
#if TLS_INSECURE
    secureClient.setInsecure();
#else
    secureClient.setCACert(SERVER_ROOT_CA);
#endif
  }
  warmServer();    // CHANGED: wake Render before the first run (OLED still shows "Connected")

  // ---- Queues ----
  // fullQ has room for every buffer plus the RUN_START and one terminal message,
  // so Core 1 never blocks on control messages.
  freeQ = xQueueCreate(NUM_BUFFERS, sizeof(uint8_t));
  fullQ = xQueueCreate(NUM_BUFFERS + 2, sizeof(BatchMsg));
  if (freeQ == NULL || fullQ == NULL) {
    haltWithError("ERROR: could not allocate queues (out of heap).", true);
  }
  for (uint8_t i = 0; i < NUM_BUFFERS; i++) {   // both buffers start EMPTY
    xQueueSend(freeQ, &i, 0);
  }

  // Final "Ready" is drawn BEFORE the tasks exist: no two cores on the OLED at once.
  showMessage("Ready");
  runState = STATE_READY;

  // ---- Task creation with core pinning ----
  // xTaskCreatePinnedToCore(fn, name, stack_bytes, param, priority, handle, core)
  xTaskCreatePinnedToCore(samplingTask, "SamplingTask", 6144, NULL, SAMPLER_PRIORITY,
                          (TaskHandle_t *)&samplingTaskHandle, 1);
  // Big stack: TLS (mbedTLS) + HTTPClient + snprintf need far more than the serial version.
  xTaskCreatePinnedToCore(transmitTask, "TransmitTask", 16384, NULL, TRANSMIT_PRIORITY, NULL, 0);

  // ---- Button + ISR: attached LAST, once the task handle exists ----
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_BUTTON), onTrigger, FALLING);
}

void loop() {
  vTaskDelay(portMAX_DELAY);   // everything happens in the FreeRTOS tasks
}

/*
 * ---------------------------------------------------------------------------
 * HARDWARE VERIFICATION CHECKLIST
 * ---------------------------------------------------------------------------
 *  [ ] Boot: OLED shows Booting -> WiFi... -> WiFi OK -> Ready; Serial prints the IP address.
 *  [ ] Edit WIFI_SSID, WIFI_PASSWORD and SERVER_URL; confirm the URL ends in /predict.
 *  [ ] Server schema: the server accepts batch_index / total_batches / run_id / remove_dc /
 *      is_final + x,y,z[800] (HTTP 200). A 4xx in the log means the server rejected the
 *      shape -> edit formatBatchJson().
 *  [ ] Retry safety: force a retry (e.g. briefly block the server) and confirm the dashboard
 *      counts that batch_index once, not twice.
 *  [ ] One press -> "Sensing...", then "Batch 1/100" ... "Batch 100/100".
 *  [ ] Serial log shows 100 lines "batch n/100 -> HTTP 200", then the OLED shows
 *      "100 Batch complete" right after the last one, then "Ready".
 *  [ ] Render free tier: the first request after idle can take 30-60 s to wake the server.
 *      The first batch of a run waits up to HTTP_FIRST_TIMEOUT_MS for this.
 *  [ ] Presses during the run or during "100 Batch complete" are ignored; one physical
 *      press never starts two runs.
 *  [ ] TIMING: the log prints late=...us (worst sample-slot lateness per batch). Values
 *      well under 1000 us, with no growth while POSTs are in flight, mean Wi-Fi is not
 *      disturbing the 1 kHz sampling. If late= regularly exceeds ~500 us, tell me.
 *  [ ] Run time: expect about 80-90 s if each POST (log "ms" column) stays under ~800 ms.
 *      Slower POSTs make the run longer (sampler waits for a free buffer) but lose no data.
 *  [ ] Fault test 1: switch the router/hotspot off mid-run -> after the retries the OLED
 *      shows "Send Error", then "Ready"; no partial run is reported as complete.
 *  [ ] Fault test 2: unplug the ADXL345 SDA wire mid-run -> "I2C Error", then "Ready".
 *  [ ] Stack check during development: print uxTaskGetStackHighWaterMark(NULL) once at the
 *      end of a run from transmitTask to confirm the 16 KB stack has headroom.
 */
