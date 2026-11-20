/*
 * V.S.P.M. - Vibration Sensing and Predictive Maintenance
 * ESP32 (NodeMCU-32S) dual-core FreeRTOS firmware - 100-batch run version
 *
 * Hardware (unchanged)
 *   ADXL345 accelerometer : I2C bus 0, SDA = 21, SCL = 22
 *   SSD1306 128x64 OLED   : I2C bus 1, SDA = 32, SCL = 33
 *   Trigger button        : GPIO 13 (INPUT_PULLUP, active LOW)
 *
 * State machine (owner in brackets)
 *
 *   READY --button--> RUNNING (batch 1..100) --100th batch sent--> COMPLETE --hold--> READY
 *   UI=Ready          UI=Sensing... / Batch n/100                  UI=100 Batch complete   UI=Ready
 *
 *   READY -> RUNNING   : samplingTask (Core 1), the moment it wakes from the ISR notification
 *   RUNNING -> COMPLETE: transmitTask (Core 0), right after batch 100 is flushed out of the UART
 *   COMPLETE -> READY  : transmitTask (Core 0), after COMPLETE_HOLD_MS
 *   RUNNING -> READY   : transmitTask (Core 0), after a sensor fault ("---RUN ABORTED---")
 *   Button presses are ignored unless the state is READY.
 *
 * Core layout
 *   Core 1 (APP_CPU) : samplingTask  (priority 3) - ONLY touches the ADXL345 (I2C bus 0)
 *   Core 0 (PRO_CPU) : transmitTask  (priority 2) - ONLY touches Serial and the OLED
 *
 * Buffer handoff (ping-pong, zero-copy)
 *   Two sample buffers exist. Two queues carry small messages, never the data:
 *     freeQ : indices of EMPTY buffers  (Core 0 -> Core 1)
 *     fullQ : messages about FILLED buffers / run events (Core 1 -> Core 0)
 *   A buffer index lives in exactly one place at a time (freeQ, the sampler, fullQ,
 *   or the transmitter), so a buffer is never read and written at the same time.
 *   Core 0 converts a batch to text in a separate staging buffer, then returns the
 *   sample buffer to freeQ BEFORE the slow UART transfer, so Core 1 can already be
 *   filling it again while the bytes are still leaving the chip.
 *   Backpressure: if both buffers are full, Core 1 blocks on freeQ (no data is
 *   overwritten or dropped). With a 115200 baud link a batch takes ~1 s to send but
 *   0.8 s to sample, so after a few batches the run is paced by the serial link.
 *   Samples inside a batch are always 1 kHz; there can be gaps BETWEEN batches.
 *
 * Serial output (115200 baud). Per batch (format unchanged):
 *   ---BATCH START---
 *   X,Y,Z          (800 lines, raw ADXL345 counts; 256 LSB/g at +/-2 g full-res)
 *   ---BATCH END---
 * After the 100th batch one extra line follows:
 *   ---RUN COMPLETE---
 * If the sensor fails mid-run, the run ends with:
 *   ---RUN ABORTED---
 *
 * IMPORTANT: once the tasks are running, ONLY transmitTask (Core 0) may write to
 * Serial or the OLED. Any stray print from Core 1 would corrupt the batch stream.
 *
 * Libraries (Arduino Library Manager): "Adafruit SSD1306", "Adafruit GFX Library"
 * The ADXL345 is driven directly over I2C (no library).
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

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

// ODR 1600 Hz (BW_RATE = 0x0E). We read at 1 kHz, so a fresh sample is always
// waiting when we poll.
#define ADXL_BW_RATE_1600HZ 0x0E
// DATA_FORMAT 0x08 = FULL_RES bit set, range +/-2 g (256 LSB/g)
#define ADXL_DATA_FMT       0x08
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
                                   // (e.g. 300 to let finger/button vibration die out)
#define COMPLETE_HOLD_MS    1500   // how long "100 Batch complete" stays on screen
#define ERROR_HOLD_MS       1500   // how long "I2C Error" stays on screen
#define SHOW_PROGRESS       1      // 1 = show "Batch n/100" on the OLED (~25 ms per batch)

// Serial framing strings (the Python collector matches these exactly)
#define MARK_START        "---BATCH START---"
#define MARK_END          "---BATCH END---"
#define MARK_RUN_COMPLETE "---RUN COMPLETE---"
#define MARK_RUN_ABORT    "---RUN ABORTED---"

// Worst case text per sample line is "-32768,-32768,-32768\n" = 21 bytes.
#define TX_TEXT_MAX (BATCH_SIZE * 22 + 128)

// One batch of raw samples (800 * 3 * 2 = 4800 bytes).
struct SampleBatch {
  int16_t x[BATCH_SIZE];
  int16_t y[BATCH_SIZE];
  int16_t z[BATCH_SIZE];
};

// Messages from Core 1 to Core 0 on fullQ.
enum MsgType : uint8_t { MSG_RUN_START = 0, MSG_BATCH = 1, MSG_RUN_ABORT = 2 };
struct BatchMsg {
  uint8_t type;
  uint8_t buf;      // which ping-pong buffer holds the data (MSG_BATCH only)
  uint8_t number;   // 1..BATCHES_PER_RUN (MSG_BATCH only)
};

enum RunState : uint8_t { STATE_READY = 0, STATE_RUNNING = 1, STATE_COMPLETE = 2 };

// --------------------------------------------------------------------------
// Globals
// --------------------------------------------------------------------------
// One hardware I2C peripheral per device so slow OLED traffic can never stall
// the 1 kHz sensor reads.
TwoWire I2C_SENSOR = TwoWire(0);   // I2C bus 0 -> ADXL345
TwoWire I2C_DISPLAY = TwoWire(1);  // I2C bus 1 -> SSD1306

Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &I2C_DISPLAY, -1,
                         I2C_FREQ_HZ, I2C_FREQ_HZ);

volatile RunState runState = STATE_READY;     // see state machine above
volatile TaskHandle_t samplingTaskHandle = NULL;

QueueHandle_t freeQ = NULL;   // uint8_t buffer indices, Core 0 -> Core 1
QueueHandle_t fullQ = NULL;   // BatchMsg,               Core 1 -> Core 0

// Large buffers are static (not on task stacks).
static SampleBatch buffers[NUM_BUFFERS];   // ping-pong sample buffers
static char txText[TX_TEXT_MAX];           // text staging buffer, owned by Core 0 only

// --------------------------------------------------------------------------
// ISR: trigger on GPIO 13 (FALLING edge = button pressed, active LOW)
// --------------------------------------------------------------------------
void IRAM_ATTR onTrigger() {
  static uint32_t lastAcceptedMs = 0;

  // Only the READY state accepts a trigger; presses during a run or during the
  // "complete" screen are ignored (this also swallows release bounce).
  if (runState != STATE_READY) return;

  // Debounce: ignore edges that arrive too soon after the last ACCEPTED press.
  uint32_t now = millis();
  if ((uint32_t)(now - lastAcceptedMs) < TRIGGER_DEBOUNCE_MS) return;
  lastAcceptedMs = now;

  // Wake samplingTask. A task notification is cheaper than a polled flag and
  // lets the sampler sleep with zero CPU use while idle.
  BaseType_t woken = pdFALSE;
  vTaskNotifyGiveFromISR(samplingTaskHandle, &woken);
  if (woken) portYIELD_FROM_ISR();
}

// --------------------------------------------------------------------------
// OLED helper: draw one message centered horizontally AND vertically
// (Core 0 only after the tasks start)
// --------------------------------------------------------------------------
void showMessage(const char *text) {
  int16_t x1, y1;
  uint16_t w, h;

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextWrap(false);

  // Prefer big text (size 2); fall back to size 1 if the string is too wide.
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

// Verify the chip ID and configure ODR / range / measurement mode.
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

// Unrecoverable boot error: report on Serial (and OLED if it works), then halt.
// Only used from setup(), before the tasks exist.
void haltWithError(const char *serialMsg, bool oledAvailable) {
  if (oledAvailable) showMessage("I2C Error");
  while (true) {
    Serial.println(serialMsg);
    delay(2000);
  }
}

// --------------------------------------------------------------------------
// Collect exactly BATCH_SIZE samples at 1 kHz. Returns false on an I2C failure.
// Timing is a fixed-period schedule (nextTick += period), so errors never
// accumulate. Runs on Core 1 only.
// --------------------------------------------------------------------------
bool collectBatch(SampleBatch &batch) {
  uint32_t nextTick = micros();
  for (int i = 0; i < BATCH_SIZE; i++) {
    // Signed subtraction is safe across micros() overflow.
    while ((int32_t)(micros() - nextTick) < 0) { /* wait for slot */ }
    nextTick += SAMPLE_PERIOD_US;

    if (!adxlReadXYZ(batch.x[i], batch.y[i], batch.z[i])) {
      return false;   // sensor disappeared / bus fault
    }
  }
  return true;
}

// Small helper: send a control message (no buffer) to Core 0.
static void sendCtrl(uint8_t type) {
  BatchMsg m = { type, 0, 0 };
  xQueueSend(fullQ, &m, portMAX_DELAY);
}

// --------------------------------------------------------------------------
// Core 1 task (producer): wait for trigger, then sample BATCHES_PER_RUN batches
// --------------------------------------------------------------------------
void samplingTask(void *param) {
  for (;;) {
    // Sleep (no polling) until the ISR gives a notification.
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    runState = STATE_RUNNING;            // READY -> RUNNING (ISR now ignores presses)
    sendCtrl(MSG_RUN_START);             // Core 0 shows "Sensing..."

    if (TRIGGER_SETTLE_MS > 0) vTaskDelay(pdMS_TO_TICKS(TRIGGER_SETTLE_MS));

    bool ok = true;
    for (uint8_t n = 1; n <= BATCHES_PER_RUN; n++) {
      // 1) Get an EMPTY buffer. Blocks here (backpressure) if Core 0 is behind.
      uint8_t idx;
      xQueueReceive(freeQ, &idx, portMAX_DELAY);

      // 2) Fill it. We exclusively own buffers[idx] until we hand it over.
      if (!collectBatch(buffers[idx])) {
        xQueueSend(freeQ, &idx, portMAX_DELAY);   // give the buffer back, nothing leaks
        ok = false;
        break;
      }

      // 3) Hand it over. After this send we must not touch buffers[idx] again.
      BatchMsg m = { MSG_BATCH, idx, n };
      xQueueSend(fullQ, &m, portMAX_DELAY);
    }

    if (!ok) sendCtrl(MSG_RUN_ABORT);
    // Back to the top: wait for the next trigger (state is reset by Core 0).
  }
}

// --------------------------------------------------------------------------
// Convert one batch to text in txText. Returns the number of bytes.
// One big buffer + one Serial.write is much faster than 800 printf() calls.
// --------------------------------------------------------------------------
static size_t formatBatch(const SampleBatch &b, bool lastOfRun) {
  char *p = txText;
  char *const end = txText + sizeof(txText);

  p += snprintf(p, end - p, MARK_START "\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    p += snprintf(p, end - p, "%d,%d,%d\n", b.x[i], b.y[i], b.z[i]);
  }
  p += snprintf(p, end - p, MARK_END "\n");
  if (lastOfRun) p += snprintf(p, end - p, MARK_RUN_COMPLETE "\n");
  return (size_t)(p - txText);
}

// --------------------------------------------------------------------------
// Core 0 task (consumer): serial TX + OLED + owner of the end-of-run states
// --------------------------------------------------------------------------
void transmitTask(void *param) {
  for (;;) {
    BatchMsg m;
    if (xQueueReceive(fullQ, &m, portMAX_DELAY) != pdTRUE) continue;

    switch (m.type) {

      case MSG_RUN_START:
        showMessage("Sensing...");
        break;

      case MSG_BATCH: {
        const bool last = (m.number == BATCHES_PER_RUN);

#if SHOW_PROGRESS
        char prog[24];
        snprintf(prog, sizeof(prog), "Batch %u/%d", m.number, BATCHES_PER_RUN);
        showMessage(prog);
#endif

        // Copy-out as text, then immediately return the sample buffer so
        // Core 1 can refill it while the UART is still busy below.
        size_t len = formatBatch(buffers[m.buf], last);
        xQueueSend(freeQ, &m.buf, portMAX_DELAY);

        Serial.write((const uint8_t *)txText, len);   // blocks until queued to UART
        Serial.flush();                               // wait until the last byte left

        if (last) {
          // 100th batch is fully on the wire -> RUNNING -> COMPLETE
          runState = STATE_COMPLETE;
          char done[32];
          snprintf(done, sizeof(done), "%d Batch complete", BATCHES_PER_RUN);
          showMessage(done);
          vTaskDelay(pdMS_TO_TICKS(COMPLETE_HOLD_MS));

          // UI first, THEN allow a new trigger, so "Ready" can never overwrite
          // a "Sensing..." from a new run.
          showMessage("Ready");
          runState = STATE_READY;
        }
        break;
      }

      case MSG_RUN_ABORT:
        // Sensor fault: earlier batches were already sent (queue order), then this.
        Serial.print(MARK_RUN_ABORT "\n");
        Serial.flush();
        showMessage("I2C Error");
        vTaskDelay(pdMS_TO_TICKS(ERROR_HOLD_MS));
        showMessage("Ready");
        runState = STATE_READY;
        break;
    }
  }
}

// --------------------------------------------------------------------------
// Arduino entry points
// --------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(200);

  // ---- Bring up bus 1 + OLED first, so later errors can be shown on screen ----
  I2C_DISPLAY.begin(PIN_OLED_SDA, PIN_OLED_SCL, I2C_FREQ_HZ);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    haltWithError("ERROR: SSD1306 OLED init failed (check wiring on SDA=32, SCL=33, address 0x3C).",
                  false);
  }
  showMessage("Booting");

  // ---- Bring up bus 0 + ADXL345 ----
  I2C_SENSOR.begin(PIN_ADXL_SDA, PIN_ADXL_SCL, I2C_FREQ_HZ);
  if (!adxlInit()) {
    haltWithError("ERROR: ADXL345 init failed (check wiring on SDA=21, SCL=22, address 0x53).",
                  true);
  }

  // ---- Queues ----
  // freeQ: NUM_BUFFERS indices. fullQ: room for every buffer plus the
  // RUN_START and RUN_ABORT control messages, so Core 1 never blocks on them.
  freeQ = xQueueCreate(NUM_BUFFERS, sizeof(uint8_t));
  fullQ = xQueueCreate(NUM_BUFFERS + 2, sizeof(BatchMsg));
  if (freeQ == NULL || fullQ == NULL) {
    haltWithError("ERROR: could not allocate queues (out of heap).", true);
  }
  // Both buffers start EMPTY, so both indices start in freeQ.
  for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
    xQueueSend(freeQ, &i, 0);
  }

  // Final "Ready" is drawn here, BEFORE the tasks exist, so there is no chance
  // of two cores talking to the OLED at once.
  showMessage("Ready");
  runState = STATE_READY;

  // ---- Task creation with core pinning ----
  // xTaskCreatePinnedToCore(fn, name, stack_bytes, param, priority, handle, core)
  // Core 1: sampler, high priority so the 1 kHz loop is never pre-empted.
  xTaskCreatePinnedToCore(samplingTask, "SamplingTask", 6144, NULL, 3,
                          (TaskHandle_t *)&samplingTaskHandle, 1);
  // Core 0: serial + OLED. Larger stack because of snprintf. Mostly blocked.
  xTaskCreatePinnedToCore(transmitTask, "TransmitTask", 8192, NULL, 2, NULL, 0);

  // ---- Button + ISR: attached LAST, once the task handle exists ----
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_BUTTON), onTrigger, FALLING);
}

void loop() {
  // Everything happens in the FreeRTOS tasks. Park the Arduino loop task forever.
  vTaskDelay(portMAX_DELAY);
}

/*
 * ---------------------------------------------------------------------------
 * HARDWARE VERIFICATION CHECKLIST
 * ---------------------------------------------------------------------------
 *  [ ] Boot: OLED shows "Booting" then "Ready"; no serial text appears at boot.
 *  [ ] One press -> "Sensing...", then "Batch 1/100" ... "Batch 100/100".
 *  [ ] Serial capture shows exactly 100 x (START, 800 lines, END), then one
 *      ---RUN COMPLETE--- line. No other text mixed into the stream.
 *  [ ] Every batch has exactly 800 "x,y,z" lines (the Python collector reports
 *      any bad batch).
 *  [ ] OLED shows "100 Batch complete" right after the last batch, then "Ready".
 *  [ ] Presses during the run or during "100 Batch complete" are ignored.
 *  [ ] A second press after "Ready" starts a clean new run (batch counter restarts).
 *  [ ] Bounce test: one physical press never starts two runs.
 *  [ ] Full run takes roughly 100-110 s at 115200 baud (serial-limited).
 *  [ ] Sample timing: put a known vibration (e.g. 200 Hz motor) on the sensor
 *      and confirm the peak sits at ~200 Hz in an FFT (confirms true 1 kHz rate).
 *  [ ] Fault test: unplug the ADXL345 SDA wire mid-run -> "---RUN ABORTED---"
 *      on serial, "I2C Error" on OLED, then back to "Ready".
 *  [ ] Optional: log uxTaskGetStackHighWaterMark() once during development to
 *      confirm both tasks have stack headroom (never print it during a run).
 */
