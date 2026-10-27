/*
 * esp32_servo_controller.ino   (v1.0)
 * ESP32 DevKit + PCA9685 + 2x MG996R (pan/tilt) + 1x MG90S (guarded actuator)
 *
 * The ESP32 is the final safety authority. Python can only REQUEST actions.
 * Every frame:  $BODY*XX\n   (XX = 2-digit hex XOR of every character of BODY)
 *
 * Library: "Adafruit PWM Servo Driver Library" (Library Manager)
 * Board:   "ESP32 Dev Module" (esp32 by Espressif Systems)
 *
 * ALL VALUES MARKED [ASSUMPTION] MUST BE VERIFIED FOR YOUR HARDWARE.
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define FW_VERSION "1.0"

// ============================ HARDWARE CONFIG ============================
constexpr uint8_t  PIN_SDA         = 21;   // [ASSUMPTION] ESP32 DevKit default I2C SDA
constexpr uint8_t  PIN_SCL         = 22;   // [ASSUMPTION] ESP32 DevKit default I2C SCL
constexpr uint8_t  PIN_OE          = 25;   // PCA9685 OE (active LOW enable). 10k pull-up to 3V3 on the board
constexpr uint8_t  PIN_ESTOP_SENSE = 27;   // E-stop contact 2 (NC) to GND, INPUT_PULLUP. HIGH = e-stop / broken wire
constexpr uint32_t I2C_HZ          = 100000;  // raise to 400000 only if your wiring is short and clean
constexpr uint8_t  PCA_ADDR        = 0x40; // [ASSUMPTION] default address, no jumpers soldered
constexpr float    PWM_FREQ_HZ     = 50.0f;   // standard analog servos
constexpr uint32_t OSC_HZ          = 25000000; // PCA9685 internal oscillator (varies per chip, calibrate if needed)

enum Chan : uint8_t { CH_PAN = 0, CH_TILT = 1, CH_ACT = 2, NUM_CH = 3 };

struct ServoCfg {
  uint8_t pcaChannel;        // PCA9685 output number
  float usMin, usMax;        // pulse width mapped to 0 deg and 180 deg   [ASSUMPTION]
  float angMin, angMax;      // allowed angle window (hard clamp at lowest layer)
  float home;                // startup angle (actuator: REST angle)
  float slewDps;             // max speed in degrees per second
  float calUsMin, calUsMax;  // absolute pulse limits while in calibration mode
};

const ServoCfg SERVO[NUM_CH] = {
  /* PAN  */ {0, 600.0f, 2400.0f, 45.0f, 135.0f, 90.0f,  90.0f, 800.0f, 2200.0f},
  /* TILT */ {1, 600.0f, 2400.0f, 60.0f, 120.0f, 90.0f,  90.0f, 800.0f, 2200.0f},
  /* ACT  */ {2, 600.0f, 2400.0f,  0.0f,  60.0f, 20.0f, 300.0f, 600.0f, 2000.0f},
};

constexpr float ACT_ACTIVE_ANGLE = 50.0f;  // [ASSUMPTION] MUST be found by calibration (Phase 2)

// ============================ BEHAVIOUR CONFIG ============================
constexpr uint32_t SERIAL_BAUD        = 115200;
constexpr uint32_t SERVO_UPDATE_MS    = 20;
constexpr uint32_t STAT_PERIOD_MS     = 200;
constexpr uint32_t I2C_CHECK_MS       = 1000;
constexpr uint32_t HB_TIMEOUT_MS      = 1500;  // no valid host frame this long => auto-disarm
constexpr uint32_t ESTOP_ASSERT_MS    = 5;
constexpr uint32_t ESTOP_RELEASE_MS   = 50;
constexpr uint32_t ERR_REPLY_MIN_GAP  = 100;   // limits error replies to noise

// Vision geometry and stability (ESP32 verifies these itself)
constexpr int      FRAME_W            = 640;
constexpr int      FRAME_H            = 480;
constexpr int      CENTER_TOL_PX      = 40;    // must be >= Python deadband
constexpr int      STAB_TOL_PX        = 25;    // allowed drift from the anchor position
constexpr uint32_t GRACE_MS           = 300;   // max tolerated gap between good frames (0 = strict)
constexpr uint32_t TGT_STALE_MS       = 500;   // TGT older than this = target not visible
constexpr float    SETTLE_DEG         = 0.5f;  // head must be within this of its commanded angles
constexpr uint32_t STABLE_DEFAULT_MS  = 5000;
constexpr uint32_t STABLE_MIN_MS      = 100;
constexpr uint32_t STABLE_MAX_MS      = 60000;
constexpr uint32_t REARM_LOSS_MS      = 1500;  // target must be absent this long to leave LOCKOUT

// Actuator cycle
constexpr uint32_t ACT_HOLD_MS        = 600;
constexpr uint32_t ACT_RETURN_MS      = 600;

// Serial parsing
constexpr size_t   RX_MAX             = 96;
constexpr int      MAX_FIELDS         = 8;

// ============================ STATE ============================
enum State : uint8_t {
  ST_BOOT, ST_DISARMED, ST_TRACKING, ST_READY, ST_ACTIVATING, ST_LOCKOUT, ST_ESTOP, ST_FAULT
};

Adafruit_PWMServoDriver pca(PCA_ADDR, Wire);

State state = ST_BOOT;
const char* faultReason = "NONE";

float cur[NUM_CH], tgt[NUM_CH], lastWritten[NUM_CH], calUs[NUM_CH];
bool calMode = false, outputsEnabled = false, pcaOk = false;

bool estopHw = false, estopRawHigh = false;
uint32_t estopRawSince = 0;
uint8_t pcaFail = 0;

uint32_t lastSeq = 0, lastHostMs = 0;
bool hostSeen = false;

uint32_t stableRequiredMs = STABLE_DEFAULT_MS;
bool stabRunning = false;
uint32_t stabStartMs = 0, lastGoodMs = 0, lastTgtMs = 0, hostStableMs = 0;
int anchorX = 0, anchorY = 0, tgtCx = 0, tgtCy = 0;
bool tgtDet = false;

uint32_t readyChallenge = 0;
uint8_t actPhase = 0;
uint32_t actPhaseStart = 0, actCount = 0;
bool lossActive = false;
uint32_t lossStartMs = 0;
uint32_t lastErrReplyMs = 0;

// ============================ HELPERS ============================
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

const char* stateName(State s) {
  switch (s) {
    case ST_BOOT:       return "BOOT";
    case ST_DISARMED:   return "DISARMED";
    case ST_TRACKING:   return "TRACKING";
    case ST_READY:      return "READY";
    case ST_ACTIVATING: return "ACTIVATING";
    case ST_LOCKOUT:    return "LOCKOUT";
    case ST_ESTOP:      return "ESTOP";
    case ST_FAULT:      return "FAULT";
  }
  return "?";
}

bool isArmedState() {
  return state == ST_TRACKING || state == ST_READY || state == ST_ACTIVATING || state == ST_LOCKOUT;
}
bool estopActive() { return estopHw || state == ST_ESTOP; }
bool hostFresh(uint32_t now) { return hostSeen && (now - lastHostMs) <= HB_TIMEOUT_MS; }

float usFromAngle(uint8_t ch, float a) {
  const ServoCfg& s = SERVO[ch];
  return s.usMin + (a / 180.0f) * (s.usMax - s.usMin);
}
float angleFromUs(uint8_t ch, float us) {
  const ServoCfg& s = SERVO[ch];
  return (us - s.usMin) / (s.usMax - s.usMin) * 180.0f;
}

// ---- framing out ----
void sendFrame(const char* body) {
  uint8_t x = 0;
  for (const char* p = body; *p; p++) x ^= (uint8_t)*p;
  Serial.printf("$%s*%02X\n", body, x);
}
void replyAck(uint32_t seq) {
  char b[48];
  snprintf(b, sizeof(b), "ACK,%lu,%s", (unsigned long)seq, stateName(state));
  sendFrame(b);
}
void replyErr(uint32_t seq, const char* code) {
  char b[48];
  snprintf(b, sizeof(b), "ERR,%lu,%s", (unsigned long)seq, code);
  sendFrame(b);
}
void replyErrLimited(uint32_t seq, const char* code) {
  uint32_t now = millis();
  if (now - lastErrReplyMs < ERR_REPLY_MIN_GAP) return;
  lastErrReplyMs = now;
  replyErr(seq, code);
}

// ---- strict number parsing ----
bool parseU32(const char* s, uint32_t& out) {
  if (!s || !*s) return false;
  size_t n = strlen(s);
  if (n > 10) return false;
  for (size_t i = 0; i < n; i++) if (!isdigit((unsigned char)s[i])) return false;
  unsigned long long v = strtoull(s, nullptr, 10);
  if (v > 0xFFFFFFFFULL) return false;
  out = (uint32_t)v;
  return true;
}
bool parseI32(const char* s, long& out) {
  if (!s || !*s) return false;
  const char* p = s;
  if (*p == '-') p++;
  if (!*p || strlen(p) > 9) return false;
  for (const char* q = p; *q; q++) if (!isdigit((unsigned char)*q)) return false;
  out = strtol(s, nullptr, 10);
  return true;
}
bool parseFloatStrict(const char* s, float& out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  float v = strtof(s, &end);
  if (end == s || *end != '\0' || !isfinite(v)) return false;
  out = v;
  return true;
}

// ============================ PCA9685 OUTPUT ============================
bool i2cPing() {
  Wire.beginTransmission(PCA_ADDR);
  return Wire.endTransmission() == 0;
}

void writeUsRaw(uint8_t ch, float us) {
  float t = us * PWM_FREQ_HZ * 4096.0f / 1000000.0f;   // 4096 counts per period
  long ticks = lroundf(t);
  if (ticks < 0) ticks = 0;
  if (ticks > 4095) ticks = 4095;
  pca.setPWM(SERVO[ch].pcaChannel, 0, (uint16_t)ticks);
}

// Lowest-layer clamp: no code path can command outside the configured angle window.
void writeAngle(uint8_t ch, float a, bool force = false) {
  const ServoCfg& s = SERVO[ch];
  a = clampf(a, s.angMin, s.angMax);
  if (!force && fabsf(a - lastWritten[ch]) < 0.01f) return;
  writeUsRaw(ch, usFromAngle(ch, a));
  lastWritten[ch] = a;
}
void writeAll(bool force) {
  for (uint8_t ch = 0; ch < NUM_CH; ch++) writeAngle(ch, cur[ch], force);
}

void outputsEnable(bool en) {
  digitalWrite(PIN_OE, en ? LOW : HIGH);
  outputsEnabled = en;
}

bool configValid() {
  for (uint8_t ch = 0; ch < NUM_CH; ch++) {
    const ServoCfg& s = SERVO[ch];
    if (!(s.angMin < s.angMax && s.home >= s.angMin && s.home <= s.angMax)) return false;
    if (!(s.usMin < s.usMax && s.calUsMin < s.calUsMax && s.slewDps > 0.0f)) return false;
    float uLo = usFromAngle(ch, s.angMin), uHi = usFromAngle(ch, s.angMax);
    if (uLo < s.calUsMin || uHi > s.calUsMax) return false;
  }
  if (ACT_ACTIVE_ANGLE < SERVO[CH_ACT].angMin || ACT_ACTIVE_ANGLE > SERVO[CH_ACT].angMax) return false;
  return true;
}

// ============================ STABILITY (ESP32-side) ============================
void resetStability() { stabRunning = false; }

uint32_t stableElapsed(uint32_t now) { return stabRunning ? (now - stabStartMs) : 0; }

void onTargetFrame(bool det, int cx, int cy, uint32_t hostStable, uint32_t now) {
  lastTgtMs = now;
  tgtDet = det;
  tgtCx = cx;
  tgtCy = cy;
  hostStableMs = hostStable;   // informational only, never trusted
  if (!det) return;            // a missing detection never counts as stable
  bool centered = abs(cx - FRAME_W / 2) <= CENTER_TOL_PX && abs(cy - FRAME_H / 2) <= CENTER_TOL_PX;
  if (!centered) { resetStability(); return; }
  if (!stabRunning) {
    stabRunning = true; stabStartMs = now; anchorX = cx; anchorY = cy;
  } else if (abs(cx - anchorX) > STAB_TOL_PX || abs(cy - anchorY) > STAB_TOL_PX) {
    stabStartMs = now; anchorX = cx; anchorY = cy;   // moved: restart timing at the new position
  }
  lastGoodMs = now;
}

bool headSettled() {
  return fabsf(cur[CH_PAN] - tgt[CH_PAN]) <= SETTLE_DEG && fabsf(cur[CH_TILT] - tgt[CH_TILT]) <= SETTLE_DEG;
}

// Single source of truth for "may we fire". Evaluated continuously AND again at ACT time.
bool interlocksOK(uint32_t now) {
  if (estopActive()) return false;
  if (calMode) return false;
  if (!pcaOk || !outputsEnabled) return false;
  if (!hostFresh(now)) return false;
  if (!stabRunning) return false;
  if (now - lastGoodMs > GRACE_MS) return false;
  if (stableElapsed(now) < stableRequiredMs) return false;
  if (!headSettled()) return false;
  return true;
}

// ============================ STATE TRANSITIONS ============================
void enterEstop(const char* why) {
  outputsEnable(false);
  state = ST_ESTOP;
  faultReason = why;
  calMode = false;
  readyChallenge = 0;
  actPhase = 0;
  resetStability();
  cur[CH_ACT] = tgt[CH_ACT] = SERVO[CH_ACT].home;   // registers hold REST for when outputs are re-enabled
  if (pcaOk) writeAngle(CH_ACT, cur[CH_ACT], true);
}

void enterFault(const char* why) {
  outputsEnable(false);
  state = ST_FAULT;
  faultReason = why;
  calMode = false;
  readyChallenge = 0;
  actPhase = 0;
  resetStability();
  cur[CH_ACT] = tgt[CH_ACT] = SERVO[CH_ACT].home;
}

void safeDisarm(const char* why) {
  state = ST_DISARMED;
  faultReason = why;
  readyChallenge = 0;
  actPhase = 0;
  lossActive = false;
  tgt[CH_ACT] = SERVO[CH_ACT].home;   // slews back to REST
  resetStability();
}

uint32_t makeChallenge() { return (esp_random() % 90000UL) + 10000UL; }

// ============================ COMMAND HANDLING ============================
void sendStatus() {
  uint32_t now = millis();
  uint32_t hbAge = hostSeen ? (now - lastHostMs) : 99999;
  char b[176];
  snprintf(b, sizeof(b), "STAT,%s,%.1f,%.1f,%.1f,%d,%d,%lu,%lu,%lu,%lu,%lu,%d,%s",
           stateName(state), cur[CH_PAN], cur[CH_TILT], cur[CH_ACT],
           estopActive() ? 1 : 0, isArmedState() ? 1 : 0,
           (unsigned long)hbAge, (unsigned long)stableElapsed(now), (unsigned long)stableRequiredMs,
           (unsigned long)readyChallenge, (unsigned long)actCount, pcaOk ? 1 : 0, faultReason);
  sendFrame(b);
}

void dispatch(char** f, int nf) {
  const char* cmd = f[0];
  uint32_t seq = 0;
  if (nf < 2 || !parseU32(f[1], seq)) { replyErrLimited(0, "FMT"); return; }
  uint32_t now = millis();

  // HELLO resynchronises the sequence counter and forces a safe state. It never clears ESTOP/FAULT.
  if (strcmp(cmd, "HELLO") == 0) {
    if (nf != 2) { replyErr(seq, "FMT"); return; }
    lastSeq = seq; hostSeen = true; lastHostMs = now;
    if (isArmedState()) safeDisarm("HELLO");
    replyAck(seq);
    return;
  }

  if (seq <= lastSeq) { replyErr(seq, "SEQ"); return; }
  lastSeq = seq; hostSeen = true; lastHostMs = now;

  // ---------- STOP: highest priority, valid in every state ----------
  if (strcmp(cmd, "STOP") == 0) {
    if (nf != 2) { replyErr(seq, "FMT"); return; }
    enterEstop("SW_STOP");
    replyAck(seq);
    return;
  }

  if (strcmp(cmd, "HB") == 0) {
    if (nf != 2) { replyErr(seq, "FMT"); return; }
    replyAck(seq);

  } else if (strcmp(cmd, "STATUS") == 0) {
    if (nf != 2) { replyErr(seq, "FMT"); return; }
    sendStatus();
    replyAck(seq);

  } else if (strcmp(cmd, "SET") == 0) {
    float pan, tilt;
    if (nf != 4 || !parseFloatStrict(f[2], pan) || !parseFloatStrict(f[3], tilt)) { replyErr(seq, "FMT"); return; }
    if (pan < SERVO[CH_PAN].angMin || pan > SERVO[CH_PAN].angMax ||
        tilt < SERVO[CH_TILT].angMin || tilt > SERVO[CH_TILT].angMax) { replyErr(seq, "RANGE"); return; }
    if (estopActive()) { replyErr(seq, "ESTOP"); return; }
    if (state == ST_FAULT) { replyErr(seq, "FAULT"); return; }
    if (state == ST_ACTIVATING || calMode) { replyErr(seq, "STATE"); return; }
    tgt[CH_PAN] = pan;
    tgt[CH_TILT] = tilt;
    replyAck(seq);

  } else if (strcmp(cmd, "TGT") == 0) {
    uint32_t det, hs;
    long cx, cy;
    if (nf != 6 || !parseU32(f[2], det) || det > 1 || !parseI32(f[3], cx) || !parseI32(f[4], cy) ||
        !parseU32(f[5], hs)) { replyErr(seq, "FMT"); return; }
    if (cx < 0 || cx > FRAME_W || cy < 0 || cy > FRAME_H || hs > 600000UL) { replyErr(seq, "RANGE"); return; }
    onTargetFrame(det == 1, (int)cx, (int)cy, hs, now);
    replyAck(seq);

  } else if (strcmp(cmd, "ARM") == 0) {
    if (nf != 2) { replyErr(seq, "FMT"); return; }
    if (estopActive()) { replyErr(seq, "ESTOP"); return; }
    if (state == ST_FAULT) { replyErr(seq, "FAULT"); return; }
    if (state != ST_DISARMED || calMode) { replyErr(seq, "STATE"); return; }
    resetStability();          // timing starts only AFTER arming
    readyChallenge = 0;
    lossActive = false;
    faultReason = "NONE";
    state = ST_TRACKING;
    replyAck(seq);

  } else if (strcmp(cmd, "DISARM") == 0) {
    if (nf != 2) { replyErr(seq, "FMT"); return; }
    if (state == ST_ESTOP || state == ST_FAULT) { replyAck(seq); return; }   // already safe
    safeDisarm("NONE");
    replyAck(seq);

  } else if (strcmp(cmd, "ACT") == 0) {
    uint32_t chal;
    if (nf != 3 || !parseU32(f[2], chal)) { replyErr(seq, "FMT"); return; }
    if (estopActive()) { replyErr(seq, "ESTOP"); return; }
    if (state == ST_FAULT) { replyErr(seq, "FAULT"); return; }
    if (state == ST_DISARMED) { replyErr(seq, "DISARMED"); return; }
    if (state == ST_LOCKOUT) { replyErr(seq, "LOCKOUT"); return; }
    if (state == ST_ACTIVATING) { replyErr(seq, "BUSY"); return; }
    if (state != ST_READY || readyChallenge == 0) { replyErr(seq, "NOTREADY"); return; }
    if (chal != readyChallenge) { replyErr(seq, "CHAL"); return; }
    if (!interlocksOK(now)) { state = ST_TRACKING; readyChallenge = 0; replyErr(seq, "NOTREADY"); return; }
    state = ST_ACTIVATING;
    readyChallenge = 0;
    actPhase = 0;
    actPhaseStart = now;
    actCount++;
    tgt[CH_ACT] = ACT_ACTIVE_ANGLE;
    replyAck(seq);

  } else if (strcmp(cmd, "RESET") == 0) {
    if (nf != 2) { replyErr(seq, "FMT"); return; }
    if (state != ST_ESTOP && state != ST_FAULT) { replyErr(seq, "STATE"); return; }
    if (estopHw) { replyErr(seq, "ESTOP"); return; }
    if (!i2cPing()) { replyErr(seq, "I2C"); return; }
    pcaOk = true; pcaFail = 0;
    cur[CH_ACT] = tgt[CH_ACT] = SERVO[CH_ACT].home;
    writeAll(true);
    outputsEnable(true);
    resetStability();
    readyChallenge = 0;
    lossActive = false;
    faultReason = "NONE";
    state = ST_DISARMED;
    replyAck(seq);

  } else if (strcmp(cmd, "CFG") == 0) {
    uint32_t ms;
    if (nf != 3 || !parseU32(f[2], ms)) { replyErr(seq, "FMT"); return; }
    if (ms < STABLE_MIN_MS || ms > STABLE_MAX_MS) { replyErr(seq, "RANGE"); return; }
    if (state != ST_DISARMED) { replyErr(seq, "STATE"); return; }
    stableRequiredMs = ms;
    replyAck(seq);

  } else if (strcmp(cmd, "CALMODE") == 0) {
    uint32_t on;
    if (nf != 3 || !parseU32(f[2], on) || on > 1) { replyErr(seq, "FMT"); return; }
    if (on == 1) {
      if (estopActive()) { replyErr(seq, "ESTOP"); return; }
      if (state != ST_DISARMED) { replyErr(seq, "STATE"); return; }
      for (uint8_t ch = 0; ch < NUM_CH; ch++) calUs[ch] = usFromAngle(ch, cur[ch]);
      calMode = true;
    } else if (calMode) {
      for (uint8_t ch = 0; ch < NUM_CH; ch++) {
        const ServoCfg& s = SERVO[ch];
        cur[ch] = clampf(angleFromUs(ch, calUs[ch]), s.angMin, s.angMax);
        tgt[ch] = s.home;                // slews gently back to home
        lastWritten[ch] = NAN;
      }
      calMode = false;
    }
    replyAck(seq);

  } else if (strcmp(cmd, "CAL") == 0) {
    long ch;
    float us;
    if (nf != 4 || !parseI32(f[2], ch) || !parseFloatStrict(f[3], us)) { replyErr(seq, "FMT"); return; }
    if (!calMode) { replyErr(seq, "STATE"); return; }
    if (estopActive()) { replyErr(seq, "ESTOP"); return; }
    if (ch < 0 || ch >= NUM_CH) { replyErr(seq, "RANGE"); return; }
    if (us < SERVO[ch].calUsMin || us > SERVO[ch].calUsMax) { replyErr(seq, "RANGE"); return; }
    writeUsRaw((uint8_t)ch, us);
    calUs[ch] = us;
    replyAck(seq);

  } else {
    replyErr(seq, "UNKNOWN");
  }
}

int hexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

// buf: text between '$' and '\n', NUL-terminated
void handleLine(char* buf) {
  size_t n = strlen(buf);
  char* star = strrchr(buf, '*');
  if (!star || (size_t)((buf + n) - star) != 3) { replyErrLimited(0, "FMT"); return; }
  int h1 = hexVal(star[1]), h2 = hexVal(star[2]);
  if (h1 < 0 || h2 < 0) { replyErrLimited(0, "CHK"); return; }
  uint8_t rx = (uint8_t)((h1 << 4) | h2);
  uint8_t x = 0;
  for (char* p = buf; p < star; p++) x ^= (uint8_t)*p;
  if (x != rx) { replyErrLimited(0, "CHK"); return; }
  *star = '\0';
  if (star == buf) { replyErrLimited(0, "FMT"); return; }

  char* f[MAX_FIELDS];
  int nf = 0;
  f[nf++] = buf;
  for (char* p = buf; *p; p++) {
    if (*p == ',') {
      *p = '\0';
      if (nf >= MAX_FIELDS) { replyErrLimited(0, "FMT"); return; }
      f[nf++] = p + 1;
    }
  }
  dispatch(f, nf);
}

// Non-blocking serial reader. '$' always restarts a frame, so partial/corrupt frames are dropped.
void pollSerial() {
  static char rxBuf[RX_MAX];
  static size_t rxLen = 0;
  static bool rxActive = false;
  int budget = 256;
  while (budget-- > 0 && Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '$') { rxLen = 0; rxActive = true; continue; }
    if (!rxActive) continue;
    if (c == '\n') {
      rxBuf[rxLen] = '\0';
      rxActive = false;
      if (rxLen > 0) handleLine(rxBuf);
      rxLen = 0;
      continue;
    }
    if (c == '\r') continue;
    if (rxLen >= RX_MAX - 1) { rxActive = false; rxLen = 0; replyErrLimited(0, "LEN"); continue; }
    rxBuf[rxLen++] = c;
  }
}

// ============================ PERIODIC TASKS ============================
void pollEstop(uint32_t now) {
  bool raw = (digitalRead(PIN_ESTOP_SENSE) == HIGH);   // HIGH = pressed or wire broken (fail-safe)
  if (raw) {
    if (!estopRawHigh) { estopRawHigh = true; estopRawSince = now; }
    if (!estopHw && (now - estopRawSince) >= ESTOP_ASSERT_MS) estopHw = true;
  } else {
    if (estopRawHigh) { estopRawHigh = false; estopRawSince = now; }
    if (estopHw && (now - estopRawSince) >= ESTOP_RELEASE_MS) estopHw = false;
  }
  if (estopHw && state != ST_ESTOP) enterEstop("HW_ESTOP");
}

void watchdog(uint32_t now) {
  if ((state == ST_TRACKING || state == ST_READY || state == ST_LOCKOUT) && !hostFresh(now)) safeDisarm("HB");
}

void checkI2C(uint32_t now) {
  static uint32_t last = 0;
  if (state == ST_BOOT || state == ST_ESTOP || state == ST_FAULT) return;
  if (now - last < I2C_CHECK_MS) return;
  last = now;
  if (i2cPing()) { pcaFail = 0; pcaOk = true; }
  else if (++pcaFail >= 3) { pcaOk = false; enterFault("I2C"); }
}

void updateStateMachine(uint32_t now) {
  // brief detection gaps beyond the grace period invalidate the timer
  if (stabRunning && (now - lastGoodMs) > GRACE_MS) resetStability();

  switch (state) {
    case ST_TRACKING:
      if (interlocksOK(now)) { state = ST_READY; readyChallenge = makeChallenge(); }
      break;
    case ST_READY:
      if (!interlocksOK(now)) { state = ST_TRACKING; readyChallenge = 0; }
      break;
    case ST_ACTIVATING:
      if (actPhase == 0 && (now - actPhaseStart) >= ACT_HOLD_MS) {
        actPhase = 1; actPhaseStart = now; tgt[CH_ACT] = SERVO[CH_ACT].home;
      } else if (actPhase == 1 && (now - actPhaseStart) >= ACT_RETURN_MS) {
        actPhase = 0; state = ST_LOCKOUT; lossActive = false; resetStability();
      }
      break;
    case ST_LOCKOUT: {
      bool visible = tgtDet && (now - lastTgtMs) <= TGT_STALE_MS;
      if (visible) lossActive = false;
      else if (!lossActive) { lossActive = true; lossStartMs = now; }
      else if ((now - lossStartMs) >= REARM_LOSS_MS) {
        lossActive = false; state = ST_TRACKING; resetStability();
      }
      break;
    }
    default: break;
  }
}

void updateServos(uint32_t now) {
  static uint32_t last = 0;
  if (now - last < SERVO_UPDATE_MS) return;
  float dt = (now - last) / 1000.0f;
  last = now;
  if (dt > 0.1f) dt = 0.1f;
  if (calMode || state == ST_BOOT || state == ST_ESTOP || state == ST_FAULT) return;
  for (uint8_t ch = 0; ch < NUM_CH; ch++) {
    float d = tgt[ch] - cur[ch];
    float m = SERVO[ch].slewDps * dt;
    if (fabsf(d) <= m) cur[ch] = tgt[ch];
    else cur[ch] += copysignf(m, d);
    writeAngle(ch, cur[ch]);
  }
}

void telemetry(uint32_t now) {
  static uint32_t last = 0;
  if (now - last < STAT_PERIOD_MS) return;
  last = now;
  sendStatus();
}

// ============================ SETUP / LOOP ============================
void setup() {
  // Outputs OFF first, before anything else.
  pinMode(PIN_OE, OUTPUT);
  digitalWrite(PIN_OE, HIGH);
  pinMode(PIN_ESTOP_SENSE, INPUT_PULLUP);

  Serial.begin(SERIAL_BAUD);
  delay(100);

  for (uint8_t ch = 0; ch < NUM_CH; ch++) {
    cur[ch] = tgt[ch] = SERVO[ch].home;
    lastWritten[ch] = NAN;
    calUs[ch] = usFromAngle(ch, SERVO[ch].home);
  }

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(I2C_HZ);
  Wire.setTimeOut(50);

  uint32_t now = millis();
  estopRawHigh = (digitalRead(PIN_ESTOP_SENSE) == HIGH);
  estopRawSince = now;
  estopHw = estopRawHigh;

  if (!configValid()) {
    enterFault("CFG");
  } else if (!i2cPing()) {
    enterFault("I2C");
  } else {
    pca.begin();                       // may log a harmless "bus already started" message
    pca.setOscillatorFrequency(OSC_HZ);
    pca.setPWMFreq(PWM_FREQ_HZ);
    delay(10);
    pcaOk = true;
    writeAll(true);                    // registers hold HOME while outputs are still disabled
    if (estopHw) {
      enterEstop("HW_ESTOP");
    } else {
      outputsEnable(true);
      state = ST_DISARMED;
    }
  }

  char b[64];
  snprintf(b, sizeof(b), "BOOT,esp32_servo_controller,%s", FW_VERSION);
  sendFrame(b);
}

void loop() {
  uint32_t now = millis();
  pollEstop(now);          // e-stop first, every iteration
  pollSerial();
  now = millis();
  pollEstop(now);
  watchdog(now);
  updateStateMachine(now);
  checkI2C(now);
  updateServos(now);
  telemetry(now);
}