#!/usr/bin/env python3
"""
V.S.P.M. - Vibration Sensing and Predictive Maintenance
Local training node: serial acquisition, feature extraction, model training.

Usage:
    python vspm_node.py collect --port COM7
    python vspm_node.py train

Install dependencies:
    pip install pyserial numpy scipy pandas scikit-learn joblib

Workflow (collect mode)
    1. The script asks for ONE label (0, 1, 2 ...) for the next burst.
    2. You press the trigger button on the ESP32.
    3. The ESP32 sends 100 batches; each batch is turned into 12 features and
       the script shows live progress.
    4. When the ESP32 reports the run is complete, all 100 rows are saved to
       the CSV with that label, and the script asks for the next label.

Expected serial framing from the ESP32 (per button press):
    ---BATCH START---
    x,y,z                (800 lines, raw ADXL345 counts)
    ---BATCH END---
    ... repeated 100 times, then:
    ---RUN COMPLETE---   (or ---RUN ABORTED--- if the sensor failed mid-run)
"""

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import serial
from scipy.stats import kurtosis

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_BAUD = 115200
SAMPLES_PER_BATCH = 800
BATCHES_PER_RUN = 100                       # must match BATCHES_PER_RUN in the firmware

# Must match the strings in the firmware exactly.
START_MARKER = "---BATCH START---"
END_MARKER = "---BATCH END---"
RUN_COMPLETE_MARKER = "---RUN COMPLETE---"
RUN_ABORT_MARKER = "---RUN ABORTED---"

CSV_PATH = Path("vspm_training_features.csv")
MODEL_PATH = Path("vspm_rf_model.joblib")

AXES = ("x", "y", "z")
FEATURE_NAMES = ("rms", "p2p", "crest", "kurt")
FEATURE_COLUMNS = [f"{axis}_{feat}" for axis in AXES for feat in FEATURE_NAMES]
CSV_COLUMNS = FEATURE_COLUMNS + ["label"]

BATCH_TIMEOUT_S = 10.0      # max time to receive one full batch once START is seen
RUN_IDLE_TIMEOUT_S = 15.0   # max silence mid-run (one batch normally takes ~1 s)
RECONNECT_DELAY_S = 3.0     # wait between reconnect attempts
BOOT_DELAY_S = 2.0          # ESP32 usually resets when the port opens (DTR)

log = logging.getLogger("vspm")


# --------------------------------------------------------------------------- #
# Serial acquisition
# --------------------------------------------------------------------------- #
def open_serial(port: str, baud: int) -> serial.Serial:
    """Open the serial port and give the ESP32 time to reboot."""
    ser = serial.Serial(port, baud, timeout=1.0)
    time.sleep(BOOT_DELAY_S)
    ser.reset_input_buffer()
    log.info("Connected to %s @ %d baud", port, baud)
    return ser


def _read_line(ser: serial.Serial) -> Optional[str]:
    """Read one line, tolerating corrupted bytes. Returns None on read timeout."""
    raw = ser.readline()
    if not raw:
        return None
    return raw.decode("ascii", errors="ignore").strip()


def _parse_sample(line: str) -> Optional[tuple]:
    """Parse 'x,y,z' into floats. Returns None if the line is malformed."""
    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_axis_features(signal: np.ndarray, remove_dc: bool = False) -> List[float]:
    """RMS, peak-to-peak, crest factor, kurtosis for a single axis."""
    if remove_dc:
        signal = signal - signal.mean()

    rms = float(np.sqrt(np.mean(signal ** 2)))
    p2p = float(np.ptp(signal))
    peak = float(np.max(np.abs(signal)))
    crest = peak / rms if rms > 1e-12 else 0.0
    kurt = float(kurtosis(signal, fisher=True))   # excess kurtosis (normal = 0)
    if not np.isfinite(kurt):                      # constant signal -> NaN
        kurt = 0.0
    return [rms, p2p, crest, kurt]


def extract_features(batch: np.ndarray, remove_dc: bool = False) -> List[float]:
    """12 features: [x_rms, x_p2p, x_crest, x_kurt, y_..., z_...]."""
    features: List[float] = []
    for i in range(3):
        features.extend(extract_axis_features(batch[:, i], remove_dc))
    return features


# --------------------------------------------------------------------------- #
# Run reception (100 batches per button press)
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """Outcome of one button-press run."""
    status: str                                   # "complete" | "aborted" | "timeout"
    features: List[List[float]] = field(default_factory=list)   # one row per good batch
    bad_batches: int = 0                          # batches discarded as corrupt/incomplete


def read_run(ser: serial.Serial, expected_batches: int, remove_dc: bool) -> RunResult:
    """
    Block until the ESP32 finishes (or aborts) one 100-batch run.

    Waits indefinitely for the first ---BATCH START--- (the user has to press the
    button), then processes every batch as it arrives so the serial buffer never
    backs up. Raises serial.SerialException if the port disconnects.
    """
    result = RunResult(status="timeout")
    samples: List[tuple] = []
    in_batch = False
    bad_lines = 0
    batch_started = 0.0
    last_activity: Optional[float] = None    # None until the run has actually begun

    def finish_batch() -> None:
        """Validate the batch that just ended and turn it into a feature row."""
        if len(samples) != SAMPLES_PER_BATCH:
            result.bad_batches += 1
            print()
            log.warning("Batch %d: got %d/%d valid samples (%d malformed lines). Discarded.",
                        len(result.features) + result.bad_batches, len(samples),
                        SAMPLES_PER_BATCH, bad_lines)
            return
        result.features.append(extract_features(np.asarray(samples, dtype=np.float64), remove_dc))
        print(f"\r  Receiving: batch {len(result.features)}/{expected_batches}", end="", flush=True)

    while True:
        line = _read_line(ser)
        now = time.monotonic()

        # ---- read timeout: check the watchdogs, keep listening ----
        if line is None:
            if last_activity is not None and now - last_activity > RUN_IDLE_TIMEOUT_S:
                print()
                log.warning("No data for %.0fs mid-run; giving up on this run.", RUN_IDLE_TIMEOUT_S)
                result.status = "timeout"
                return result
            continue

        # ---- one batch taking too long ----
        if in_batch and now - batch_started > BATCH_TIMEOUT_S:
            print()
            log.warning("Batch timed out after %d samples; discarding.", len(samples))
            result.bad_batches += 1
            in_batch = False

        # ---- framing markers ----
        if line == START_MARKER:
            if in_batch:                      # previous END was lost
                print()
                log.warning("Unexpected START mid-batch; previous batch discarded.")
                result.bad_batches += 1
            in_batch = True
            samples = []
            bad_lines = 0
            batch_started = now
            last_activity = now

        elif line == END_MARKER:
            if in_batch:
                in_batch = False
                last_activity = now
                finish_batch()
            # END without START: stray marker, ignore

        elif line == RUN_COMPLETE_MARKER:
            if in_batch:                      # END was lost for the final batch
                result.bad_batches += 1
            print()
            result.status = "complete"
            return result

        elif line == RUN_ABORT_MARKER:
            print()
            result.status = "aborted"
            return result

        # ---- sample data ----
        elif in_batch:
            sample = _parse_sample(line)
            if sample is None:
                bad_lines += 1
            else:
                samples.append(sample)

        # ---- anything else outside a batch (boot text, firmware errors) ----
        else:
            if line:
                log.info("ESP32: %s", line)


# --------------------------------------------------------------------------- #
# CSV handling
# --------------------------------------------------------------------------- #
def append_rows_to_csv(path: Path, rows: List[List[float]], label: int) -> None:
    """Append many feature rows (one open/close) with the same label."""
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(CSV_COLUMNS)
        for features in rows:
            writer.writerow([f"{v:.6f}" for v in features] + [label])


# --------------------------------------------------------------------------- #
# Mode 1: collection
# --------------------------------------------------------------------------- #
def prompt_label(valid_labels: set) -> Optional[int]:
    """Ask for ONE label for the next run. Returns int label, or None to quit."""
    options = "/".join(str(l) for l in sorted(valid_labels))
    while True:
        answer = input(f"\nLabel for the next {BATCHES_PER_RUN}-batch run [{options}], 'q' = quit: ").strip().lower()
        if answer == "q":
            return None
        try:
            label = int(answer)
        except ValueError:
            print("  Invalid input.")
            continue
        if label in valid_labels:
            return label
        print(f"  Label must be one of {sorted(valid_labels)}.")


def run_collection(port: str, baud: int, num_classes: int, remove_dc: bool, batches: int) -> None:
    valid_labels = set(range(num_classes))
    saved_rows = 0
    saved_runs = 0
    ser: Optional[serial.Serial] = None
    label: Optional[int] = None

    print(f"Collecting from {port}. Features append to '{CSV_PATH}'. Ctrl+C to stop.")

    try:
        while True:
            # (Re)connect as needed
            if ser is None or not ser.is_open:
                try:
                    ser = open_serial(port, baud)
                except (serial.SerialException, OSError) as exc:
                    log.error("Cannot open %s: %s. Retrying in %.0fs...", port, exc, RECONNECT_DELAY_S)
                    time.sleep(RECONNECT_DELAY_S)
                    continue

            try:
                # Ask the label ONCE per run, BEFORE the button is pressed.
                if label is None:
                    label = prompt_label(valid_labels)
                    if label is None:
                        break
                    ser.reset_input_buffer()          # drop anything left over from before
                    print(f"Label {label} set. Press the trigger button on the ESP32 "
                          f"(screen should say 'Ready')...")

                result = read_run(ser, batches, remove_dc)

            except (serial.SerialException, OSError) as exc:
                log.error("Serial connection lost: %s. Reconnecting...", exc)
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(RECONNECT_DELAY_S)
                continue          # keeps the chosen label; waits for a fresh button press

            # ---- run finished: decide what to do with it ----
            if result.status == "complete":
                good = len(result.features)
                if good:
                    append_rows_to_csv(CSV_PATH, result.features, label)
                    saved_rows += good
                    saved_runs += 1
                print(f"Run complete: {good}/{batches} good batches saved with label {label}"
                      + (f" ({result.bad_batches} discarded)" if result.bad_batches else "")
                      + f". Session total: {saved_rows} rows from {saved_runs} run(s).")
                if good != batches:
                    log.warning("Expected %d batches but saved %d.", batches, good)
            elif result.status == "aborted":
                log.error("Run aborted by the ESP32 (sensor fault) after %d good batches. "
                          "Nothing saved - check the ADXL345 wiring.", len(result.features))
            else:
                log.error("Run timed out after %d good batches. Nothing saved.", len(result.features))

            label = None   # next run asks for its own label

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if ser is not None and ser.is_open:
            ser.close()
        print(f"Done. {saved_rows} row(s) from {saved_runs} run(s) saved to '{CSV_PATH}'.")


# --------------------------------------------------------------------------- #
# Mode 2: training
# --------------------------------------------------------------------------- #
def run_training(csv_path: Path, model_path: Path, n_estimators: int) -> None:
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        log.error("'%s' not found. Run 'collect' mode first.", csv_path)
        sys.exit(1)
    except Exception as exc:
        log.error("Could not read '%s': %s", csv_path, exc)
        sys.exit(1)

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        log.error("CSV is missing columns: %s", missing)
        sys.exit(1)

    df = df.dropna(subset=CSV_COLUMNS)
    if len(df) < 10:
        log.error("Only %d rows available - collect more data before training.", len(df))
        sys.exit(1)

    X = df[FEATURE_COLUMNS].values
    y = df["label"].astype(int).values

    class_counts = pd.Series(y).value_counts().sort_index()
    print("Samples per class:")
    print(class_counts.to_string(), "\n")

    # Stratify only when every class has enough samples to appear in both splits
    stratify = y if class_counts.min() >= 2 and len(df) * 0.2 >= len(class_counts) else None
    if stratify is None:
        log.warning("Too few samples per class for a stratified split; using a plain random split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, y_pred), "\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Feature importances:")
    ranked = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda t: -t[1])
    for name, imp in ranked:
        print(f"  {name:<10} {imp:.4f}")

    # Bundle feature order with the model so live inference can't mismatch it
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, model_path)
    print(f"\nModel saved to '{model_path}'.")
    print("Load later with: bundle = joblib.load(path); bundle['model'].predict(features_2d)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V.S.P.M. local training node")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_collect = sub.add_parser("collect", help="Acquire 100-batch runs and label them")
    p_collect.add_argument("--port", required=True, help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    p_collect.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p_collect.add_argument("--classes", type=int, default=3, help="Number of classes (labels 0..N-1)")
    p_collect.add_argument("--batches", type=int, default=BATCHES_PER_RUN,
                           help="Batches per button press (must match the firmware)")
    p_collect.add_argument("--remove-dc", action="store_true",
                           help="Subtract per-axis mean (removes gravity offset) before computing features")

    p_train = sub.add_parser("train", help="Train a RandomForest on the collected features")
    p_train.add_argument("--csv", type=Path, default=CSV_PATH)
    p_train.add_argument("--model", type=Path, default=MODEL_PATH)
    p_train.add_argument("--trees", type=int, default=200)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    args = build_parser().parse_args()

    if args.mode == "collect":
        run_collection(args.port, args.baud, args.classes, args.remove_dc, args.batches)
    elif args.mode == "train":
        run_training(args.csv, args.model, args.trees)


if __name__ == "__main__":
    main()
