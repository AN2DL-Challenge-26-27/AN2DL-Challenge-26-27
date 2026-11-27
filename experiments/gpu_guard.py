"""GPU temperature guard: pause training when the GPU gets too hot, resume when it has cooled.

Run it next to a training queue (from the repository root):

    python experiments/gpu_guard.py                     # pause at 82 C, resume at 72 C, >= 45 s pauses
    python experiments/gpu_guard.py --pause-at 80 --resume-at 70 --min-pause 60

Every few seconds it reads the GPU temperature with ``nvidia-smi``. At or above ``--pause-at``
it suspends every running ``python -m spaghetti1`` process and its DataLoader workers. They
continue exactly where they stopped once the temperature is at or below ``--resume-at`` and at
least ``--min-pause`` seconds have passed (a laptop GPU core cools within seconds, but the cooler
needs longer; without a minimum pause the GPU would bounce around the limit). Pausing
only makes the run longer; results are unchanged. The guard stops by itself when no
``spaghetti1`` process has been seen for ``--idle-exit`` seconds, and always resumes what it
paused before exiting. Events are printed and appended to ``outputs/runs/logs/gpu_guard.log``.

If the guard is killed during a pause (e.g. its terminal is closed), the training stays frozen;
``python experiments/gpu_guard.py --resume-all`` resumes every spaghetti1 process.
"""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time

import psutil

LOG = Path(__file__).resolve().parent.parent / "outputs" / "runs" / "logs" / "gpu_guard.log"


def gpu_temperature() -> int | None:
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        return int(output.splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def training_processes() -> list[psutil.Process]:
    """Every ``python ... spaghetti1`` process and its child processes (DataLoader workers)."""
    found = {}
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if "python" in (process.info["name"] or "").lower() and "spaghetti1" in command and "gpu_guard" not in command:
                found[process.pid] = process
                for child in process.children(recursive=True):
                    found[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return list(found.values())


def log(message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def set_paused(processes: list[psutil.Process], paused: bool) -> int:
    changed = 0
    for process in processes:
        try:
            process.suspend() if paused else process.resume()
            changed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pause-at", type=int, default=82, help="pause at or above this temperature (C)")
    parser.add_argument("--resume-at", type=int, default=72, help="resume at or below this temperature (C)")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between checks")
    parser.add_argument("--min-pause", type=float, default=45.0, help="shortest pause in seconds")
    parser.add_argument("--idle-exit", type=float, default=300.0, help="exit after this many seconds without training")
    parser.add_argument("--resume-all", action="store_true", help="resume frozen training processes and exit")
    args = parser.parse_args()
    if args.resume_all:
        log(f"manual resume: resumed {set_paused(training_processes(), False)} processes")
        return
    if args.resume_at >= args.pause_at:
        parser.error("--resume-at must be lower than --pause-at")

    log(f"guard started: pause at {args.pause_at} C, resume at {args.resume_at} C, pauses >= {args.min_pause:.0f} s")
    paused: list[psutil.Process] = []
    paused_since, last_seen, hottest, pauses, last_report = None, time.time(), 0, 0, 0.0
    try:
        while True:
            temperature = gpu_temperature()
            processes = training_processes()
            if processes:
                last_seen = time.time()
            if temperature is not None:
                hottest = max(hottest, temperature)
            if temperature is not None and not paused and processes and temperature >= args.pause_at:
                paused = processes
                pauses += 1
                paused_since = time.time()
                log(f"PAUSE  GPU {temperature} C >= {args.pause_at} C: suspended {set_paused(paused, True)} processes")
            elif paused and (temperature is None or temperature <= args.resume_at)                     and time.time() - paused_since >= args.min_pause:
                seconds = time.time() - paused_since
                log(f"RESUME GPU {temperature} C <= {args.resume_at} C after {seconds:.0f} s: "
                    f"resumed {set_paused(paused, False)} processes")
                paused, paused_since = [], None
            if time.time() - last_report >= 300:  # status line every 5 minutes
                log(f"status GPU {temperature} C | training processes {len(processes)} | "
                    f"{'PAUSED' if paused else 'running'} | pauses so far {pauses} | hottest {hottest} C")
                last_report = time.time()
            if not processes and not paused and time.time() - last_seen > args.idle_exit:
                log(f"no training for {args.idle_exit:.0f} s: guard stopped (pauses {pauses}, hottest {hottest} C)")
                return
            time.sleep(args.interval)
    finally:
        if paused:
            log(f"guard exiting: resumed {set_paused(paused, False)} paused processes")


if __name__ == "__main__":
    main()
