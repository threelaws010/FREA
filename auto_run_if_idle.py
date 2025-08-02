import os
import time
import subprocess
from dotenv import load_dotenv


# Requires `pyxhook` on Linux or `pynput` on Windows/macOS
import ctypes
import platform

load_dotenv()

CHECK_INTERVAL = 15 # seconds
IDLE_THRESHOLD = 30# 5 minutes

def is_idle():
    try:
        idle_ms = int(subprocess.check_output(['xprintidle']).decode().strip())
        idle_sec = idle_ms / 1000
        return idle_sec > IDLE_THRESHOLD
    except Exception as e:
        print(f"Idle check failed: {e}")
        return False


def run_make_md():
    print("🖼️ Running gpt_process_images.py..")
    return subprocess.call(["python3", "gpt_process_images.py"])

def run_vector_store():
    print("🧠 Running vector_store.py...")
    return subprocess.call(["python3", "/home/frea/FREA/store_vectors/vector_store.py"])

def main_loop():
    while True:
        if is_idle():
            print("🛌 System idle — beginning processing...")
            run_make_md()
            run_vector_store()
            print("✅ Processing complete. Sleeping before re-check...")
            time.sleep(CHECK_INTERVAL * 2)
        else:
            print("💻 System active — waiting...")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()
