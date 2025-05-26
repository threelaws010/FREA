import os
import time
import subprocess
from dotenv import load_dotenv

# Requires `pyxhook` on Linux or `pynput` on Windows/macOS
import ctypes
import platform

load_dotenv()

CHECK_INTERVAL = 60  # seconds
IDLE_THRESHOLD = 300  # 5 minutes

def is_idle():
    if platform.system() == "Linux":
        class XScreenSaverInfo(ctypes.Structure):
            _fields_ = [("window", ctypes.c_ulong),
                        ("state", ctypes.c_int),
                        ("kind", ctypes.c_int),
                        ("since", ctypes.c_ulong),
                        ("idle", ctypes.c_ulong),
                        ("event_mask", ctypes.c_ulong)]

        xlib = ctypes.cdll.LoadLibrary("libX11.so")
        dpy = xlib.XOpenDisplay(None)
        if not dpy:
            return False

        xss = ctypes.cdll.LoadLibrary("libXss.so")
        xss.XScreenSaverAllocInfo.restype = ctypes.POINTER(XScreenSaverInfo)
        info = xss.XScreenSaverAllocInfo()
        xss.XScreenSaverQueryInfo(dpy, xlib.XDefaultRootWindow(dpy), info)
        idle_time = info.contents.idle / 1000
        xlib.XCloseDisplay(dpy)
        return idle_time > IDLE_THRESHOLD
    else:
        print("Idle check only supported on Linux in this script.")
        return False

def run_make_md():
    print("🖼️ Running make_md_input_GPU.py...")
    return subprocess.call(["python3", "make_md_input_GPU.py"])

def run_vector_store():
    print("🧠 Running vector_store.py...")
    return subprocess.call(["python3", "vector_store.py"])

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
