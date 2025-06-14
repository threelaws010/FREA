from flask import Flask, render_template, request, redirect
import json
import os
import subprocess
from datetime import datetime

app = Flask(__name__)

CONFIG_FILE = 'config.json'
LOG_FILE = 'activity.log'

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        # Save updated config values
        new_config = {
            "conf_threshold": float(request.form.get('conf_threshold', 0.25)),
            "max_segments": int(request.form.get('max_segments', 10)),
            "min_size": int(request.form.get('min_size', 20))
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(new_config, f, indent=4)

        # Build command-line args
        flags = []
        if 'force' in request.form:
            flags.append('--force')
        if 'ocr_only' in request.form:
            flags.append('--ocr-only')
        if 'no_segmentation' in request.form:
            flags.append('--no-segmentation')

        command = ['python3', 'make_md_input_GPU.py'] + flags

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            log_entry = f"[{datetime.now()}] Ran: {' '.join(command)}\n{result.stdout}\n{result.stderr}\n"
        except Exception as e:
            log_entry = f"[{datetime.now()}] Error running script: {e}\n"

        # Save log
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(log_entry)

        return redirect('/admin')

    # Load config
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)

    # Load recent log
    log_lines = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log_lines = f.readlines()[-50:]

    return render_template('admin.html', config=config, log_lines=log_lines)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
