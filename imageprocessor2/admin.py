from flask import Flask, render_template, request, redirect
import json
import os

app = Flask(__name__)

CONFIG_FILE = 'config.json'
LOG_FILE = 'activity.log'

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        # Save new config
        new_config = {
            "conf_threshold": float(request.form['conf_threshold']),
            "max_segments": int(request.form['max_segments']),
            "min_size": int(request.form['min_size'])
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(new_config, f, indent=4)
        return redirect('/admin')

    # Load config
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    else:
        config = {}

    # Load recent log
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log_lines = f.readlines()[-50:]  # Last 50 actions
    else:
        log_lines = []

    return render_template('admin.html', config=config, log_lines=log_lines)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
