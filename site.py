from flask import Flask, render_template, request, jsonify
from youtube_extract import download_and_extract_frames
from threading import Thread
import subprocess
import time
import os

app = Flask(__name__)

status_message = ""

# Set the path to your train.py and chord_run.py scripts using os.path.abspath
TRAIN_SCRIPT_PATH = os.path.abspath("Guitar_Chord_Recognition/train.py")
CHORD_RUN_SCRIPT_PATH = os.path.abspath("Guitar_Chord_Recognition/chord_run.py")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_frames', methods=['POST'])
def extract_frames():
    global status_message
    youtube_url = request.form['youtube_url']

    status_message = "Extracting frames..."

    def process_video(url):
        global status_message
        start_time = time.time()
        try:
            download_and_extract_frames(url)
            elapsed_time = time.time() - start_time
            status_message = f"DONE! Extraction completed in {elapsed_time:.2f} seconds."
        except Exception as e:
            status_message = f"Failed to extract video: {str(e)}"

    thread = Thread(target=process_video, args=(youtube_url,))
    thread.start()

    return jsonify({'status': 'started', 'message': 'Frame extraction started.'})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        process = subprocess.Popen(['python', TRAIN_SCRIPT_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        if process.returncode == 0:
            process_run = subprocess.Popen(['python', CHORD_RUN_SCRIPT_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output_run, error_run = process_run.communicate()

            if process_run.returncode == 0:
                return jsonify({'status': 'success', 'message': 'Training Done and Chord Run Complete!'})
            else:
                try:
                    error_message = error_run.decode('utf-8')
                except UnicodeDecodeError:
                    error_message = error_run.decode('latin-1')
                return jsonify({'status': 'failed', 'message': 'Chord run failed.', 'error': error_message})
        else:
            try:
                error_message = error.decode('utf-8')
            except UnicodeDecodeError:
                error_message = error.decode('latin-1')
            return jsonify({'status': 'failed', 'message': 'Model training failed.', 'error': error_message})
    except FileNotFoundError as e:
        return jsonify({'status': 'failed', 'message': 'File not found.', 'error': str(e)})

@app.route('/status', methods=['GET'])
def status():
    global status_message
    return jsonify({'status': status_message})

if __name__ == "__main__":
    app.run(debug=True)
