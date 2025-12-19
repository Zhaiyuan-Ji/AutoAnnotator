from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # 引入CORS模块
import subprocess
import threading
import queue
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
output_queue = queue.Queue()
process = None

PYTHON_SCRIPT_PATH = "main_for_test.py"

PORT = 8080


@app.route('/')
def serve_index():
    """Serve the index.html file"""
    return send_file('index.html')


@app.route('/run_script', methods=['POST'])
def run_script():
    global process

    if not os.path.isfile(PYTHON_SCRIPT_PATH):
        return jsonify({'error': f'Script "{PYTHON_SCRIPT_PATH}" does not exist'}), 400

    def execute_script():
        try:
            process = subprocess.Popen(
                ['python', PYTHON_SCRIPT_PATH],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                bufsize=1,
                errors='replace'
            )

            while True:
                line = process.stdout.readline()
                if not line:
                    break
                output_queue.put(line.rstrip('\n'))

            process.wait()
            output_queue.put(f"=== Script execution completed (Return Code: {process.returncode}) ===")
        except Exception as e:
            error_msg = str(e)
            if "permission" in error_msg.lower() or "access" in error_msg.lower():
                error_msg += "\n\n error"
            output_queue.put(f"Error: {error_msg}")


    if process and process.poll() is None:
        process.terminate()
        process.wait()

    while not output_queue.empty():
        output_queue.get()

    threading.Thread(target=execute_script, daemon=True).start()
    return jsonify({'status': 'started', 'script_path': PYTHON_SCRIPT_PATH})


@app.route('/get_output', methods=['GET'])
def get_output():
    lines = []
    while not output_queue.empty():
        lines.append(output_queue.get())
    return jsonify({'output': lines})


@app.route('/stop_script', methods=['POST'])
def stop_script():
    global process
    if process and process.poll() is None:
        process.terminate()
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_running'})


if __name__ == '__main__':
    try:
        app.run(debug=False, threaded=True, port=PORT)
    except PermissionError:
        print(f"错误：端口 {PORT} 权限不足，尝试使用其他端口（如非root用户避免使用<1024的端口）")
    except OSError as e:
        if "address already in use" in str(e):
            print(f"错误：端口 {PORT} 已被占用，请关闭占用进程或更换端口")
        else:
            print(f"套接字错误：{e}")