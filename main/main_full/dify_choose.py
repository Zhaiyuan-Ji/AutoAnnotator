import requests
import json
import uuid
import os


def send_streaming_request(api_key, api_base, query):
    url = f"{api_base}/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    user = str(uuid.uuid4())
    data = {
        "query": query,
        "user": user,
        "response_mode": "streaming",
        "inputs": {}
    }
    generated_content = ""
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        for line in response.iter_lines():
            if line:
                line = line.lstrip(b'data:').strip()
                try:
                    result = json.loads(line)
                    answer = result.get("answer", "")
                    if answer:
                        print(answer, end="", flush=True)
                        generated_content += answer
                except json.JSONDecodeError:
                    continue
        print()
    except Exception:
        pass


    lines = generated_content.splitlines()
    if len(lines) > 2:
        processed_content = '\n'.join(lines[1:-1])
    else:
        processed_content = ""




api_key = "app-rqo1bV6SjWQlHjzEFI66SrNh"
api_base = "http://127.0.0.1/v1"
query = """
Now I need you to help me write a code. Note that you need to strictly follow my requirements and the content you generate should be directly runnable code.
Requirements:
1. Model search: Find text annotation models similar to BERT on Hugging Face according to the text annotation requirements of sentiment classification (only supporting positive, negative, neutral and their respective confidence scores) and toxic content detection (only supporting toxic and its confidence score). Note that models with multiple labels like unitary/toxic-bert do not meet the requirements.
2. Quantity requirements: 3 toxic content detection models.
3. After selection, set up a UI interface for me to view the information of the selected models (using tkinter). 
4. Save the table in the UI interface to the local path 'C:\\Users\\21888\\Desktop\\1-24\\model.xlsx'.
Note:
1. The table is for reference only. The number of models, model parameters, and HF downloads are inaccurate and need to be investigated by you.
2. You need to search for models that meet the conditions and remember them. Your code only includes the UI part (including models that meet the conditions).
3. Please strictly follow the above requirements when writing the code.
4. Your answer should only contain the code and nothing else.
use english 

"""

send_streaming_request(api_key, api_base, query)