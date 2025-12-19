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

    # 去掉第一行和最后一行
    lines = generated_content.splitlines()
    if len(lines) > 2:
        processed_content = '\n'.join(lines[1:-1])
    else:
        processed_content = ""




api_key = "app-2AsDyyk9Jwv1InDdNtn0Ox0F"
api_base = "http://127.0.0.1/v1"
query = """
Now I need you to help me write a code. Note that you need to strictly follow my requirements and the content you generate should be directly runnable code.
I need to deploy a Hugging Face model locally. The model address is https://hf-mirror.com/models. Please help me complete the following tasks:
1. Model ID acquisition: Get the model IDs I want to download from the 'Model ID' column in the file 'C:\\Users\\21888\\Desktop\\1-24\\model.xlsx'.
2. Save path: I want to save the models to the local directory 'E:\model'. For example, save 'cardiffnlp_twitter-roberta-base-sentiment-latest' to 'E:\\model\\cardiffnlp_twitter-roberta-base-sentiment-latest'.
3. Code generation: Please generate a complete and directly runnable Python code to download the models locally.
Please generate the code according to the above requirements to complete the deployment!
Note: Your answer should only contain the code.
use english 

"""

send_streaming_request(api_key, api_base, query)