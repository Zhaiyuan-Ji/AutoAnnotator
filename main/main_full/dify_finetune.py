import requests
import json
import uuid
import os
import pandas as pd
import re


def extract_python_code(content):

    pattern = r'```python(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:

        return matches[0].strip()
    return content.strip()


def send_streaming_request(api_key, api_base, query, file_name):
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
                    response_result = json.loads(line)
                    answer = response_result.get("answer", "")
                    if answer:
                        print(answer, end="", flush=True)
                        generated_content += answer
                except json.JSONDecodeError:
                    continue
        print()
    except Exception as e:
        print(f"error: {e}")


    python_code = extract_python_code(generated_content)


    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(current_dir, exist_ok=True)



api_key = "app-AWP4E1DhI7ddkdnWXHZdRX5n"
api_base = "http://127.0.0.1/v1"

# 读取 Excel 文件
excel_path = r'C:\Users\21888\Desktop\1-24\model.xlsx'
df = pd.read_excel(excel_path)

for index, row in df.iterrows():
    model_id = row['Model ID']
    model_type = row['Type']
    k = row['Serial Number']
    model_address = os.path.join(r'E:\model', model_id.replace('/', '_'))


    label_col_name = f"{model_type}_k_{k}"
    confidence_col_name = f"{model_type}_confidence_k_{k}"


    result_path = f"result_{model_type}_{k}"

    query = f"""
Now I need you to help me write a code. Note that you must strictly follow my requirements, and the content you generate should be directly runnable code. Your task is to write code to perform full-parameter fine-tuning on the {model_id} model. The following are the specific requirements of the task:
Task description:
1. Model type: Use a pre-trained model like BERT or similar ones (e.g., RoBERTa, DistilBERT, etc.).
2. Model ID: {model_id}
3. Local model address: {model_address}
4. Fine-tuning data: The 'text' column in the 'C:\\Users\\21888\\Desktop\\test\\finetune_{model_type}.xlsx' file is the text column of the fine-tuning data, and the 'label' column is the label column of the fine-tuning data.
5. Save address for the fine-tuned model: {model_address}.
Note: Your answer should only contain the code and nothing else.
use english 

"""

    file_name = f'finetune_{model_type}_{k}.py'

    send_streaming_request(api_key, api_base, query, file_name)
