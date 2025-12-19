import requests
import json
import uuid
import os
import pandas as pd


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
                    # 避免命名冲突，将变量名修改为 response_result
                    response_result = json.loads(line)
                    answer = response_result.get("answer", "")
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



api_key = "app-gmcRiLLv6HWrl0V2nL003Nhe"
api_base = "http://127.0.0.1/v1"

# 读取 Excel 文件
excel_path = r'C:\Users\21888\Desktop\1-24\model.xlsx'
df = pd.read_excel(excel_path)

for index, row in df.iterrows():
    # 获取所需信息
    model_id = row['Model ID']
    model_type = row['Type']
    k = row['Serial Number']
    model_address = os.path.join(r'E:\model', model_id.replace('/', '_'))

    # 构造标签列名和置信分数列名
    label_col_name = f"{model_type}_{k}"
    confidence_col_name = f"{model_type}_confidence_{k}"

    # 定义 result_path 变量
    result_path = f"result_{model_type}_{k}"

    # 构造 query
    query = f"""
Now I need you to help me write a code. Note that you need to strictly follow my requirements and the content you generate should be directly runnable code. Your task is to help me call a pre - trained BERT model (or similar models) from the local and use this model to label text data. The following are the specific requirements of the task:
Task description:
1. Model type: Use a pre - trained model like BERT or similar ones (e.g., RoBERTa, DistilBERT, etc.).
2. Model ID: {model_id}
3. Local model address: {model_address}
4. Task type: {model_type}.
5. Input data: The address of the data file in the first column of the local xlsx file is 'C:\\Users\\21888\\Desktop\\test\\11.xlsx'.
6. Output result: The labeled result. If there is a confidence score, it is also required. Save it to 'C:\\Users\\21888\\Desktop\\test\\{result_path}.xlsx'.
7. Requirements for the result file: Change the column name of the label column in the saved file to '{label_col_name}', and change the column name of the confidence score column to '{confidence_col_name}'.
Code requirements:
1. Model loading: Load the pre - trained model and tokenizer from the local path.
2. Inference logic: Call the model for inference and parse the output result.
3. Result saving: Save the labeled result to the specified location.
Note: Your answer should only contain the code and nothing else.
use english 

"""

    file_name = f'label_{model_type}_{k}.py'

    send_streaming_request(api_key, api_base, query, file_name)
