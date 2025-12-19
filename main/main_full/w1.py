import pandas as pd
from openai import OpenAI
import re
import os

# Define the expected labels for validation.
EXPECTED_LABELS = {'positive', 'negative', 'neutral', 'toxic', 'non-toxic'}


def extract_label(text):
    # Use regex to find the label in the text. This assumes the label is on a line by itself or after "Output: ".
    match = re.search(r'\b(?:positive|negative|neutral|toxic|non-toxic)\b', text, re.IGNORECASE)
    if match:
        return match.group().lower()
    else:
        print(f"Could not find an expected label in the response: {text}. Returning 'unknown'.")
        return 'unknown'


def classify_text(text, client):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "You are an autoclassifier that's responsible for labeling input text. You must respond with only one of these labels: toxic, non-toxic."},
                {"role": "user", "content": f"Input sentence: '{text}'"}
            ],
            stream=False
        )
        label = extract_label(response.choices[0].message.content.strip())
        return label
    except Exception as e:
        print(f"An error occurred while classifying text: {e}")
        return 'unknown'


def remove_extra_columns(input_excel_file, output_excel_file, sheet_name):

    try:
        df = pd.read_excel(input_excel_file, sheet_name=sheet_name)

        if len(df.columns) < 2:
            print(f"工作表 '{sheet_name}' 中少于两列")
            return

        df = df.iloc[:, :2]

        os.makedirs(os.path.dirname(output_excel_file), exist_ok=True)

        df.to_excel(output_excel_file, sheet_name=sheet_name, index=False, engine='openpyxl')

    except Exception as e:
        print(f"processing {input_excel_file} error: {e}")


# Initialize the OpenAI client with your API key and base URL.
client = OpenAI(api_key="sk-5e0f127935dc46a0a5aa56fb50727345", base_url="https://api.deepseek.com")

# 指定xlsx文件路径
file_path = r'C:\Users\21888\Desktop\test\finetune_toxic.xlsx'
remove_extra_columns(file_path, file_path, 'Sheet1')
# Load the specified xlsx file into a DataFrame.
df = pd.read_excel(file_path, header=0)  # Use header=0 if your file has headers.

# Check if the DataFrame is empty or does not contain any rows after the header.
if df.empty:
    print(f"The provided Excel file {os.path.basename(file_path)} is empty.")
else:

    df.columns = ['text', 'label'] if len(df.columns) >= 2 else ['text']

    # Classify each sentence in the first column and store the result in a new list.
    results = []
    for index, row in df.iterrows():
        sentence = row.iloc[0]  # Using iloc to ensure we get the correct column.
        label = classify_text(sentence, client)
        results.append(label)
        print(f"Classified sentence {index + 1} in {os.path.basename(file_path)}: {label}")  # Debugging output.

    # Ensure the length of results matches the number of rows in the DataFrame.
    if len(results) == len(df):
        # If 'label' column exists, update it; otherwise, insert it.
        if 'label' in df.columns:
            df['label'] = results  # Update existing 'label' column
        else:
            df.insert(1, 'label', results)  # Insert new 'label' column

        # Save the updated DataFrame back to the xlsx file.
        df.to_excel(file_path, index=False)
    else:
        print(f"Error: The number of classification results does not match the number of sentences in {os.path.basename(file_path)}.")