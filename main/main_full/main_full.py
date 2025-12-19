# -*- coding: utf-8 -*-
import subprocess
import time
import pandas as pd
import os


os.environ["TRANSFORMERS_VERBOSITY"] = "error"
def pause_execution(prompt):

    input(prompt)

print("Starting...")

python_files = [
    {
        "file": "dify_choose.py",
        "message": "Model selection coding"
    },
    {
        "file": "dify_download.py",
        "message": "Downloading model coding"
    },
    {
        "file": "dify_label.py",
        "message": "Label coding"
    },
    {
        "file": "dify_finetune.py",
        "message": "Fine-tune coding"
    },
]


for item in python_files:
    file = item["file"]
    message = item["message"]

    subprocess.run(["python", file], check=True)

    if file == "dify_download.py":
        print("The model is being deployed on-premises, please wait...")
        time.sleep(25)
    print("Successfully executed. Proceed to the next step...")



def pause_execution(prompt):

    input(prompt)



source_file = 'C:\\Users\\21888\\Desktop\\test\\dataset.xlsx'
target_file_1 = 'C:\\Users\\21888\\Desktop\\test\\11.xlsx'
target_file_2 = 'C:\\Users\\21888\\Desktop\\test\\22.xlsx'
df = pd.read_excel(source_file)


chunk_size = 10


start_row = 0

while start_row < len(df):

    end_row = start_row + chunk_size
    chunk = df[start_row:end_row]


    empty_row = pd.DataFrame([[None] * len(chunk.columns)], columns=chunk.columns)

    data_with_empty_row = pd.concat([empty_row, chunk], ignore_index=True)


    data_with_empty_row.to_excel(target_file_1, index=False, header=False)
    data_with_empty_row.to_excel(target_file_2, index=False, header=False)


    print(f"Processed rows {start_row + 1} to {end_row}:")


    python_files1 = [
        {
            "file": "label_toxic_1.py",
            "message": "Model_1 labeling"
        },
        {
            "file": "label_toxic_2.py",
            "message": "Model_2 labeling"
        },
        {
            "file": "label_toxic_3.py",
            "message": "Model_3 labeling"
        },
        {
            "file": "all_label.py",
            "message": "Data processing"
        },
        {
            "file": "get_finetune_dataset.py",
            "message": "Data processing"
        },
        {
            "file": "w1.py",
            "message": " "
        },
        {
            "file": "w2.py",
            "message": " "
        },
        {
            "file": "finetune_toxic_1.py",
            "message": "Model_1 fine-tuning"
        },
        {
            "file": "finetune_toxic_2.py",
            "message": "Model_2 fine-tuning"
        },
        {
            "file": "finetune_toxic_3.py",
            "message": "Model_3 fine-tuning"
        },
        {
            "file": "save_to_final.py",
            "message": "Finish! Next batch"
        },
    ]


    for item in python_files1:
        file = item["file"]
        message = item["message"]
        try:
            print(f"{message}...")

            subprocess.run(["python", file], check=True)

            print("Successfully executed. Proceed to the next step...")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {file}: {e}")
        except FileNotFoundError:
            print(f"The {file} file was not found.")

    empty_df = pd.DataFrame()
    empty_df.to_excel(target_file_1, index=False)
    empty_df.to_excel(target_file_2, index=False)


    start_row = end_row

print("All the data has been labeled.")