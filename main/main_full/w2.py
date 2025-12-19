import pandas as pd
import os

def replace_content_in_excel(input_excel_file, output_excel_file, sheet_name, column_name):

    try:
        df = pd.read_excel(input_excel_file, sheet_name=sheet_name)

        if column_name not in df.columns:
            print(f" '{column_name}' does not exist in sheet '{sheet_name}' ")
            return

        # 替换内容
        replacements = {
            'positive': 0,
            'negative': 1,
            'neutral': 0,
            'toxic': 1,
            'non-toxic': 0,
            'non': 0,
            'normal': 0,
            'unknown': 0,
        }
        df[column_name] = df[column_name].replace(replacements)

        os.makedirs(os.path.dirname(output_excel_file), exist_ok=True)

        df.to_excel(output_excel_file, sheet_name=sheet_name, index=False, engine='openpyxl')

        print("LLM completed the SecondaryReview of difficult samples")
    except Exception as e:
        print(f"processing {input_excel_file} error: {e}")

###########################################################################################
input_excel_file = r'C:\Users\21888\Desktop\test\finetune_toxic.xlsx'
output_excel_file = input_excel_file
sheet_name = 'Sheet1'
column_name = 'label'


replace_content_in_excel(input_excel_file, output_excel_file, sheet_name, column_name)



