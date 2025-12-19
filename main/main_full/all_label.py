import pandas as pd
import os

# 定义基础路径
base_path = r'C:\Users\21888\Desktop\test'

file_11_path = os.path.join(base_path, '11.xlsx')
finetune1_path = os.path.join(base_path, 'finetune.xlsx')

df_11 = pd.read_excel(file_11_path, header=None, skiprows=1)
df_11.columns = ['text']

final_df = df_11.copy()

for filename in os.listdir(base_path):
    if filename.startswith('result') and filename.endswith('.xlsx'):
        file_path = os.path.join(base_path, filename)
        df_result = pd.read_excel(file_path)

        toxic_columns = [col for col in df_result.columns if col.startswith('toxic')]
        df_toxic = df_result[toxic_columns]

        max_rows = max(len(final_df), len(df_toxic))
        final_df = final_df.reindex(range(max_rows)).fillna('')
        df_toxic = df_toxic.reindex(range(max_rows)).fillna('')

        final_df = pd.concat([final_df, df_toxic], axis=1)

final_df.insert(1, 'label', '')

final_df.to_excel(finetune1_path, index=False)




