import pandas as pd

finetune1_path = r'C:\Users\21888\Desktop\test\finetune.xlsx'
try:
    final_df = pd.read_excel(finetune1_path)
except FileNotFoundError:
    print(f"未找到文件: {finetune1_path}")
else:
    toxic_columns = ['toxic_1', 'toxic_2', 'toxic_3']

    toxic_consistent_indexes = []
    toxic_inconsistent_indexes = []
    if all(col in final_df.columns for col in toxic_columns):
        non_empty_mask_toxic = final_df[toxic_columns].notnull().all(axis=1)
        df_filtered_toxic = final_df[non_empty_mask_toxic]

        consistent_toxic_mask = (
            df_filtered_toxic[toxic_columns[0]] == df_filtered_toxic[toxic_columns[1]]
        ) & (
            df_filtered_toxic[toxic_columns[1]] == df_filtered_toxic[toxic_columns[2]]
        )
        toxic_consistent_indexes = df_filtered_toxic[consistent_toxic_mask].index.tolist()

        inconsistent_labels_toxic = (
            (df_filtered_toxic['toxic_1'] != df_filtered_toxic['toxic_2']) |
            (df_filtered_toxic['toxic_2'] != df_filtered_toxic['toxic_3']) |
            (df_filtered_toxic['toxic_1'] != df_filtered_toxic['toxic_3'])
        )
        toxic_inconsistent_indexes = df_filtered_toxic[inconsistent_labels_toxic].index.tolist()

        df_inconsistent_toxic = df_filtered_toxic[inconsistent_labels_toxic].copy()

        if not df_inconsistent_toxic.empty:
            toxic_output_path = r'C:\Users\21888\Desktop\test\finetune_toxic.xlsx'
            df_inconsistent_toxic.to_excel(toxic_output_path, index=False)
        else:
            print("No rows with inconsistent toxic labels found.")
    else:
        print("输出文件中缺少必要的 toxic 列。")

    all_consistent_indexes = sorted(set(toxic_consistent_indexes))
    all_inconsistent_indexes = sorted(set(toxic_inconsistent_indexes))

    consistent_result_df = final_df.loc[all_consistent_indexes]
    inconsistent_result_df = final_df.loc[all_inconsistent_indexes].iloc[:, :2]

    result_df = pd.concat([consistent_result_df, inconsistent_result_df]).sort_index()

    result_output_path = r'C:\Users\21888\Desktop\test\66.xlsx'
    result_df.to_excel(result_output_path, index=False)

    final_result_df = pd.read_excel(result_output_path)

    if 'label' not in final_result_df.columns:
        final_result_df['label'] = None

    for index, row in final_result_df.iterrows():
        if pd.notna(row['toxic_1']):
            final_result_df.at[index, 'label'] = row['toxic_1']

    final_result_df.to_excel(result_output_path, index=False)