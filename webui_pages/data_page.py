import os

import streamlit as st
import json
def data_page():
    st.title('Upload File!')
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "jsonl"])

    # 如果用户上传了文件，根据文件类型使用pandas读取文件并显示数据
    if uploaded_file is not None:
        import pandas as pd

        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.jsonl'):
            data = pd.read_json(uploaded_file, lines=True)
        st.write(data.head(10))

        st.title('Data Format')

        data_dir=st.text_input("数据存储位置", uploaded_file.name[:uploaded_file.name.rfind('.')])

        # 添加一个选择框让用户选择数据处理模型
        model_option = st.selectbox(
            'Choose a data processing model',
            options=['text', 'completions', 'chat'])

        # 显示用户选择的结果
        'You selected:', model_option

        json_format = {}
        # 根据用户选择的模型进行数据处理
        if model_option == 'text':
            json_format = {"text":""}
            text_column = st.selectbox('Choose a column for "text"', options=data.columns)
            st.write('Processing data with text format...')
        elif model_option == 'completions':
            json_format = {"prompt":"","completion":""}
            prompt_column = st.selectbox('Choose a column for "prompt"', options=data.columns)
            completion_column = st.selectbox('Choose a column for "completion"', options=data.columns)
            st.write('Processing data with completions format...')
        elif model_option == 'chat':
            messages_column = st.selectbox('Choose a column for "messages"', options=data.columns)
            st.write('Processing data with chat format...')

        # 让用户选择训练集的比例
        train_ratio = st.slider('Training set ratio', min_value=0.0, max_value=1.0, value=0.6)

        # 让用户选择验证集的比例，最大值为1减去训练集的比例
        valid_ratio = st.slider('Validation set ratio', min_value=0.0, max_value=1.0 - train_ratio, value=0.2)

        # 测试集的比例自动设置为剩下的部分
        test_ratio = 1.0 - train_ratio - valid_ratio
        st.write('Test set ratio: ', test_ratio)

        if st.button('Start processing data'):
            train_size = (int)(data.shape[0] * train_ratio)
            valid_size = min((int)(data.shape[0] * valid_ratio), data.shape[0] - train_size)
            test_size = (int)(data.shape[0] - train_size - valid_size)

            train_data = data.sample(train_size)
            valid_data = data.drop(train_data.index).sample(valid_size)
            test_data = data.drop(train_data.index).drop(valid_data.index)
            if not os.path.exists(st.session_state.DEFAULT_DATA_CENTER+data_dir):
                os.makedirs(st.session_state.DEFAULT_DATA_CENTER+data_dir)
            for dataset, filename in zip(
                    [train_data, valid_data, test_data],
                    ['train.jsonl', 'valid.jsonl', 'test.jsonl']
            ):
                with open(st.session_state.DEFAULT_DATA_CENTER+data_dir+"/"+filename, 'a', encoding='utf-8') as f:
                    for _, row in dataset.iterrows():
                        if model_option == 'text':
                            json_format['text']=row[text_column]
                            f.write(json.dumps(json_format,ensure_ascii=False))
                            f.write('\n')
                        elif model_option == 'completions':
                            json_format['prompt']=row[prompt_column]
                            json_format['completion']=row[completion_column]
                            f.write(json.dumps(json_format,ensure_ascii=False))
                            f.write('\n')
                        elif model_option == 'chat':
                            f.write(json_format.format(messages=row[messages_column]))  # Adjust as needed.
            st.write('Data processing completed!')