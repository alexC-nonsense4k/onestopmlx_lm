import streamlit as st
import os
from concurrent.futures import ThreadPoolExecutor

from streamlit.errors import StreamlitAPIException


def download_model(hf_path, save_path):
    try:
        download_shell = (
                "huggingface-cli download --local-dir-use-symlinks False --resume-download %s --local-dir %s"
                % (hf_path, save_path + '/' + hf_path)
        )
        os.system(download_shell)
    except StreamlitAPIException:
        st.session_state['download_in_progress'] = False
        st.session_state['button_text'] = 'Download'
        raise
    else:
        st.session_state['download_in_progress'] = False
        st.session_state['button_text'] = 'Download'

def model_page():
    st.title('Download Model!')

    if 'download_in_progress' not in st.session_state:
        st.session_state['download_in_progress'] = False

    if 'button_text' not in st.session_state:
        st.session_state['button_text'] = 'Download'

    # 插入第一个文本输入框并获取用户输入
    hf_path = st.text_input("模型hf路径", "Qwen/Qwen1.5-7B-Chat")
    # 插入第二个文本输入框并获取用户输入
    #save_path = st.text_input("本地存储目录", "./modelcenter")
    save_path="./modelcenter"
    executor = ThreadPoolExecutor(max_workers=1)

    if st.button(st.session_state['button_text'], disabled=st.session_state['download_in_progress']):
        if not st.session_state['download_in_progress']:
            st.session_state['download_in_progress'] = True
            st.session_state['button_text'] = 'Downloading...'
            future = executor.submit(download_model, hf_path, save_path)
            st.write("Download started. You can continue using the app.")


