# 导入streamlit库
import sys
import types

import streamlit as st
import yaml
import mlx.optimizers as optim
from mlx_lm.lora import print_trainable_parameters
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, build_schedule
from mlx_lm.utils import load, save_config
import webui_pages.data_page as dp
import webui_pages.model_page as mp
import os
from pathlib import Path


DEFAULT_DATA_CENTER='./datacenter/'
DEFAULT_MODEL_CENTER='./modelcenter/'
DEFAULT_TRAIN_CENTER='./traincenter/'
# 配置文件名称
CONFIG_FILE = 'config.yaml'

def workSpaceInit():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file)
        data_center = config.get('data_center', DEFAULT_DATA_CENTER)
        model_center = config.get('model_center', DEFAULT_MODEL_CENTER)
        train_center = config.get('train_center', DEFAULT_TRAIN_CENTER)

    else:
        data_center = DEFAULT_DATA_CENTER
        model_center = DEFAULT_MODEL_CENTER
        train_center = DEFAULT_TRAIN_CENTER

        # 创建配置文件
        config = {
            'data_center': data_center,
            'model_center': model_center,
            'train_center': train_center
        }
        with open(CONFIG_FILE, 'w') as file:
            yaml.safe_dump(config, file)
    
    if not os.path.exists(DEFAULT_DATA_CENTER):
        os.makedirs(DEFAULT_DATA_CENTER)

    if not os.path.exists(DEFAULT_MODEL_CENTER):
        os.makedirs(DEFAULT_MODEL_CENTER)

    if not os.path.exists(DEFAULT_TRAIN_CENTER):
        os.makedirs(DEFAULT_TRAIN_CENTER)

    st.session_state.DEFAULT_DATA_CENTER = data_center
    st.session_state.DEFAULT_MODEL_CENTER = model_center
    st.session_state.DEFAULT_TRAIN_CENTER = train_center


workSpaceInit()
# 在侧边栏中添加一个选择框
option = st.sidebar.selectbox(
    'Happy Happy A Nice Day',
     options=['Data Space', 'Model Space','Train Space','Battle Space'])

# 如果用户选择 "Data Space"，则提供文件上传选项
if option == 'Data Space':
    dp.data_page()
elif option == 'Model Space':
    mp.model_page()
elif option == 'Train Space':
    st.title('Train Your Model!')
    train_option = st.selectbox(
        'Choose a training method',
        options=['lora', 'q-lora'])
    model_list=[]
    for item in os.listdir(DEFAULT_MODEL_CENTER):
        item_path = os.path.join(DEFAULT_MODEL_CENTER, item)
        # 检查这个路径是不是目录
        if os.path.isdir(item_path):
            # 如果是目录，遍历这个目录下的所有子目录
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                # 检查子路径是不是目录，如果是，添加到 model_list
                if os.path.isdir(sub_item_path):
                    # 这里可以添加一些逻辑来处理或格式化目录名，如果需要的话
                    model_list.append(os.path.join(item, sub_item).replace("\\","/"))  # 添加完整路径或相对路径，根据需要修改
    model_option=st.selectbox(
        'Choose a training model',
        options=model_list,
        index=0,
    )

    dataset_list=[]
    for item in os.listdir(DEFAULT_DATA_CENTER):
        item_path = os.path.join(DEFAULT_DATA_CENTER, item)
        if os.path.isdir(item_path):
            dataset_list.append(item_path)
    dataset_option=st.selectbox(
        'Choose a training dataset',
        options=dataset_list,
        index=0,
    )


    if train_option=="lora":
        if model_option !=' ':
            params = {
                'model': st.session_state.DEFAULT_MODEL_CENTER+model_option,
                'train': True,
                'data': dataset_option,
                'seed': 0,
                'lora_layers': 16,
                'batch_size': 1,
                'iters': 1000,
                'val_batches': 25,
                "learning_rate": 1e-5,
                "adapter_path": st.session_state.DEFAULT_TRAIN_CENTER+"lora/"+model_option,
                "rank": 8,
                "alpha": 16.0,
            }

            with st.form(key='my_form'):
                for param, default in params.items():
                    if type(default)== str:
                        params[param] = st.text_input(param, default)
                    elif type(default)== bool:
                        params[param] = bool(st.number_input(param, default))
                    else:
                        params[param]=st.number_input(param,default)
                submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                params["lora_parameters"]={"rank":params["rank"],"alpha":params["alpha"],"scale": 10.0,"dropout": 0.10}
                params["test"]=False
                params["use_dora"]=False
                params["save_every"]=100
                params["steps_per_eval"]=50
                params["steps_per_report"]=10
                params["max_seq_length"]=4096
                params["grad_checkpoint"]=False
                params["lr_schedule"]=None
                del params["rank"]
                del params["alpha"]
                # 动态更新控制台输出
                output_area = st.empty()
                args = types.SimpleNamespace(**params)
                print(args)
                model, tokenizer = load(args.model)
                model.freeze()
                adapter_path = Path(args.adapter_path)
                adapter_file = adapter_path / "adapters.safetensors"
                adapter_path.mkdir(parents=True, exist_ok=True)
                save_config(vars(args), adapter_path / "adapter_config.json")
                linear_to_lora_layers(model, args.lora_layers, args.lora_parameters, args.use_dora)
                print_trainable_parameters(model)
                train_set, valid_set, test_set = load_dataset(args, tokenizer)

                training_args = TrainingArgs(
                    batch_size=args.batch_size,
                    iters=args.iters,
                    val_batches=args.val_batches,
                    steps_per_report=args.steps_per_report,
                    steps_per_eval=args.steps_per_eval,
                    steps_per_save=args.save_every,
                    adapter_file=adapter_file,
                    max_seq_length=args.max_seq_length,
                    grad_checkpoint=args.grad_checkpoint,
                )
                model.train()
                opt = optim.Adam(
                    learning_rate=(
                        build_schedule(args.lr_schedule)
                        if args.lr_schedule
                        else args.learning_rate
                    )
                )
                train(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    optimizer=opt,
                    train_dataset=train_set,
                    val_dataset=valid_set,
                )


