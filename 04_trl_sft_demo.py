from datasets import load_dataset
# 1.1 加载数据
data = load_dataset("json",data_files={"train":"./data/keywords_data_train.jsonl","test":"./data/keywords_data_test.jsonl"})
data["train"]=data["train"].shuffle()
data["train"]=data["train"].select(range(16000))

# 1.2 将数据转换成SFTTrainer所需要的 Language Modeling 这种类型，对话格式的数据
def convert_func(examples:dict[str, list]):
    """
    接收的参数，就是.map方法传递的，原始的数据，以批次形式接收
    """
    conversation_lists: list[list] =examples["conversation"]
    messages_lists:list[list] = []
    for conversation in conversation_lists:
        # conversation是单条样本所对应的列表：
        human_message = conversation[0]["human"]
        assistant_message = conversation[0]["assistant"]
        message_list = [ 
            {"role":"user","content":human_message},
            {"role":"assistant","content":assistant_message}
        ]
        messages_lists.append(message_list)

    return {"messages":messages_lists}



converted_data = data.map(convert_func,batched=True,remove_columns=data["train"].column_names,)



from trl.trainer.sft_config import SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/accelerate_demo"
config = SFTConfig(
    # 数据规模相关的
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps= 8,
    max_steps=500,
    # num_train_epochs= # max_steps会比num_train_epochs的优先级更高
    # 训练可视化相关
    logging_strategy="steps",
    logging_steps=25,
    report_to="tensorboard", # 要想去进一步制定tensorboard 日志文件保存位置，需要通过os.environ去指定,
    # 学习率和优化器相关
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_steps= 0.1,
    # optim="" 优化器的类型，默认值就是adamW
    # 评估和保存相关
    eval_strategy="steps",
    eval_steps=50,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    output_dir="./finetuned/accelerate_demo", # 保存的是检查点
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length=700,
    # 原生的qwen3的聊天模板，和assistant_only_loss参数不兼容，所以需要基于原生的chat_template进行修改，得到new_chat_template.jinja文件
    # 可以通过chat_template_path传递新的chat_template文件
    assistant_only_loss=True,
    chat_template_path="./new_chat_template.jinja"
)


from transformers import AutoModelForCausalLM,AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-8B/")
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-8B")
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=converted_data["train"],
    eval_dataset=converted_data["test"],
    # processing_class指的就是tokenizer参数
    processing_class= tokenizer
)


# 封装了整个，训练，验证，保存完整的过程，可以传递resume_from_checkpoint，表示从某个检查点开始训练，不传表示从头开始
trainer.train()


trainer.save_model("./finetuned/accelerate_demo")