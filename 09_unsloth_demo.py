# 1、需要先引入Unsloth，从而确保，该库能够对trl，transformers和peft做深度优化
import unsloth
from trl.trainer.sft_trainer import SFTTrainer


# 2、通过unsloth加载模型
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./model/Qwen3-8B",
    load_in_4bit=True,
    # 通过传入以下两个参数，让Unsloth仅加载本地模型，边加载，边量化
    use_exact_model_name=True,
    local_files_only = True
)

# 3、通过unsloth去配LoRA
quantized_peft_model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=8,
    # target_modules= # 默认值就是所有的线性层gate,up,down, q, k, v, o
    lora_dropout=0.05,
)

# 4、数据处理：需要我们自己调tokenizer.apply_chat_template方法，将message_list转化成纯文本，以text键输出
from datasets import  load_dataset
data = load_dataset("json",data_files={"train":"./data/psychology_data.jsonl"})
data["train"]=data["train"].shuffle()
data["train"]=data["train"].select(range(16000))
data = data["train"].train_test_split(test_size=0.05)

# 1.2 将数据转换成SFTTrainer所需要的 Language Modeling 这种类型，对话格式的数据
def convert_func(examples:dict[str, list]):
    """
    接收的参数，就是.map方法传递的，原始的数据，以批次形式接收
    """
    conversation_lists: list[list] =examples["conversation"]
    text_lists:list[list] = []
    for conversation in conversation_lists:
        # conversation是单条样本所对应的列表：
        human_message = conversation[0]["human"]
        assistant_message = conversation[0]["assistant"]
        message_list = [ 
            {"role":"user","content":human_message},
            {"role":"assistant","content":assistant_message}
        ]
        result = tokenizer.apply_chat_template(message_list,tokenize=False)
        text_lists.append(result)

    return {"text":text_lists}



converted_data = data.map(convert_func,batched=True,remove_columns=data["train"].column_names,)

from trl.trainer.sft_config import SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/09_Unsloth_demo"
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
    # LoRA微调，一般要设置的比全参要更高，一般高一个数量级
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps= 0.1,
    optim="paged_adamw_32bit", # 优化器的类型，默认值就是adamW，可以选择使用paged_adamw_32bit，也即分页优化器
    # 评估和保存相关
    eval_strategy="steps",
    eval_steps=50,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    output_dir="./finetuned/09_Unsloth_demo", # 保存的是检查点
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length=700,
    
    # 原生的qwen3的聊天模板，和assistant_only_loss参数不兼容，所以需要基于原生的chat_template进行修改，得到new_chat_template.jinja文件
    # 可以通过chat_template_path传递新的chat_template文件
    # 注意：此处不需要再传入下面两个参数
    # assistant_only_loss=True,
    # chat_template_path="./new_chat_template.jinja"
)

from transformers import  AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-8B")

trainer = SFTTrainer(
    model=quantized_peft_model, # 此处传递的，不再是原模型，而是通过get_peft_model所得到的新模型
    args=config,
    train_dataset=converted_data["train"],
    eval_dataset=converted_data["test"],
    # processing_class指的就是tokenizer参数
    processing_class= tokenizer
)

from unsloth.chat_templates import train_on_responses_only
# 通过调用该方法，就能够实现，仅对assistant回答部分算损失，这是unsloth的写法
# <|im_start|>user xxxxx <|im_end|>\n <|im_start|>assistant xxxxxx <|im_end|><|im_start|>user xxxxx
trainer = train_on_responses_only(
    trainer=trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"
    )


trainer.train()
trainer.save_model("./finetuned/09_Unsloth_demo")