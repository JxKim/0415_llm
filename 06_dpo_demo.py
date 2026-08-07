from datasets import load_dataset

dataset = load_dataset("./data/ultrafeedback_binarized")

# 将数据构造成trl所需要的 implicit prompt 类型的preference数据
new_dataset = dataset.remove_columns(['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])


from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/06_dpo_demo"
dpo_config = DPOConfig(
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
        output_dir="./finetuned/06_dpo_demo", # 保存的是检查点
        bf16=True,
        gradient_checkpointing=False,
        # activation_offloading=False,
        max_length=700,
        # 不需要显示声明assistant_only_loss，但是在底层，算损失的时候，算log_prob(被训练/参考模型，基于prompt，输出chosen和rejected回答，也会使用到assistant answer mask)
        # assistant_only_loss=True,
        # chat_template_path="./new_chat_template.jinja"
        beta=0.1
)


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
model = AutoModelForCausalLM.from_pretrained("./finetuned/02_sft_demo")
tokenizer = AutoTokenizer.from_pretrained("./finetuned/02_sft_demo")
model.warnings_issued = {}
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(model,lora_config)
trainer = DPOTrainer(
    model=peft_model,
    args=dpo_config,
    processing_class=tokenizer,
    train_dataset=new_dataset["train_prefs"],
    eval_dataset=new_dataset["test_prefs"]
)


trainer.train()
trainer.save_model("./finetuned/06_dpo_demo")