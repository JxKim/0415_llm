"""
将LoRA微调·后的适配器的模型权重，和基座模型，做合并，从而避免，LoRA微调之后，推理有额外的开销
"""
from peft import PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM
# 1、加载适配器和基座模型权重
adpater_path = "/root/autodl-tmp/finetune_proj/finetuned/lora_sweep/compare_learning_rate/learning_rate_3em4/final_model"
tokenizer = AutoTokenizer.from_pretrained(adpater_path)
model = AutoModelForCausalLM.from_pretrained("./model/Qwen3-0.6B")
# peft_model = PeftModel.from_pretrained("./model/Qwen3-0.6B",adpater_path) # 基座模型，不能直接传入一个路径
peft_model = PeftModel.from_pretrained(model,adpater_path)


# 2、调peft_model的merge_and_unload()方法，得到合并之后的模型
merged_model = peft_model.merge_and_unload()


# 3、保存
# 
# 3.1 将merged_model保存
merged_model.save_pretrained("./finetuned/lora_merged_model")
# 3.2 将tokenzier保存到和merged_model相同的目录当中去，从而使得后面，可以基于同一个目录加载模型和tokenizer
tokenizer.save_pretrained("./finetuned/lora_merged_model")



from peft import prepare_model_for_kbit_training