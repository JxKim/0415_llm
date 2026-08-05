"""
模型推理的脚本：
    输入prompt、需要验证的模型的参数路径，输出模型自回归生成的结果
"""

# 1、从命令行 读取 prompt，和模型的路径
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--prompt",type=str)
parser.add_argument("--model_path",type=str)

args = parser.parse_args()

prompt = args.prompt
model_path = args.model_path


# 2、做自回归生成

# 2.1 基于传入model_path，加载模型和tokenizer ✅
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(model_path,device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2.2 基于prompt，调用tokenizer.apply_chat_template方法，构建模型输入的token_ids

message_list = [
    {"role":"user","content":prompt}
]
token_ids:list[int] = tokenizer.apply_chat_template(message_list,tokenize=True, add_generation_prompt=True)["input_ids"]

# 加上batch_size维度，然后转化成 tensor
input_tensor = torch.tensor([token_ids],dtype=torch.long).to("cuda")
# 2.3 基于input_tensor，让模型进行自回归生成
# model.generate就是让模型做自回归生成的方法，得到result也是一个tensor实例，它的shape：batch_size, seq_len。 此处的seq_len包含两部分内容：第一部分是输入的token_ids的seq_len，第二部分是模型自回归生成的，新的token的seq_len
# 例如：input_tensors:[[23,34,56]] result: [[23,34,56,98,45]]
result = model.generate(input_tensor,max_new_tokens=500)

# 2.4 解析模型自回归生成的结果
# 2.4.1 将新生成的token_ids 切片出来
generated_new_tokens  = result[:,len(token_ids):]
# 2.4.2 调用tokenizer，对token_ids进行解码
result_txt_list = tokenizer.decode(generated_new_tokens)

print("当前新生成的txt为：\n\n",result_txt_list[0])
