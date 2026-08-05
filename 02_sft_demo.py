from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/Qwen3-0.6B-Base")


def get_train_data(sft_config):
    """
    获取训练数据
    """

    from datasets import load_dataset

    train_data = load_dataset("./data/ultrachat_200k")["train_sft"]

    result_token_ids = []

    for i in range(sft_config.train_data_size):

        message_list: list[dict] = train_data[i]["messages"]

        token_ids_list: list[int] = tokenizer.apply_chat_template(message_list, tokenize=True,truncation=True, max_length = 2400)["input_ids"]
        result_token_ids.append(token_ids_list)

    return result_token_ids

def get_test_data(sft_config):
    """
    获取训练数据
    """

    from datasets import load_dataset

    test_data = load_dataset("./data/ultrachat_200k")["test_sft"]

    result_token_ids = []

    for i in range(sft_config.test_data_size):

        message_list: list[dict] = test_data[i]["messages"]

        token_ids_list: list[int] = tokenizer.apply_chat_template(message_list, tokenize=True)["input_ids"]
        result_token_ids.append(token_ids_list)

    return result_token_ids


from transformers import PreTrainedTokenizerFast
from typing import List
import torch 
def create_answer_mask(labels,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从labels当中找出assistant回答的部分，然后输出一个与labels相同shape的mask
    """
    # 构建answer mask，输入的labels为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和labels相同shape的全0矩阵
    answer_mask = torch.zeros_like(labels)

    # 2、找到<|im_end|> 所对应的token_id
    eos_token_id = tokenizer.encode("<|im_end|>")[0]

    # 3、遍历labels中的每一个样本
    # labels.shape: batch_size, seq_len
    for idx,ids in enumerate(labels):
        # 3.1、获取到所有的eos_position
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()
        # 3.2、解析获得user_ends和assistant_ends
        user_ends,assistant_ends = _parse_conversation_turns(eos_position)
        # 3.3、设置answer mask
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)   
    
    # 4、结果返回:
    return answer_mask

def _parse_conversation_turns(eos_positions:List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为10，第二个为15，第三个为20，第四个为25
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取，每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[10,20]
        assistant_ends:[15,25]
    """

    use_ends = [pos for pos in eos_positions[::2]]
    assistant_ends = [pos for pos in eos_positions[1::2]]

    return use_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为10，第二个为15，第三个为20，第四个为25
    那么user_turns:[10,20]，assistant_ends:[15,25]

    
    要想获取到assistant的回答的起始位置，就需要跳过<|im_end|>, \n, <|im_start|>,assistant , \n 这5个token
    要想获取到assistant的回答的结束位置，需要将<|im_end|>也包括进去，又因为列表切片是左闭右开的，所以需要向后移动一位
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    # 多轮对话没有被截断或者最后一轮整个assistant回答被截断，user轮数和assistant轮数一致
    if num_user_turns == num_assistant_turns:
        for user_end,assistant_end in zip(user_ends,assistant_ends):
            answer_start = user_end + 5
            answer_end = assistant_end + 1
            mask[answer_start:answer_end] = 1

    # 最后一轮，assistant回答被部分截断，此时user轮数比assistant轮数多一轮
    elif num_user_turns == num_assistant_turns + 1:
        for user_end,assistant_end in zip(user_ends[:-1],assistant_ends):
            answer_start = user_end + 5
            answer_end = assistant_end + 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1] 
        last_answer_start = last_user_end + 5
        mask[last_answer_start:] = 1


def compute_loss(output_logits, labels, assistant_answer_mask):
    """
    基于模型输出的结果，和答案，以及assistant_answer_mask，算SFT的损失，
    Args:
        output_logits: shape, [batch_size, seq_len, vocab_size],
        labels: shape, [batch_size, seq_len]
        assistant_answer_mask:  shape, [batch_size, seq_len]
    """
    # 1、获取到此表维度的，所有token的对数概率分布
    # log_probs: batch_size, seq_len, vocab_size
    log_probs = torch.nn.functional.log_softmax(output_logits,dim=-1)


    # 2、从对数概率分布当中，获取真实标签，所对应的对数概率；此处需要使用到一个torch.gather算子来获取
    # shape: batch_size, seq_len
    label_log_prob = torch.gather(
            input= log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

    # 3、将log_prob和assistant_answer_mask做掩码，从而使得，需要计算loss的位置，仍保留原值，不需要计算loss的位置，将其置为0
    negative_masked_label_log_prob = (-1) * label_log_prob*assistant_answer_mask

    # 4、求当前批次的一个平均损失
    loss = negative_masked_label_log_prob.sum() / assistant_answer_mask.sum() # assistant_answer_mask.sum()表示的是当前批次里面，有效token数量

    return loss


from dataclasses import dataclass

@dataclass
class SFTConfig:

    lr:float = 3e-5 # SFT全参微调：1e-5 - 5e-5 ，

    train_data_size:int = 20000 # 训练样本数量，在get_train_data()方法当中，会调用

    batch_size:int = 4

    warmup_ratio:float = 0.1

    eval_iter:int = 100

    test_data_size:int = 500 

    log_dir:str = "./logs/02_sft_demo"

    log_iter:int = 100

    save_dir:str = "./finetuned/02_sft_demo"



import numpy as np
def cosine_decay(batch, total_batch, lr, warmup_ratio):
    """

    """

    warmup_batch =  total_batch * warmup_ratio

    if batch< warmup_batch:

        k = lr / warmup_batch

        return k * (batch+1)

    else:
        # progress: 表示衰减的进度，从0到1
        # cosine(progress * π)： 从1到-1
        # 我们所希望的值的变化范围，是从1到0： cosine(progress * π)+1 -> 从2到0，再乘以0.5， 从1到0
        
        progress = (batch - warmup_batch) / (total_batch - warmup_batch)

        decay_value = (np.cos(progress * np.pi)+1) * 0.5

        return lr * decay_value
        

def get_eval_data_loss(model,test_data,sft_config):
    """
    训练过程当中，评估:
    1、遍历test_data，每次使用一小批次，给到model，让模型做前向传播
    2、基于logits, labels, assistant_answer_mask，算得当前批次的损失
    3、算得所有批次的平均损失，
    """
    model.eval()
    total_batch = (len(test_data) + sft_config.batch_size - 1) // sft_config.batch_size
    all_batch_loss = []
    for batch in range(total_batch):

        # 2、张量准备：遍历数据，对input_ids进行padding，获取input_ids以及labels， assistant_answer_mask
        # data[0:4] data[4:8], data[8:12], data[12:16]
        # 2.1 对总的数据进行切片，获取到当前批次所对应的数据
        batch_data:list[list[int]] = test_data[batch*sft_config.batch_size : (batch+1)*sft_config.batch_size]

        # 2.2 对该批次的数据，做一个padding
        batch_max_length = max([len(sample) for sample in batch_data])

        for sample in batch_data:
            padding_length = batch_max_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)

        batch_tensor = torch.tensor(batch_data, dtype=torch.long).to("cuda")


        # 2.3 构造labels, input_ids, assistant_answer_mask

        input_ids = batch_tensor[:,:-1]
        labels = batch_tensor[:,1:]
        assistant_answer_mask = create_answer_mask(labels=labels, tokenizer=tokenizer)


        # 3、模型前向传播：前向传播，获取lm head logits；
        # softmax(output_logits)
        with torch.no_grad():
            output_logits = model(input_ids).logits

        # 4、损失计算：对logits和labels，使用损失函数，计算当前batch损失；

        loss = compute_loss(output_logits, labels, assistant_answer_mask)

        all_batch_loss.append(loss)

    return sum(all_batch_loss) / len(all_batch_loss)




def train(sft_config:SFTConfig):
    """
    1、初始化模型，优化器等状态 ✅    
    2、张量准备：遍历数据，对input_ids进行padding，获取input_ids以及labels， assistant_answer_mask； ✅
	3、模型前向传播：前向传播，获取lm head logits； ✅
	4、损失计算：对logits和labels，使用损失函数，计算当前batch损失； ✅
	5、反向传播：反向传播，计算梯度； ✅
	6、调度学习率，做参数更新：使用余弦调度器去调度学习率，再使用特定的优化器（例如AdamW）对参数进行更新。
    """

    # 1、初始化模型，优化器等状态，获取训练数据, 获取tensorboard日志记录所需要的summary writer的实例，和tqdm的实例，来记录日志: 加载qwen3-0.6B-Base模型，构建一个优化器AdamW，获取训练数据
    from transformers import AutoModelForCausalLM # 大语言模型的加载，一般都是使用AutoModelForCausalLM
    from torch.optim import AdamW
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B-Base/")
    model.to("cuda")
    model.train()
    # 大模型微调，使用的AdamW，而不使用Adam, AdamW是对Adam的一个优化：将梯度更新和权重衰减，这个过程做了一个解耦
    optimizer = AdamW(model.parameters(),lr=sft_config.lr)
    # 获取训练用的token_ids
    data:list[list[int]] = get_train_data(sft_config)
    test_data: list[list[int]] = get_test_data(sft_config)
    # 如果len(data)能够被sft_config.batch_size 整除， 如果len(data) 不能够被sft_config.batch_size 整除，可能会余1,2,.. batch_size-1 + batch_size-1 最大可能是2batch_size-2
    total_batch = (len(data) + sft_config.batch_size - 1) // sft_config.batch_size

    # 
    writer = SummaryWriter(log_dir=sft_config.log_dir)
    progress_bar = tqdm(total=total_batch)

    total_loss_list = []
    for batch in range(total_batch):

        # 2、张量准备：遍历数据，对input_ids进行padding，获取input_ids以及labels， assistant_answer_mask
        # data[0:4] data[4:8], data[8:12], data[12:16]
        # 2.1 对总的数据进行切片，获取到当前批次所对应的数据
        batch_data:list[list[int]] = data[batch*sft_config.batch_size : (batch+1)*sft_config.batch_size]

        # 2.2 对该批次的数据，做一个padding
        batch_max_length = max([len(sample) for sample in batch_data])

        for sample in batch_data:
            padding_length = batch_max_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)

        batch_tensor = torch.tensor(batch_data, dtype=torch.long).to("cuda")


        # 2.3 构造labels, input_ids, assistant_answer_mask

        input_ids = batch_tensor[:,:-1]
        labels = batch_tensor[:,1:]
        assistant_answer_mask = create_answer_mask(labels=labels, tokenizer=tokenizer)


        # 3、模型前向传播：前向传播，获取lm head logits；
        # softmax(output_logits)
        output_logits = model(input_ids).logits

        # 4、损失计算：对logits和labels，使用损失函数，计算当前batch损失；

        loss = compute_loss(output_logits, labels, assistant_answer_mask)


        # 5、反向传播：反向传播，计算梯度；

        loss.backward()

        total_loss_list.append(loss.item())

        # 6、调度学习率，做参数更新：使用余弦调度器去调度学习率，再使用特定的优化器（例如AdamW）对参数进行更新。

        # 6.1 调度学习率
        
        lr = cosine_decay(batch, total_batch, sft_config.lr, sft_config.warmup_ratio)

        writer.add_scalar("train/lr",lr,batch)

        optimizer.param_groups[0]["lr"] = lr
        optimizer.step()

        optimizer.zero_grad()

        should_eval = batch % sft_config.eval_iter == 0

        should_log = batch!=0 &  batch % sft_config.log_iter == 0 

        if should_eval:
            # test_data做验证：
            current_eval_average_loss = get_eval_data_loss(model,test_data,sft_config)
            writer.add_scalar("eval/loss",current_eval_average_loss,batch)
            model.train()

        if should_log:

            last_iter_loss_list = total_loss_list[-sft_config.log_iter:]

            average_loss = sum(last_iter_loss_list) / len(last_iter_loss_list)

            writer.add_scalar("train/loss",average_loss,batch)

        progress_bar.update(1)
        progress_bar.set_postfix(lr=f"{lr:.2e}" )

    model.save_pretrained(sft_config.save_dir)
    # tokenizer save_pretrained,想让tokeneizer，相关的配置文件(tokenizer.json, tokenizer_config.json等)保存到和模型一样的目录，后面推理的时候，可以基于相同的目录，去加载模型和tokenizer
    tokenizer.save_pretrained(sft_config.save_dir)

            

if __name__=="__main__":
    sft_config = SFTConfig()
    train(sft_config)