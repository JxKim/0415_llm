from cProfile import label

from functorch import dim
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/Qwen3-0.6B-Base")


def get_train_data(dpo_config):
    """
    获取训练数据
    """

    from datasets import load_dataset

    train_data = load_dataset("./data/ultrafeedback_binarized")["train_prefs"]

    chosen_data_token_ids = []

    rejected_data_token_ids = []

    for i in range(dpo_config.train_data_size):

        # 对于chosen数据的处理
        chosen_message_list: list[dict] = train_data[i]["chosen"]

        chosen_token_ids_list: list[int] = tokenizer.apply_chat_template(chosen_message_list, tokenize=True)["input_ids"]
        chosen_data_token_ids.append(chosen_token_ids_list)

        # 对于rejected数据的处理
        rejected_message_list: list[dict] = train_data[i]["rejected"]
        
        rejected_token_ids_list: list[int] = tokenizer.apply_chat_template(rejected_message_list, tokenize=True)["input_ids"]
        rejected_data_token_ids.append(rejected_token_ids_list)



    return chosen_data_token_ids,rejected_data_token_ids

def get_test_data(dpo_config):
    """
    获取测试数据
    """

    from datasets import load_dataset
    
    test_data = load_dataset("./data/ultrafeedback_binarized")["test_prefs"]

    chosen_data_token_ids = []

    rejected_data_token_ids = []

    for i in range(dpo_config.test_data_size):

        # 对于chosen数据的处理
        chosen_message_list: list[dict] = test_data[i]["chosen"]

        chosen_token_ids_list: list[int] = tokenizer.apply_chat_template(chosen_message_list, tokenize=True)["input_ids"]
        chosen_data_token_ids.append(chosen_token_ids_list)

        # 对于rejected数据的处理
        rejected_message_list: list[dict] = test_data[i]["rejected"]
        
        rejected_token_ids_list: list[int] = tokenizer.apply_chat_template(rejected_message_list, tokenize=True)["input_ids"]
        rejected_data_token_ids.append(rejected_token_ids_list)



    return chosen_data_token_ids,rejected_data_token_ids


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


def compute_log_prob(output_logits, labels, assistant_answer_mask):
    """
    用来计算，对数概率（被训练/参考 模型，基于prompt ，输出 chosen / rejected 回答的对数概率）
    Returns:
    """

    # 1、基于logits，做log_softmax
    # output_log_probs表示的是，模型预测下一个token的，对数概率分布，shape: batch_size, seq_len, vocab_size
    
    output_log_probs = torch.log_softmax(output_logits, dim=-1)

    # 2、找到模型输出labels当中实际token的对数概率
    # label_log_prob表示的就是，模型基于数据当中，前n个token，输出答案当中的第n+1个token的实际的对数概率，shape: (batch_size, seq_len)
    label_log_prob = torch.gather(
        output_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # masked_label_log_prob.shape : (batch_size, seq_len)
    masked_label_log_prob = label_log_prob * assistant_answer_mask


    final_log_prob = masked_label_log_prob.sum(dim=-1)

    return final_log_prob





# 假设我们的一个样本是：prompt:xxxx chosen: xxxx rejected:xxxx 
def compute_loss(chosen_log_prob, rejected_log_prob, reference_chosen_log_prob, reference_rejected_log_prob,beta):
    """
    DPO的损失计算:
    Args:
        chosen_log_prob: shape. (batch_size, ) chosen_log_prob[0]表示的就是，被训练的模型，基于当前批次的第0条样本的prompt，输出当前批次第0条样本的chosen回答的，对数概率
        rejected_log_prob: shape,(batch_size, ) rejected_log_prob[0]表示的就是，被训练的模型，基于当前批次的第0条样本的prompt，输出当前批次第0条样本的rejected回答的，对数概率
        reference_chosen_log_prob：（batch_size,） reference_chosen_log_prob[0]表示的就是，参考模型，基于当前批次的第0条样本的prompt，输出当前批次第0条样本的chosen回答的，对数概率
        reference_rejected_log_prob: (batch_size,) reference_rejected_log_prob[0]表示的就是，参考模型，基于当前批次的第0条样本的prompt，输出当前批次第0条样本的rejected回答的，对数概率
        beta: 超参数
    """
    
    margin = chosen_log_prob- rejected_log_prob - (reference_chosen_log_prob - reference_rejected_log_prob)

    # loss: shape，(batch_size, )
    loss = -1 * torch.nn.functional.logsigmoid(beta * margin)

    average_loss = loss.mean()
    return average_loss


from dataclasses import dataclass

@dataclass
class DPOConfig:

    lr:float = 3e-6 # DPO全参微调：1e-6 - 5e-6 ， 

    train_data_size:int = 20000 # 训练样本数量，在get_train_data()方法当中，会调用

    batch_size:int = 4

    warmup_ratio:float = 0.1

    eval_iter:int = 100

    test_data_size:int = 500 

    log_dir:str = "./logs/03_dpo_demo"

    log_iter:int = 100

    save_dir:str = "./finetuned/03_dpo_demo"

    beta:float = 0.1



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
        

def get_eval_data_loss(model,ref_model,test_chosen_data,test_rejected_data,dpo_config):
    """
    训练过程当中，评估:
    1、遍历test_data，每次使用一小批次，给到model，让模型做前向传播
    2、基于logits, labels, assistant_answer_mask，算得当前批次的损失
    3、算得所有批次的平均损失，
    """
    model.eval()
    total_batch = (len(test_chosen_data) + dpo_config.batch_size - 1) // dpo_config.batch_size
    all_batch_loss = []
    for batch in range(total_batch):

        # 2、基于chosen_data 准备，chosen_input_ids, chosen_labels, chosen_assistant_answer_mask
        # 2.1 对总的数据进行切片，获取到当前批次所对应的数据
        chosen_batch_data:list[list[int]] = test_chosen_data[batch*dpo_config.batch_size : (batch+1)*dpo_config.batch_size]

        # 2.2 对该批次的数据，做一个padding
        chosen_batch_max_length = max([len(sample) for sample in chosen_batch_data])

        for sample in chosen_batch_data:
            padding_length = chosen_batch_max_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)

        chosen_batch_tensor = torch.tensor(chosen_batch_data, dtype=torch.long).to("cuda")


        # 2.3 构造labels, input_ids, assistant_answer_mask

        chosen_input_ids = chosen_batch_tensor[:,:-1]
        chosen_labels = chosen_batch_tensor[:,1:]
        chosen_assistant_answer_mask = create_answer_mask(labels=chosen_labels, tokenizer=tokenizer)



        # 3、基于rejected_data 准备，rejected_input_ids, rejected_labels, rejected_assistant_answer_mask
        # 2.1 对总的数据进行切片，获取到当前批次所对应的数据
        rejected_batch_data:list[list[int]] = test_rejected_data[batch*dpo_config.batch_size : (batch+1)*dpo_config.batch_size]

        # 2.2 对该批次的数据，做一个padding
        rejected_batch_max_length = max([len(sample) for sample in rejected_batch_data])

        for sample in rejected_batch_data:
            padding_length = rejected_batch_max_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)

        rejected_batch_tensor = torch.tensor(rejected_batch_data, dtype=torch.long).to("cuda")


        # 2.3 构造labels, input_ids, assistant_answer_mask

        rejected_input_ids = rejected_batch_tensor[:,:-1]
        rejected_labels = rejected_batch_tensor[:,1:]
        rejected_assistant_answer_mask = create_answer_mask(labels=rejected_labels, tokenizer=tokenizer)

        # 3、模型前向传播：4次

        with torch.no_grad():
            # 第一次：被训练的模型，基于chosen数据，算得logits
            chosen_output_logits = model(chosen_input_ids).logits
            # 第二次：被训练的模型，基于rejected数据，算得logits
            rejected_output_logits = model(rejected_input_ids).logits
            # 第三次：参考模型，基于chosen数据，算得的logits
            reference_chosen_output_logits  = ref_model(chosen_input_ids).logits
            # 第四次：参考模型，基于rejected数据，算得的logits
            reference_rejected_output_logits = ref_model(rejected_input_ids).logits


        # 调用compute_log_probs，算得四个对数概率

        # 被训练模型，基于prompt，输出chosen答案的对数概率
        chosen_log_prob  = compute_log_prob(chosen_output_logits,chosen_labels,chosen_assistant_answer_mask)


        # 被训练模型，基于prompt，输出rejected答案的对数概率
        rejected_log_prob  = compute_log_prob(rejected_output_logits,rejected_labels,rejected_assistant_answer_mask)


        # 参考模型，基于prompt，输出chosen答案的对数概率
        reference_chosen_log_prob  = compute_log_prob(reference_chosen_output_logits,chosen_labels,chosen_assistant_answer_mask)


        # 参考模型，基于prompt，输出rejected答案的对数概率

        reference_rejected_log_prob  = compute_log_prob(reference_rejected_output_logits,rejected_labels,rejected_assistant_answer_mask)


        loss = compute_loss(chosen_log_prob,rejected_log_prob,reference_chosen_log_prob,reference_rejected_log_prob,dpo_config.beta)

        all_batch_loss.append(loss)

    return sum(all_batch_loss) / len(all_batch_loss)




def train(dpo_config:DPOConfig):
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
    model = AutoModelForCausalLM.from_pretrained("./finetuned/02_sft_demo") # DPO是基于SFT之后的模型去做的进一步训练
    ref_model = AutoModelForCausalLM.from_pretrained("./finetuned/02_sft_demo")

    model.to("cuda")
    model.train()

    ref_model.to("cuda")
    ref_model.eval()

    # 大模型微调，使用的AdamW，而不使用Adam, AdamW是对Adam的一个优化：将梯度更新和权重衰减，这个过程做了一个解耦
    optimizer = AdamW(model.parameters(),lr=dpo_config.lr)
    # 获取训练用的token_ids
    chosen_data,rejected_data = get_train_data(dpo_config)
    test_chosen_data, test_rejected_data = get_test_data(dpo_config)
    # 如果len(data)能够被dpo_config.batch_size 整除， 如果len(data) 不能够被dpo_config.batch_size 整除，可能会余1,2,.. batch_size-1 + batch_size-1 最大可能是2batch_size-2
    total_batch = (len(chosen_data) + dpo_config.batch_size - 1) // dpo_config.batch_size

    # 
    writer = SummaryWriter(log_dir=dpo_config.log_dir)
    progress_bar = tqdm(total=total_batch)

    total_loss_list = []
    for batch in range(total_batch):

        # 2、基于chosen_data 准备，chosen_input_ids, chosen_labels, chosen_assistant_answer_mask
        # 2.1 对总的数据进行切片，获取到当前批次所对应的数据
        chosen_batch_data:list[list[int]] = chosen_data[batch*dpo_config.batch_size : (batch+1)*dpo_config.batch_size]

        # 2.2 对该批次的数据，做一个padding
        chosen_batch_max_length = max([len(sample) for sample in chosen_batch_data])

        for sample in chosen_batch_data:
            padding_length = chosen_batch_max_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)

        chosen_batch_tensor = torch.tensor(chosen_batch_data, dtype=torch.long).to("cuda")


        # 2.3 构造labels, input_ids, assistant_answer_mask

        chosen_input_ids = chosen_batch_tensor[:,:-1]
        chosen_labels = chosen_batch_tensor[:,1:]
        chosen_assistant_answer_mask = create_answer_mask(labels=chosen_labels, tokenizer=tokenizer)



        # 3、基于rejected_data 准备，rejected_input_ids, rejected_labels, rejected_assistant_answer_mask
        # 2.1 对总的数据进行切片，获取到当前批次所对应的数据
        rejected_batch_data:list[list[int]] = rejected_data[batch*dpo_config.batch_size : (batch+1)*dpo_config.batch_size]

        # 2.2 对该批次的数据，做一个padding
        rejected_batch_max_length = max([len(sample) for sample in rejected_batch_data])

        for sample in rejected_batch_data:
            padding_length = rejected_batch_max_length - len(sample)
            sample.extend([tokenizer.pad_token_id] * padding_length)

        rejected_batch_tensor = torch.tensor(rejected_batch_data, dtype=torch.long).to("cuda")


        # 2.3 构造labels, input_ids, assistant_answer_mask

        rejected_input_ids = rejected_batch_tensor[:,:-1]
        rejected_labels = rejected_batch_tensor[:,1:]
        rejected_assistant_answer_mask = create_answer_mask(labels=rejected_labels, tokenizer=tokenizer)

        # 3、模型前向传播：4次

        # 第一次：被训练的模型，基于chosen数据，算得logits
        chosen_output_logits = model(chosen_input_ids).logits

        # 第二次：被训练的模型，基于rejected数据，算得logits
        rejected_output_logits = model(rejected_input_ids).logits

        with torch.no_grad():
            # 第三次：参考模型，基于chosen数据，算得的logits
            reference_chosen_output_logits  = ref_model(chosen_input_ids).logits
            # 第四次：参考模型，基于rejected数据，算得的logits
            reference_rejected_output_logits = ref_model(rejected_input_ids).logits


        # 调用compute_log_probs，算得四个对数概率

        # 被训练模型，基于prompt，输出chosen答案的对数概率
        chosen_log_prob  = compute_log_prob(chosen_output_logits,chosen_labels,chosen_assistant_answer_mask)


        # 被训练模型，基于prompt，输出rejected答案的对数概率
        rejected_log_prob  = compute_log_prob(rejected_output_logits,rejected_labels,rejected_assistant_answer_mask)


        # 参考模型，基于prompt，输出chosen答案的对数概率
        reference_chosen_log_prob  = compute_log_prob(reference_chosen_output_logits,chosen_labels,chosen_assistant_answer_mask)


        # 参考模型，基于prompt，输出rejected答案的对数概率

        reference_rejected_log_prob  = compute_log_prob(reference_rejected_output_logits,rejected_labels,rejected_assistant_answer_mask)


        loss = compute_loss(chosen_log_prob,rejected_log_prob,reference_chosen_log_prob,reference_rejected_log_prob,dpo_config.beta)


        # 5、反向传播：反向传播，计算梯度；

        loss.backward()

        total_loss_list.append(loss.item())

        # 6、调度学习率，做参数更新：使用余弦调度器去调度学习率，再使用特定的优化器（例如AdamW）对参数进行更新。

        # 6.1 调度学习率
        
        lr = cosine_decay(batch, total_batch, dpo_config.lr, dpo_config.warmup_ratio)

        writer.add_scalar("train/lr",lr,batch)

        optimizer.param_groups[0]["lr"] = lr
        optimizer.step()

        optimizer.zero_grad()

        should_eval = batch % dpo_config.eval_iter == 0

        should_log = batch!=0 and  batch % dpo_config.log_iter == 0 

        if should_eval:
            # test_data做验证：
            current_eval_average_loss = get_eval_data_loss(model,ref_model, test_chosen_data,test_rejected_data,dpo_config)
            writer.add_scalar("eval/loss",current_eval_average_loss,batch)
            model.train()

        if should_log:

            last_iter_loss_list = total_loss_list[-dpo_config.log_iter:]

            average_loss = sum(last_iter_loss_list) / len(last_iter_loss_list)

            writer.add_scalar("train/loss",average_loss,batch)

        progress_bar.update(1)
        progress_bar.set_postfix(lr=f"{lr:.2e}" )

    model.save_pretrained(dpo_config.save_dir)
    # tokenizer save_pretrained,想让tokeneizer，相关的配置文件(tokenizer.json, tokenizer_config.json等)保存到和模型一样的目录，后面推理的时候，可以基于相同的目录，去加载模型和tokenizer
    tokenizer.save_pretrained(dpo_config.save_dir)

            

if __name__=="__main__":
    dpo_config = DPOConfig()
    train(dpo_config)