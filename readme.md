## 数据来源和处理

数据来源是hugging face上的2个数据集：stingning/ultrachat, lmsys_chat_englist_dataset
由于2个数据集加起来有点多，本人资金有限，因此对数据集进行了一些过滤（根据语言，轮数，对话总字数，是否被编辑等条件），并统一会话格式，第一次处理后得到的数据为merge，有大约45W条。

之后为了将数据方便转换为llama 2的聊天微调格式：

```
<s>[INST] <<SYS>>
{your_system_message}
<</SYS>>

{user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]
```

又进行了一次处理，并将处理后的数据切分为了训练集（train）和测试集(test)。其中train有
约20w条数据,test有5w条数据。最后只取了10W条数据做训练，最终的数据是small_train_tokens和small_test_tokens。(参考 data2tokens.ipynb)

处理的数据百度云链接:    [data](https://pan.baidu.com/s/1nZdcSNwyImRjKU39u7b3Qw?pwd=2h4d)  
train和test可以通过 2-train-test-data.ipynb运行得到。  
train_tokens，test_tokens运行 3-data2tokens.ipynb运行得到。

## 训练

先安装环境:
pip install -r requirementx.txt

在4块A800-80G上运行:

```
torchrun --nproc_per_node=4 train.py \
    --data_path ./small_train_tokens \
    --eval_data_path ./small_test_tokens \
    --bf16 True \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "steps" \
    --eval_steps 250 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```
大概需要10h。train.py代码为参考stanford_alpaca的指令微调代码而来，针对聊天数据做了更改。  

## 测试

```
python test.py --path your-fine-tuning-model-path
```

## 备注

聊天数据采用了和llama-2-chat相同的聊天格式。如果是基于llama-2进行聊天微调，是可以采用任何一种聊天的格式的。
如果是基于llama-2-chat进行微调的，那么采用上述的方式是能够从预训练模型中受益的。  

参考: https://medium.com/@xuebinbin12/fine-tuning-chat-based-llm-with-multi-turn-conversational-data-part-i-d8c64d01a20d
