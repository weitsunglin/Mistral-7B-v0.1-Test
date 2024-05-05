from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys  # 引入 sys 模塊來退出腳本

print(torch.cuda.is_available())

# 檢查是否有可用的GPU，如果沒有則退出程序
if not torch.cuda.is_available():
    print("CUDA is not available. Exiting...")
    sys.exit()  # 退出腳本

device = torch.device("cuda")

hf_token = "Hugging_Face_token"

# 加載分詞器
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=hf_token)

# 確保分詞器具有填充符號
tokenizer.pad_token = tokenizer.eos_token  # 使用 EOS 符號作為填充符號

# 加載模型
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=hf_token)
# 將模型移至GPU
model.to(device)

# 準備輸入文本
input_text = "介紹transformers"


# 編碼輸入文本並自動處理填充和生成注意力遮罩
# return_tensors='pt': 指定返回的張量格式為 PyTorch 張量。
# padding=True: 自動添加填充，以使所有輸入序列長度一致。
# truncation=True: 若輸入長度超過最大長度限制，將進行截斷。
# max_length=50: 設置輸入序列的最大長度為50。
encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=50)
# 將輸入張量移至GPU
encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}

# 生成文本
# max_length=1000: 在生成文本時，設置生成序列的最大長度為1000。
# num_return_sequences=1: 生成文本的數量為1。
# no_repeat_ngram_size=2: 避免生成文本中出現重複的二元組。
# early_stopping=True: 如果滿足停止條件（如達到最大長度），提前停止生成。
outputs = model.generate(**encoded_input, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=3, temperature=0.7, top_p=0.9, early_stopping=True)

# 解碼並輸出生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
