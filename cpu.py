from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = "Hugging_Face_token"

# 載入分詞器
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=hf_token)

# 確保分詞器有填充標記
tokenizer.pad_token = tokenizer.eos_token  # 將 EOS 標記用作填充標記

# 載入模型
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=hf_token)

# 準備專業的半導體相關輸入文本
input_text = "介紹半導體晶圓的製造過程和主要挑戰。"

# 編碼輸入文本並自動處理填充和生成注意力掩碼
# return_tensors='pt': 指定返回的張量格式為 PyTorch 張量。
# padding=True: 自動添加填充，以使所有輸入序列長度一致。
# truncation=True: 若輸入長度超過最大長度限制，將進行截斷。
# max_length=50: 設置輸入序列的最大長度為50。
encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=50)

# 生成文本
# max_length=1000: 在生成文本時，設置生成序列的最大長度為1000。
# num_return_sequences=1: 生成文本的數量為1。
# no_repeat_ngram_size=2: 避免生成文本中出現重複的二元組。
# early_stopping=True: 如果滿足停止條件（如達到最大長度），提前停止生成。
outputs = model.generate(**encoded_input, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

# 解碼並輸出生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
