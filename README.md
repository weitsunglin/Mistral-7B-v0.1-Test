# Model Card for Mistral-7B-v0.1

The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. 
Mistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks we tested.

For full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/announcing-mistral-7b/).

## Model Architecture

Mistral-7B-v0.1 is a transformer model, with the following architecture choices:
- Grouped-Query Attention: 這種注意力機制將輸入分成多個群組，並分別對每個群組進行處理。這樣可以降低計算複雜度，同時保持對關鍵信息的關注。此技術有助於模型在處理大規模數據時，更有效地學習和提取特徵。
  ![Grouped-Query](https://github.com/weitsunglin/Mistral-7B-v0.1/blob/main/Grouped-Query%20Attention.png)
- Sliding-Window Attention: 此技術通過限制模型在特定大小的滑動窗口內執行注意力機制，從而提高效率和處理速度。這使得模型能夠專注於輸入中的局部特徵，並在長文本處理中顯著提升性能。
 ![Sliding-Window](https://github.com/weitsunglin/Mistral-7B-v0.1/blob/main/Sliding-Window%20Attention.png)
- Byte-fallback BPE tokenizer: 遇到未知或罕見字詞時，分詞器會回退到字節級別進行處理，這有助於改善模型對於非標準或新奇文本的處理能力，特別是在多語言環境中更是如此。
 ![Byte-fallback BPE](https://github.com/weitsunglin/Mistral-7B-v0.1/blob/main/Byte-fallback%20BPE%20tokenizer.jpg)


## Requirement

- my environment python3.9, pytorch & cuda12.4 &　30gb ram　＆ i7-13700 & gtx 1060

- model: https://huggingface.co/mistralai/Mistral-7B-v0.1

- cpu version & gpu version can inference  & prediction this model

- huggingface token

## Demo

- cpu inference  & prediction

 ![cpu](https://github.com/weitsunglin/Mistral-7B-v0.1/blob/main/cpu.jpg)

- gpu inference  & prediction
  
 ![gpu](https://github.com/weitsunglin/Mistral-7B-v0.1/blob/main/gpu.jpg)

 gpu version is  faster than cpu version lol
