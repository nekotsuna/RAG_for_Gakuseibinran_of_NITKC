import os
import sys
from torch import Tensor
import torch
from transformers import pipeline
import json

if __name__ == '__main__':
    args = sys.argv

    dataset_filepath = args[1] 
    role_message = "あなたは木更津高専に関する質問に答えるチャットボットです。以下の文章を参考にして、ユーザーの質問に答えてください\n\n"

    result = []
    qa = []

    with open(dataset_filepath) as f:
        qa = json.load(f)

    model_path = "../Meta-Llama-3.1-8B-Instruct"

    # テキスト生成パイプラインの構築
    pipeline = pipeline(
      # テキスト生成を指定
      "text-generation",
      model=model_path,
      tokenizer=model_path,
      # bfloat16で量子化
      dtype = torch.bfloat16,
      # 自動的にGPUなどのデバイスを割り当て
      device_map="auto",
    )

    for query_set in qa:
        query_id = query_set['id']
        query = query_set['question']
        answer = query_set['answer']

        messages = []
        messages.append({"role": "system", "content": role_message})
        messages.append({"role": "user", "content": query}) 
        outputs = pipeline(messages, max_new_tokens=256, do_sample=False)
        response = outputs[0]["generated_text"][-1]["content"]

        result.append({
          "id"       : query_id,
          "query"    : query,
          "answer"   : answer,
          "response" : response,
        })

        print("query " + str(query_id) + " complete")

    result_filepath = os.path.splitext(os.path.basename(dataset_filepath))[0] + '_model_result.json'
    with open(result_filepath, 'w') as f:
      json.dump(result, f, indent=2, ensure_ascii=False)
