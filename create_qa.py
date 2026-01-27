import os
import sys
import torch
from transformers import pipeline
import json
import random
import re

args = sys.argv

model_path = "../Meta-Llama-3.1-8B-Instruct"

pipeline = pipeline(
  "text-generation",
  model = model_path,
  tokenizer = model_path,
  dtype = torch.bfloat16,
  device_map = "auto",
)

print("complete model loading")

# テキストファイルのパス
TEXTFILEPATH = args[1]
# 質問例のファイルパス
EXAMPLEFILEPATH = args[2]
# チャンクの大きさ
CHUNK_LENGTH = 300 
# few-shot learningの質問例の数
EXAMPLE_NUM = 3

with open(TEXTFILEPATH, 'r') as fp:
  text = fp.read()

chunk_list = [text[i:i+CHUNK_LENGTH] for i in range(0, len(text), CHUNK_LENGTH)]

print("chunk list length: " + str(len(chunk_list)))

with open(EXAMPLEFILEPATH) as fp:
  style_examples = json.load(fp)

result = []

query_id = 0

for chunk in chunk_list:
  
  system_message = "You are topic extracter."
  user_message = f"""
Passage: {chunk}

与えられたパッセージを分析して、主要なトピックを特定してください。回答はキーが'topics'で値が主要なトピックの配列になるようなJSONフォーマットにしてください。例は以下の通りです。

{{
"topics": ["topic1", "topic2", "topic3"]
}}
"""

  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
  ] 

  outputs = pipeline(messages, max_new_tokens=256)
  response = outputs[0]["generated_text"][-1]["content"]

  # 生成されたjsonが想定通りの形式かチェック
  if re.match('\s*\{\s*".*"\s*:\s*\[\s*(".*",\s*)*".*"\s*\]\s*\}\s*', response) is None:
    print("skip json")
    continue

  # jsonのロードに失敗したときスキップ
  try: 
    topics = json.loads(response)

  except json.decoder.JSONDecodeError as e:
    print("JSONDecodeError: \n" + topics)
    continue

  topics = topics["topics"] 

  print("extract " + str(len(topics)) + " topics") 

  for style in style_examples:
      style_name = style["class"]
      examples = style["examples"] 
      samples = [example["question"] for example in random.sample(examples, EXAMPLE_NUM)]

      for topic in topics:
        system_message = "You are question generator."
        user_message = f"""
Passage: {chunk}

上記のパッセージには以下のトピックが含まれています。
{', '.join(topics)}

以下の質問の例を参考にして{topic}に関連する質問を一つ生成してください。

Examples:
{'\n\n'.join(samples)}

回答はキーが'question'で値が生成した質問になるようなJSONフォーマットにしてください。例は以下の通りです。

{{
"question": "question"
}}
"""

        messages = [
          {"role": "system", "content": system_message},
          {"role": "user", "content": user_message},
        ] 

        outputs = pipeline(messages, max_new_tokens=256)
        response = outputs[0]["generated_text"][-1]["content"]

        # 生成されたjsonが想定通りの形式かチェック
        if re.match('\s*\{\s*".*"\s*:\s*".*"\s*\}\s*', response) is None:
          print("skip json")
          continue

        # jsonのロードに失敗したときスキップ
        try: 
          question = json.loads(response)["question"]

        except json.decoder.JSONDecodeError as e:
          print("JSONDecodeError: \n" + topics)
          continue

        result.append({
          "id": query_id,
          "question": question,
          "document": chunk,
          "topic": topic,
          "examples": samples,
        })

        print(str(query_id) + ": " + topic + ": " + style_name + ": " + "complete generation")
        query_id += 1

result_filepath = os.path.splitext(os.path.basename(TEXTFILEPATH))[0] + '_q_ExpertGenQA.json'
with open(result_filepath, 'w') as f:
  json.dump(result, f, indent=2, ensure_ascii=False)

