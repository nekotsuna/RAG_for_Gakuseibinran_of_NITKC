import sys
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import sqlite3
import faiss
import time


class Time_Count:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = time.time()


    def get_elapsed(self):
        return time.time() - self.start_time


    def get_elapsed_last(self):
        ret = time.time() - self.last_time
        self.last_time = time.time()

        return ret

    
    def get_time_text(self):
        return str(self.get_elapsed()) + ";" + str(self.get_elapsed_last()) + ": "


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class RAG_for_Gakuseibinran:
  def __init__(self):
    self.time = Time_Count()

    # 埋め込みモデルの構築 
    embedding_id = "intfloat/multilingual-e5-large"
    embedding_path = "../multilingual-e5-large"

    print(self.time.get_time_text() + "start loading") 

    self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_path)
    self.embedding_model = AutoModel.from_pretrained(embedding_path)

    print(self.time.get_time_text() + "load embedding model") 


    # テキスト生成モデルの構築
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path = "../Meta-Llama-3.1-8B-Instruct"

    # テキスト生成パイプラインの構築
    self.pipeline = pipeline(
      # テキスト生成を指定
      "text-generation",
      model=model_path,
      tokenizer=model_path,
      # bfloat16で量子化
      dtype = torch.bfloat16,
      # 自動的にGPUなどのデバイスを割り当て
      device_map="auto",
    )

    print(self.time.get_time_text() + "load text generation pipeline")


    # ベクトルストアの読み込み
    self.index = faiss.read_index("database/index.faiss")

    print(self.time.get_time_text() + "load index")


    # データベースの読み込み
    self.conn = sqlite3.connect("database/passage.db")
    self.cur = self.conn.cursor()

    print(self.time.get_time_text() + "load database")


  def __del__(self):
    self.conn.close()


  def to_embeddings(self, input_texts: list[str]) -> Tensor:
    batch_dict = self.embedding_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = self.embedding_model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return embeddings


  def generate(self, query, role_message, top_k):
    print(self.time.get_time_text() + "start generation")

    user_message = query
    system_message = role_message

    query_embeddings = self.to_embeddings("query: " + query)
    distance, I = self.index.search(query_embeddings.to('cpu').detach().numpy(), top_k)
    ids =  ",".join(list(map(str, I[0])))

    print(self.time.get_time_text() + "retrieve passage")

    self.cur.execute(f"SELECT text FROM passages WHERE id IN ({ids})")
    documents = self.cur.fetchall()
    for passage in documents:
      system_message += passage[0].replace('passage:', '') + '\n\n'

    messages = []

    messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    outputs = self.pipeline(messages, max_new_tokens=256, do_sample=True)

    response = outputs[0]["generated_text"][-1]["content"]

    print(self.time.get_time_text() + "complete generation")

    return response, documents


if __name__ == '__main__':
  args = sys.argv

  query = args[1]
  role_message = "あなたは木更津高専に関する質問に答えるチャットボットです。以下の文章を参考にして、ユーザーの質問に答えてください\n\n"
  top_k = 3

  rag = RAG_for_Gakuseibinran()

  response, documents = rag.generate(query, role_message, top_k) 

  print(response)

  del rag
