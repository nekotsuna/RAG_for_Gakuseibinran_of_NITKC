import os
import sys
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import sqlite3
import faiss
import json

from rag import RAG_for_Gakuseibinran

if __name__ == '__main__':
  args = sys.argv 

  dataset_filepath = args[1] 
  role_message = "あなたは木更津高専に関する質問に答えるチャットボットです。以下の文章を参考にして、ユーザーの質問に答えてください\n\n"
  top_k = 3
  result = []
  qa = []

  with open(dataset_filepath) as f:
    qa = json.load(f)

  rag = RAG_for_Gakuseibinran()

  for query_set in qa: 
    query_id = query_set['id']
    query = query_set['question']
    answer = query_set['answer']
    response, documents = rag.generate(query, role_message, top_k) 
  
    result.append({
      "id"       : query_id,
      "query"    : query,
      "answer"   : answer,
      "response" : response,
      "documents": documents,
    })

    print("query " + str(query_id) + " complete")

  result_filepath = os.path.splitext(os.path.basename(dataset_filepath))[0] + '_rag_result.json'
  with open(result_filepath, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

  del rag
