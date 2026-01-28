import sys
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import faiss
import sqlite3

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def to_embeddings(input_texts: list[str], tokenizer, model) -> Tensor:
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return embeddings

embedding_id = "intfloat/multilingual-e5-large"
embedding_path = "../multilingual-e5-large"

embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_path)
embedding_model = AutoModel.from_pretrained(embedding_path)

args = sys.argv

# db化するテキストファイルのパス
FILEPATH = args[1]
# チャンクの大きさ
CHUNK_LENGTH = 100

chunk_list = []

with open(FILEPATH, 'r') as fp:
  text = fp.read()

# テキストファイルをチャンクに分ける
# multilingual-e5-large用に[passage:]プレフィックスを追加
chunk_list = ["passage: " + text[i:i+CHUNK_LENGTH] for i in range(0, len(text), CHUNK_LENGTH)]


# faissを用いたベクトルストアの構築
storename = "database/index.faiss"
dimention = 1024
index = faiss.IndexFlatL2(dimention)

print("start embedding")
# メモリ節約のために10個ずつ変換
for i in range(0, len(chunk_list), 10):
  passage_embeddings = to_embeddings(chunk_list[i:i+10], embedding_tokenizer, embedding_model)
  # tensorをndarrayに変換
  numpy_embeddings = passage_embeddings.to('cpu').detach().numpy()

  index.add(numpy_embeddings)

  # メモリ節約のために、メモリを解放
  del passage_embeddings
  del numpy_embeddings

  print(str(i) + "-" + str(i + 10) + " complete")

faiss.write_index(index, storename)


# sqliteを用いたdatabaseの構築
dbname = "database/passage.db"
conn = sqlite3.connect(dbname)
cur = conn.cursor()

cur.execute('CREATE TABLE IF NOT EXISTS passages(id INTEGER PRIMARY KEY, text TEXT);')

cur.execute('DELETE FROM passages')
cur.executemany(
    'INSERT INTO passages (id, text) VALUES (?, ?)', list(enumerate(chunk_list))
)

conn.commit()
conn.close()
