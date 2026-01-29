import sys
import csv 
import numpy as np
import matplotlib.pyplot as plt

FILEPATH = "evaluation.csv"

evaluation = []

with open(FILEPATH) as fp:
    reader = csv.DictReader(fp)

    for row in reader:
        evaluation.append(row) 

model_eval = [int(row["model_evaluation"]) for row in evaluation]
rag_eval = [int(row["RAG_evaluation"]) for row in evaluation]

model_count = [model_eval.count(i) for i in range(3)]
rag_count = [rag_eval.count(i) for i in range(3)]

position = np.arange(2)
width = 0.2
labels = ["×", "△", "○"]

plt.bar(position, [model_count[0], rag_count[0]], width=width, label=labels[0]) 
plt.bar(position + width, [model_count[1], rag_count[1]], width=width, label=labels[1]) 
plt.bar(position + width*2, [model_count[2], rag_count[2]], width=width, label=labels[2]) 
plt.xticks(position + width, ["model only", "RAG"])
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), ncol=3)
plt.savefig("evaluation.png")
plt.show()
