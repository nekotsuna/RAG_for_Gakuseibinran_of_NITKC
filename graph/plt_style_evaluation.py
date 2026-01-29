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

rag_what_is = [int(row["RAG_evaluation"]) for row in evaluation if row["style"] == "what-is"]
rag_how_to = [int(row["RAG_evaluation"]) for row in evaluation if row["style"] == "how-to"]
rag_scenario_base = [int(row["RAG_evaluation"]) for row in evaluation if row["style"] == "scenario-base"]

rag_what_is_count = [rag_what_is.count(i) / len(rag_what_is) for i in range(3)]
rag_how_to_count = [rag_how_to.count(i) / len(rag_how_to) for i in range(3)]
rag_scenario_base_count = [rag_scenario_base.count(i) / len(rag_scenario_base) for i in range(3)]

position = np.arange(3)
width = 0.2
labels = ["×", "△", "○"]

plt.bar(position, [rag_what_is_count[0], rag_how_to_count[0], rag_scenario_base_count[0]], width=width, label=labels[0]) 
plt.bar(position + width, [rag_what_is_count[1], rag_how_to_count[1], rag_scenario_base_count[1]], width=width, label=labels[1]) 
plt.bar(position + width*2, [rag_what_is_count[2], rag_how_to_count[2], rag_scenario_base_count[2]], width=width, label=labels[2]) 
plt.xticks(position + width, ["what-is", "how-to", "scenario-base"])
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), ncol=3)
plt.savefig("style_evaluation.png")
plt.show()
