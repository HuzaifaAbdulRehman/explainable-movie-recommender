import json
from collections import defaultdict
from math import log2
from pathlib import Path

from recommender import Model


def evaluate(model):
    totals = defaultdict(lambda: defaultdict(float))
    for user, target in model.test.items():
        for method, result in model.run(user, explain=False).items():
            ids = [movie["id"] for movie in result["movies"]]
            assert len(set(ids)) == 10 and not set(ids).intersection(model.history[user])
            if target in ids:
                totals[method]["recall@10"] += 1
                totals[method]["ndcg@10"] += 1 / log2(ids.index(target) + 2)
            for metric in ("relevance", "similarity", "genre_error", "runtime_ms"):
                totals[method][metric] += result[metric]
    metrics = ("recall@10", "ndcg@10", "relevance", "similarity", "genre_error", "runtime_ms")
    return {method: {metric: round(values[metric] / len(model.test), 6)
                     for metric in metrics} for method, values in totals.items()}


if __name__ == "__main__":
    model = Model()
    result = {"users": len(model.test), "diversity": 0.45, "balance": 0.30,
              "seed": 42, "methods": evaluate(model)}
    rendered = json.dumps(result, indent=2)
    Path(__file__).with_name("results.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
