from collections import defaultdict
from math import exp
from pathlib import Path
from random import Random
from time import perf_counter
from zipfile import ZipFile

import numpy as np

GENRES = ("Unknown", "Action", "Adventure", "Animation", "Children's",
          "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
          "War", "Western")
SIZE = 10
POOL = 60
DATASET = Path(__file__).with_name("ml-100k.zip")


class Model:
    def __init__(self, dataset=DATASET):
        self.movies = {}
        interactions = defaultdict(list)
        with ZipFile(dataset) as archive:
            for line in archive.read("ml-100k/u.item").decode("latin-1").splitlines():
                item, title, _, _, _, *flags = line.split("|")
                self.movies[int(item)] = (title, [g for g, f in zip(GENRES, flags) if f == "1"])
            for line in archive.read("ml-100k/u.data").decode("ascii").splitlines():
                user, item, _, timestamp = map(int, line.split())
                interactions[user].append((timestamp, item))

        self.history = {}
        self.test = {}
        matrix = np.zeros((len(interactions), len(self.movies) + 1), dtype=np.float32)
        for row, (user, events) in enumerate(sorted(interactions.items())):
            events.sort(reverse=True)
            self.test[user] = events[0][1]
            self.history[user] = [item for _, item in events[1:]]
            matrix[row, self.history[user]] = 1

        counts = matrix.sum(axis=0)
        overlap = matrix.T @ matrix
        denominator = np.sqrt(counts[:, None] * counts[None, :])
        cosine = np.divide(overlap, denominator, out=np.zeros_like(overlap),
                           where=denominator > 0)
        self.similarity = cosine * (overlap / (overlap + 10))
        np.fill_diagonal(self.similarity, 0)

        self.exposure = np.zeros((len(self.movies) + 1, len(GENRES)))
        for item, (_, genres) in self.movies.items():
            for genre in genres:
                self.exposure[item, GENRES.index(genre)] = 1 / len(genres)
        self.target = np.full(len(GENRES), SIZE / (len(GENRES) - 1))
        self.target[0] = 0

    def run(self, user, diversity=0.45, balance=0.30, seed=42, *, explain=True):
        return self.run_history(self.history[user], diversity, balance, seed, explain=explain)

    def run_history(self, history, diversity=0.45, balance=0.30, seed=42, *, explain=True):
        started = perf_counter()
        related = self.similarity[:, history]
        count = min(20, len(history))
        neighbors = np.partition(related, -count, axis=1)[:, -count:]
        scores = neighbors.sum(axis=1)
        scores[history] = -1
        scores[0] = -1
        pool = np.argsort(-scores, kind="stable")[:POOL]
        relevance = scores[pool] / max(float(scores[pool[0]]), 1e-12)
        similarity = self.similarity[np.ix_(pool, pool)]
        exposures = self.exposure[pool]
        retrieval_ms = (perf_counter() - started) * 1000

        def top():
            return range(SIZE)

        def mmr():
            remaining = set(range(POOL))
            selected = []
            for _ in range(SIZE):
                item = max(sorted(remaining), key=lambda i: relevance[i] -
                           2 * diversity * max((similarity[i, j] for j in selected), default=0))
                selected.append(item)
                remaining.remove(item)
            return selected

        def quota():
            remaining = set(range(POOL))
            selected = []
            counts = np.zeros(len(GENRES))
            for _ in range(SIZE):
                eligible = [i for i in sorted(remaining)
                            if np.all(counts + exposures[i] <= 2)]
                item = eligible[0] if eligible else min(remaining, key=lambda i: (
                    float(np.maximum(counts + exposures[i] - 2, 0).sum()), i))
                selected.append(item)
                remaining.remove(item)
                counts += exposures[item]
            return selected

        def qubo():
            # Expand squared genre deviation and distribute the pair penalty symmetrically.
            linear = -relevance + balance * (
                (exposures ** 2).sum(axis=1) - 2 * exposures @ self.target)
            pair = diversity * similarity + balance * (exposures @ exposures.T)
            np.fill_diagonal(pair, 0)
            rng = Random(seed)
            selected = list(range(SIZE))
            outside = list(range(SIZE, POOL))
            energy = float(linear[selected].sum() + pair[np.ix_(selected, selected)].sum())
            best = selected.copy()
            best_energy = energy
            for step in range(600):
                slot = rng.randrange(SIZE)
                other = rng.randrange(len(outside))
                remove, add = selected[slot], outside[other]
                delta = linear[add] - linear[remove]
                delta += 2 * (pair[add, selected].sum() - pair[add, remove]
                              - pair[remove, selected].sum())
                temperature = 0.30 * (0.005 / 0.30) ** (step / 599)
                if delta <= 0 or rng.random() < exp(-delta / temperature):
                    selected[slot], outside[other] = add, remove
                    energy += delta
                    if energy < best_energy:
                        best_energy = energy
                        best = selected.copy()
            return best

        output = {}
        for method, select in (("Top-k", top), ("MMR", mmr),
                               ("Genre quota", quota), ("QUBO", qubo)):
            started = perf_counter()
            selected = sorted(select())
            runtime_ms = retrieval_ms + (perf_counter() - started) * 1000
            items = pool[selected]
            movies = []
            for index, item in zip(selected, items):
                row = {"id": int(item)}
                if explain:
                    contributors = np.argsort(-related[item], kind="stable")[:3]
                    row.update(title=self.movies[item][0], genres=self.movies[item][1],
                               relevance=float(relevance[index]), reasons=[
                                   self.movies[history[j]][0] for j in contributors
                                   if related[item, j] > 0])
                movies.append(row)
            exposure = exposures[selected].sum(axis=0)
            output[method] = {
                "movies": movies,
                "relevance": float(relevance[selected].mean()),
                "similarity": float(similarity[np.ix_(selected, selected)].sum()
                                    / (SIZE * (SIZE - 1))),
                "genre_error": float(np.abs(exposure - self.target).sum() / (2 * SIZE)),
                "exposure": {g: round(float(value), 2) for g, value in zip(GENRES, exposure)
                             if value > 0},
                "runtime_ms": runtime_ms,
            }
        return output
