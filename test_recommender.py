import unittest
from pathlib import Path
from recommender import Model

class RecommenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model(Path(__file__).with_name("ml-100k.zip"))

    def test_split(self):
        self.assertEqual(len(self.model.history), 943)
        self.assertEqual(len(self.model.movies), 1682)
        self.assertEqual(sum(map(len, self.model.history.values())), 99057)
        for user, held_out in self.model.test.items():
            self.assertNotIn(held_out, self.model.history[user])

    def test_lists(self):
        self.assertEqual(self.model.movies[1][0], "Toy Story (1995)")
        shortest = min(self.model.history, key=lambda user: len(self.model.history[user]))
        for user in (1, 42, 943, shortest):
            first = self.model.run(user)
            second = self.model.run(user)
            for method in first:
                ids = [movie["id"] for movie in first[method]["movies"]]
                self.assertEqual(len(set(ids)), 10)
                self.assertFalse(set(ids).intersection(self.model.history[user]))
                self.assertEqual(first[method]["movies"], second[method]["movies"])


    def test_custom_profile(self):
        favorites = [1, 71, 95]
        results = self.model.run_history(favorites)
        for result in results.values():
            ids = {movie["id"] for movie in result["movies"]}
            self.assertEqual(len(ids), 10)
            self.assertFalse(ids.intersection(favorites))
        known = set(self.model.movies[item][0] for item in favorites)
        explained = [reason for row in results["QUBO"]["movies"] for reason in row["reasons"]]
        self.assertTrue(explained)
        self.assertTrue(set(explained).issubset(known))

if __name__ == "__main__":
    unittest.main()
