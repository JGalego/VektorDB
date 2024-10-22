# pylint: disable=invalid-name,redefined-outer-name
r"""
____   ____      __      __              ________ __________
\   \ /   /____ |  | ___/  |_  __________\______ \\______   \
 \   Y   // __ \|  |/ /\   __\/  _ \_  __ \    |  \|    |  _/
  \     /\  ___/|    <  |  | (  <_> )  | \/    `   \    |   \
   \___/  \___  >__|_ \ |__|  \____/|__| /_______  /______  /
              \/     \/                          \/       \/
"""

# Standard imports
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple

# Library imports
import numpy as np

# Local imports
from ann import is_left, split_nodes

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class VectorDatabase(ABC):
    """Abstract vector database class"""

    def __init__(self):
        self.vectors = defaultdict(np.ndarray)

    def insert(self, key: str, vector: np.ndarray) -> None:
        """Inserts a new key/vector pair into the database."""
        self.vectors[key] = vector

    def retrieve(self, key: str) -> np.ndarray:
        """Returns the vector indexed by a given key."""
        return self.vectors.get(key, None)

    @abstractmethod
    def search(self, query: np.ndarray, n: int) -> List[Tuple[str, float]]:
        """Returns the top n results for a given query."""


class NaiveVectorDatabase(VectorDatabase):
    """A very simple vector database implementation"""

    def search(self, query: np.ndarray, n: int):
        similarities = [
            (key, cosine_similarity(query, vector))
                for key, vector in self.vectors.items()
        ]
        similarities.sort(key=lambda entry: entry[1], reverse=True)
        return similarities[:n]


class ANNVectorDatabase(VectorDatabase):
    """Simplified Annoy-like vector database backed by ANN search"""

    def __init__(self):
        super().__init__()
        self.trees = []

    def build(self, n_trees=1, k=10):
        """Builds a forest of n_trees trees, where each node
        is a random split with at most k items."""
        for _ in range(n_trees):
            self.trees.append(split_nodes(self.vectors, max_num_items=k))

    def search(self, query: np.ndarray, n: int):
        assert self.trees, "⛔ Don't forget to call build before search!"

        nodes = []
        for tree in self.trees:
            node = tree
            while node.instances >= n and node.hyperplane is not None:
                if is_left(query, node.hyperplane) is True:
                    node = node.left
                else:
                    node = node.right
            nodes.append(node)

        # Return similaries so we can compare
        # the results with the naive implementation
        similarities = [
            (key, cosine_similarity(query, vector))
                for node in nodes
                    for key, vector in node.values.items()
        ]
        similarities = list(set(similarities))
        similarities.sort(key=lambda entry: entry[1], reverse=True)
        return similarities[:n]

if __name__ == '__main__':

    NUM_RECORDS = 1000
    VEC_DIM = 100

    # Set random seed
    np.random.seed(42)

    # Initialize the vector databases
    vector_db = ANNVectorDatabase()

    # Insert records
    for key in range(NUM_RECORDS):
        vector_db.insert(str(key), np.random.rand(VEC_DIM))

    # Build inner structure
    if isinstance(vector_db, ANNVectorDatabase):
        vector_db.build(n_trees=3)

    # Search for similar vectors
    query = np.random.rand(VEC_DIM)
    vector_db.search(query, NUM_RECORDS // 10)

    # Retrieve specific vectors
    for key in np.random.permutation(NUM_RECORDS)[:NUM_RECORDS // 10]:
        vector_db.retrieve(str(key))
