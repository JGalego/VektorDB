# pylint: disable=invalid-name,redefined-outer-name
"""
Core implementation
"""

# Standard imports
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Union

# Library imports
import numpy as np

from prettytable import PrettyTable

# Local imports
from .ann import is_left, split_nodes
from .types import SimilarityScore, Vector
from .utils import cosine_similarity


class VectorDatabase(ABC):
    """Abstract vector database class"""

    def __init__(self):
        self.vectors = defaultdict(Vector)

    def insert(self, key: str, vector: Union[List[float],np.ndarray,Vector]) -> None:
        """Inserts a new key/vector pair into the database."""
        if isinstance(vector, Vector):
            pass
        elif isinstance(vector, np.ndarray):
            vector = Vector(vector)
        elif isinstance(vector, list):
            vector = Vector(np.array(vector))
        else:
            raise ValueError("Expected vector of type Vector, list or np.ndarray!")

        assert vector.data.ndim == 1, f"✋ Expected 1D array (Got: {vector.data.ndims}D)!"
        self.vectors[key] = vector

    def retrieve(self, key: str) -> Vector:
        """Returns the vector indexed by a given key."""
        return self.vectors.get(key, None)

    def dump(self) -> Dict:
        """Dumps the entire vector database"""
        return dict(self.vectors)

    def display(  # pylint: disable=dangerous-default-value
        self,
        keys: Union[List,None]=None,
        np_format: Dict={
            'precision': 4,
            'threshold': 5,
            'suppress': False
        },
        pt_format: Dict={
			'max_width': 100
        }) -> None:
        """Prints the entire database"""

        # Handle vector formatting via NumPy
        # https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
        np.set_printoptions(**np_format)

		# and let PrettyTable handle the rest
        dbt = PrettyTable(['Key', 'Data', 'Metadata'], **pt_format)

        if keys is not None:
            for key in keys:
                vector = self.dump()[key]
                dbt.add_row([key, vector.data, vector.metadata])
        else:
            for key, vector in self.vectors.items():
                dbt.add_row([key, vector.data, vector.metadata])

        print(dbt)

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        n: int, distance: callable = cosine_similarity) -> List[Tuple[Vector, float]]:
        """Returns the top n results for a given query and their similarity scores."""


class NaiveVectorDatabase(VectorDatabase):
    """A very simple and naive vector database implementation"""

    def search(
        self,
        query: np.ndarray,
        n: int, distance: callable = cosine_similarity) -> List[Tuple[int, float]]:

        # Calculate the similarity scores for each entry in the database
        similarities = [
            SimilarityScore(key, distance(query, vector.data))
                for key, vector in self.vectors.items()
        ]

        # Sort the similarity scores in ascending order
        similarities.sort(key=lambda entry: entry.score)
        return similarities[:n]


class ANNVectorDatabase(VectorDatabase):
    """Annoy-like vector database backed by ANN search"""

    def __init__(self):
        super().__init__()
        self.trees = []

    def build(self, n_trees: int=1, k: int=10):
        """Builds a forest of n_trees trees, where each node
        is a random split with at most k items."""
        for _ in range(n_trees):
            self.trees.append(split_nodes(self.vectors, max_num_items=k))

    def search(
        self,
        query: np.ndarray,
        n: int, distance: callable = cosine_similarity) -> List[Tuple[int, float]]:
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
            SimilarityScore(key, distance(query, vector.data))
                for node in nodes
                    for key, vector in node.values.items()
        ]
        similarities = list(set(similarities))
        similarities.sort(key=lambda entry: entry.score)
        return similarities[:n]
