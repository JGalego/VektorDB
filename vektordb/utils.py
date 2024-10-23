"""
Utilities and helper functions
"""

# Standard Import
from typing import List

# Library imports
import numpy as np

from prettytable import PrettyTable

# Local imports
from .types import SimilarityScore

# Distance functions 📏
#
# 🙋 Want to learn more? Check out the following references
# https://weaviate.io/blog/distance-metrics-in-vector-search
# https://www.maartengrootendorst.com/blog/distances/

def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the Manhattan (L1) distance between two vectors."""
    return np.linalg.norm(v1 - v2, ord=1)


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the Euclidean (L2) distance between two vectors"""
    # NOTE: ord defaults to 2, so there's no need to change it
    return np.linalg.norm(v1 - v2)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the cosine similarity between two vectors."""
    return 1.0 - float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# LinAlg stuff 🧮
#
# 💡 Need a refresher? Take a look at 18.06SC (MIT) | Linear Algebra
# https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/

def find_hyperplane(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Finds the hyperplane equidistant from two n-dimensional vectors"""

    # Find the midpoint between the two vectors
    midpoint = (v1 + v2) / 2

    # Find the direction vector and normalize it
    direction_vector = v2 - v1
    normal_vector = direction_vector / np.linalg.norm(direction_vector)

    # Calculate the distance between the midpoint and the hyperplane
    distance = np.dot(midpoint, normal_vector)

    # Define the equation of the hyperplane (w^T x + b)
    hyperplane = np.concatenate((normal_vector, [distance]))

    return hyperplane


def is_left(v: np.ndarray, hyperplane: np.ndarray) -> bool:
    """Returns true if a given vector is to the left of an hyperplane."""
    # Calculate the signed distance from v to the hyperplane
    signed_distance = np.dot(hyperplane[:-1], v) - hyperplane[-1]

    if signed_distance <= 0:
        return True
    return False


# Helper functions

def print_similarity_scores(results: List[SimilarityScore]):
    """Prints a list of similarity scores as a table"""
    sst = PrettyTable(['Key', 'Score'])
    for result in results:
        sst.add_row([result.key, result.score])
    print(sst)
