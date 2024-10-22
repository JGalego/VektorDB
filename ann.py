"""
Approximate Nearest Neighbors (ANN) search module

Inspired by
https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html
"""

# Standard imports
import random

from collections import defaultdict
from dataclasses import dataclass
from typing import Union

# Library imports
import numpy as np

@dataclass
class Node:
    """Represents a random split of the hyperplane"""
    values: defaultdict[np.ndarray]
    instances: int
    left: Union['Node',None] = None
    right: Union['Node',None] = None
    hyperplane: Union[np.ndarray,None] = None

def find_hyperplane(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Finds the hyperplane equidistant from two n-dimensional vectors"""

    # Find the midpoint between the two vectors
    midpoint = (v1 + v2) / 2

    # Find the direction vector and normalize it
    direction_vector = v2 - v1
    normal_vector = direction_vector / np.linalg.norm(direction_vector)

    # Calculate the distance between the midpoint and the hyperplane
    distance = np.dot(midpoint, normal_vector)

    # Define the equation of the hyperplane
    hyperplane = np.concatenate((normal_vector, [distance]))

    return hyperplane


def is_left(v, hyperplane):
    """Returns true if a given vector is to the left of an hyperplane."""
    # Calculate the signed distance from v to the hyperplane
    signed_distance = np.dot(hyperplane[:-1], v) - hyperplane[-1]

    if signed_distance <= 0:
        return True
    return False


def split_nodes(vectors: defaultdict[np.ndarray], max_num_items: int=5) -> Node:
    """Splits nodes into an hierarchical tree structure."""
    # Pick two random points
    k1, k2 = random.sample(list(vectors), 2)

    # Find the hyperplane equidistant from those two vectors
    hyperplane = find_hyperplane(vectors[k1], vectors[k2])

    # Split vectors into left and right nodes
    left_nodes = {}
    right_nodes = {}
    for k, v in vectors.items():
        if is_left(v, hyperplane=hyperplane):
            left_nodes[k] = v
        else:
            right_nodes[k] = v

	# Initialize current node
    current_node = Node(
        hyperplane=hyperplane,
        values=vectors,
        instances=len(vectors)
    )

	# Process left subtree
    if len(left_nodes) > max_num_items:
        current_node.left = split_nodes(
            left_nodes,
            max_num_items=max_num_items
        )
    else:
        current_node.left = Node(
            values=left_nodes,
            instances=len(left_nodes)
        )

	# Process right subtree
    if len(right_nodes) > max_num_items:
        current_node.right = split_nodes(
            right_nodes,
            max_num_items=max_num_items,
        )
    else:
        current_node.right = Node(
            values=right_nodes,
            instances=len(right_nodes)
        )

    return current_node
