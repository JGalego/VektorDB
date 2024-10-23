"""
Approximate Nearest Neighbors (ANN) search module

Inspired by
https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html
"""

# Standard imports
import random

from collections import defaultdict

# Local imports
from .types import Node, Vector
from .utils import find_hyperplane, is_left

def split_nodes(vectors: defaultdict[Vector], max_num_items: int=5) -> Node:
    """Splits nodes into an hierarchical tree structure."""
    # Pick two random points
    k1, k2 = random.sample(list(vectors), 2)

    # Find the hyperplane equidistant from those two vectors
    hyperplane = find_hyperplane(vectors[k1].data, vectors[k2].data)

    # Split vectors into left and right nodes
    left_nodes = {}
    right_nodes = {}
    for k, v in vectors.items():
        if is_left(v.data, hyperplane=hyperplane):
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
