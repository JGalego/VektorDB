"""
Data types
"""

# Standard imports
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Union

# Library imports
import numpy as np
import numpy.typing as npt

@dataclass
class Vector:
    """Represents a vector entry to our database"""
    data: npt.ArrayLike
    metadata: Union[Dict,None] = None

    def __init__(self, data: npt.ArrayLike, metadata: Union[Dict,None] = None):
        self.data = np.array(data)
        self.metadata = metadata


@dataclass
class Node:
    """Represents a random split of the hyperplane"""
    values: defaultdict[np.ndarray]
    instances: int
    left: Union['Node',None] = None
    right: Union['Node',None] = None
    hyperplane: Union[np.ndarray,None] = None


@dataclass
class SimilarityScore:
    """Represents a query result"""
    key: Union[int, str]
    score: float

    def __hash__(self):
        return hash(self.key)
