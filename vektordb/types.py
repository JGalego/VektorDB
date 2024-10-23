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

    def display(self):  # pylint: disable=too-many-locals
        """Returns list of strings, width, height, and horizontal coordinate of the root.

        Adapted from
        https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
        """
        # No children
        if self.right is None and self.left is None:
            line = f"{self.instances}"
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child
        if self.right is None:
            lines, n, p, x = self.left.display()
            s = f"{self.instances}"
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child
        if self.left is None:
            lines, n, p, x = self.right.display()
            s = f"{self.instances}"
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children
        left, n, p, x = self.left.display()
        right, m, q, y = self.right.display()
        s = f"{self.instances}"
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def __str__(self):
        lines, *_ = self.display()
        return "\n".join(lines)

@dataclass
class SimilarityScore:
    """Represents a query result"""
    key: Union[int, str]
    score: float

    def __hash__(self):
        return hash(self.key)
