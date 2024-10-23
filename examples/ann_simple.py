"""
Simple ANN
"""

import numpy as np

from vektordb import ANNVectorDatabase
from vektordb.utils import print_similarity_scores

NUM_RECORDS = 1000
VEC_DIM = 100

# Set random seed
np.random.seed(42)

# Initialize database
vector_db = ANNVectorDatabase()

# Insert records
for key in range(NUM_RECORDS):
    vector_db.insert(str(key), np.random.rand(VEC_DIM))

# Print database
vector_db.display(keys=map(str, range(5)))

# Build inner structure
vector_db.build(n_trees=3, k=5)

# Search for similar vectors
query = np.random.rand(VEC_DIM)
results = vector_db.search(query, 5)
print_similarity_scores(results)

# Retrieve specific vectors
for key in np.random.permutation(NUM_RECORDS)[:NUM_RECORDS // 10]:
    vector_db.retrieve(str(key))
