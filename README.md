# VektorDB 🚧

A minimal [vector database](https://aws.amazon.com/what-is/vector-databases/) for educational purposes.

> **Tagline:** keep it simple and they will learn...

**Want to learn more?** Check out the 'References' section below 👇

<img src="vektordb.png" width="30%"/>

## Example

```python
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
vector_db.display(keys=map(str, range(5)), precision=4, threshold=5)

# Build inner structure
vector_db.build(n_trees=3, k=5)

# Search for similar vectors
query = np.random.rand(VEC_DIM)
results = vector_db.search(query, 5)
print_similarity_scores(results)

# Retrieve specific vectors
for key in np.random.permutation(NUM_RECORDS)[:NUM_RECORDS // 10]:
    vector_db.retrieve(str(key))
```

## References

### Articles & Books 📚

- (Bernhardsson, 2015a) [Nearest neisghbor methods and vector models – part 1](https://erikbern.com/2015/09/24/nearest-neighbor-methods-vector-models-part-1)
- (Bernhardsson, 2015b) [Nearest neighbors and vector models - part 2 - algorithms and data structures](https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html)
- (Bruch, 2024) [Foundations of Vector Retrieval](https://arxiv.org/abs/2401.09350)
- (Manning, Raghavan & Schütze, 2008) [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
- (Pan, Wang & Li, 2023) [Survey of Vector Database Management Systems](https://arxiv.org/abs/2310.14021)
- (Teofili, 2019) [Deep Learning for Search](https://www.manning.com/books/deep-learning-for-search)

### Courses 👩‍🏫

- `COS 597A` (Princeton): [Long Term Memory in AI - Vector Search and Databases](https://edoliberty.github.io/vector-search-class-notes/)
- `CMU 15-445/645` (Carnegie Mellon): [Database Systems](https://15445.courses.cs.cmu.edu/fall2024/)

### Links 🌐

- (Superlinked) [Vector DB Comparison](https://superlinked.com/vector-db-comparison)
- [Awesome Vector Database](https://github.com/dangkhoasdc/awesome-vector-database) [![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)