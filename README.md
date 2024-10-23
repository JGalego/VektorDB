# VektorDB 🚧

A minimal [vector database](https://aws.amazon.com/what-is/vector-databases/) for educational purposes.

> **Tagline:** keep it simple and they will learn...

**Want to learn more?** Check out the 'References' section below 👇

<img src="vektordb.png" width="30%"/>

## Example

```python
# Standard imports
import json

# Library imports
import boto3

from datasets import load_dataset
from tqdm import tqdm

from vektordb import ANNVectorDatabase
from vektordb.types import Vector
from vektordb.utils import print_similarity_scores

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime")

def embed(texts: list, model_id="cohere.embed-english-v3"):
    """Generates embeddings for an array of strings using Cohere Embed models."""
    model_provider = model_id.split('.')[0]
    assert model_provider == "cohere", \
        f"Invalid model provider (Got: {model_provider}, Expected: cohere)"

    # Prepare payload
    accept = "*/*"
    content_type = "application/json"
    body = json.dumps({
        'texts': texts,
        'input_type': "search_document"
    })

    # Call model
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    # Process response
    response_body = json.loads(response.get('body').read())
    return response_body.get('embeddings')


# Load dataset
# https://huggingface.co/datasets/openai/gsm8k
ds = load_dataset("openai/gsm8k", "main", split="train")[:96]
questions = ds['question']
answers = ds['answer']

# Initialize database
vector_db = ANNVectorDatabase()

# Insert records
embeddings = embed(answers)
for idx in tqdm(range(len(embeddings)), "Loading embeddings"):
    vector_db.insert(idx, Vector(embeddings[idx], {'answer': answers[idx][:20]}))

# Print database
vector_db.display(
    np_format={
        'edgeitems': 1,
        'precision': 5,
        'threshold': 3,
        'suppress': True
    },
    keys=range(10)
)

# Build inner tree structure
vector_db.build(n_trees=3, k=3)
print(vector_db.trees[0], "\n")

# Search query
query = questions[0]
print("\nQuery:", query, "\n")
results = vector_db.search(embed([query])[0], 3)
print_similarity_scores(results)

```

**Output:**

```
+-----+-------------------------+------------------------------------+
| Key |           Data          |              Metadata              |
+-----+-------------------------+------------------------------------+
|  0  | [-0.00618 ... -0.00047] | {'answer': 'Natalia sold 48/2 = '} |
|  1  | [-0.01997 ... -0.01791] | {'answer': 'Weng earns 12/60 = $'} |
|  2  | [-0.00623 ... -0.0061 ] | {'answer': 'In the beginning, Be'} |
|  3  | [-0.07849 ...  0.00721] | {'answer': 'Maila read 12 x 2 = '} |
|  4  | [-0.01669 ...  0.01263] | {'answer': 'He writes each frien'} |
|  5  |  [0.02484 ... 0.05185]  | {'answer': 'There are 80/100 * 1'} |
|  6  | [-0.01807 ... -0.01859] | {'answer': 'He eats 32 from the '} |
|  7  | [ 0.01265 ... -0.02016] | {'answer': 'To the initial 2 pou'} |
|  8  | [-0.00504 ...  0.0143 ] | {'answer': 'Let S be the amount '} |
|  9  | [-0.0239  ... -0.00905] | {'answer': 'She works 8 hours a '} |
+-----+-------------------------+------------------------------------+

                                  ___________________________96__________________________________
                                 /                                                               \
                         _______59_____________                                            _____37_
                        /                      \                                          /        \
                _______29_____             ___30____                                  ___32_       5_
               /              \           /         \                                /      \     /  \
       _______20_____        _9     _____14_      _16__                          ___24_     8_    1  4
      /              \      /  \   /        \    /     \                        /      \   /  \     / \
    _13_____      ___7     _6  3   8_       6    5    11___           _________18_     6   2  6     2 2
   /        \    /    \   /  \    /  \     / \  / \  /     \         /            \   / \    / \
  _6     ___7    6_   1   5  1    2  6_    3 3  2 3  3    _8       _14_           4   3 3    3 3
 /  \   /    \  /  \     / \        /  \                 /  \     /    \         / \
 4  2   5_   2  2  4     2 3        2  4                 5  3    _5    9_____    1 3
/ \    /  \       / \                 / \               / \     /  \  /      \
2 2    1  4       3 1                 2 2               2 3     4  1  2     _7
         / \                                                   / \         /  \
         1 3                                                   3 1        _5  2
                                                                         /  \
                                                                         4  1
                                                                        / \
                                                                        2 2


Query: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
How many clips did Natalia sell altogether in April and May?

+-----+---------------------+
| Key |        Score        |
+-----+---------------------+
|  0  | 0.15148634752350043 |
|  88 |  0.6031578104905267 |
|  4  |  0.6684896968705925 |
+-----+---------------------+
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