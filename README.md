# VektorDB 🚧

A minimal [vector database](https://aws.amazon.com/what-is/vector-databases/) for educational purposes.

> **Tagline:** keep it simple and they will learn...

**Want to learn more?** Check out the 'References' section below 👇

<img src="vektordb.png" width="30%"/>

## Example

```python
import json

import boto3

from datasets import load_dataset

from vektordb import ANNVectorDatabase
from vektordb.types import Vector
from vektordb.utils import print_similarity_scores

# Load dataset
# https://huggingface.co/datasets/openai/gsm8k
ds = load_dataset("openai/gsm8k", "main", split="train")[:10]
questions = ds['question']
answers = ds['answer']

# Initialize database
vector_db = ANNVectorDatabase()

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime")

def embed(texts: list, model_id="cohere.embed-english-v3"):
    """Generates embeddings for an array of strings using Cohere Embed models."""
    model_provider = model_id.split('.')[0]
    assert model_provider == "cohere", \
        f"Invalid model provider (Got: {model_provider}, Expected: cohere)"

    # Prepare payload
    accept = '*/*'
    content_type = 'application/json'
    body = json.dumps({
        "texts": texts,
        "input_type": "search_document"
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

# Insert data
for idx, embeddings in enumerate(embed(answers)):
    vector_db.insert(idx, Vector(embeddings, {'answer': answers[idx][:20]}))

# Print database
vector_db.display(
    np_format={
        'edgeitems': 1,
        'precision': 5,
        'threshold': 3,
        'suppress': True
    }
)

# Build inner tree structure
vector_db.build(n_trees=3, k=3)

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
|  0  | [-0.00704 ... -0.00034] | {'answer': 'Natalia sold 48/2 = '} |
|  1  | [-0.01981 ... -0.01781] | {'answer': 'Weng earns 12/60 = $'} |
|  2  | [-0.00595 ... -0.00608] | {'answer': 'In the beginning, Be'} |
|  3  | [-0.0791  ...  0.00657] | {'answer': 'Maila read 12 x 2 = '} |
|  4  | [-0.01663 ...  0.01281] | {'answer': 'He writes each frien'} |
|  5  |  [0.02496 ... 0.05203]  | {'answer': 'There are 80/100 * 1'} |
|  6  | [-0.0184  ... -0.01807] | {'answer': 'He eats 32 from the '} |
|  7  | [ 0.01228 ... -0.02016] | {'answer': 'To the initial 2 pou'} |
|  8  | [-0.00512 ...  0.01397] | {'answer': 'Let S be the amount '} |
|  9  | [-0.02402 ... -0.00873] | {'answer': 'She works 8 hours a '} |
+-----+-------------------------+------------------------------------+

Query: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

+-----+---------------------+
| Key |        Score        |
+-----+---------------------+
|  0  | 0.15227363589742293 |
|  8  |  0.6226389000890077 |
|  4  |  0.6686830772608823 |
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