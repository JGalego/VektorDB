# VektorDB 🏹

A minimal [vector database](https://aws.amazon.com/what-is/vector-databases/) for educational purposes.

> **Tagline:** keep it simple and they will learn...

**Want to learn more?** Check out the 'References' section below 👇

<img src="vektordb.png" width="30%"/>

## Example: Amazon Bedrock meets VektorDB

In this example, we'll use the [Grade School Math 8K](https://huggingface.co/datasets/openai/gsm8k) (GSM8K) dataset

```python
from datasets import load_dataset

# Number of samples we want to process
N_SAMPLES = 100

# Load dataset
# https://huggingface.co/datasets/openai/gsm8k
ds = load_dataset("openai/gsm8k", "main", split="train")[:N_SAMPLES]
questions = ds['question']
answers = ds['answer']
```

which contains "high quality linguistically diverse grade school math word problems" in the form of `question-answer` pairs like the one shown below

```
### Question

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
How many clips did Natalia sell altogether in April and May?

### Answer

Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
```

Our goal is to turn these `question-answer` pairs into embeddings, store them in VektorDB and perform some operations.

> 💡 **Embeddings** are just numerical representations of a piece of information, usually in the form of vectors. You can turn any kind of data into embeddings (e.g. [💬](https://huggingface.co/blog/getting-started-with-embeddings) [🖼️](https://www.pinecone.io/learn/series/image-search/) [🔊](https://huggingface.co/blog/cappuch/audio-embedding-wtf) [🎞️](https://github.com/iejMac/clip-video-encode) [🦠](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1)) and they'll *preserve* the meaning of the original data. **If you want to learn more about embeddings**, check out [Mapping Embeddings: from meaning to vectors and back](https://jgalego.github.io/MappingEmbeddings).

<img src="https://miro.medium.com/v2/resize:fit:2000/1*SYiW1MUZul1NvL1kc1RxwQ.png" width="70%"/>

Let's define a helper function to call [Cohere Embed](https://aws.amazon.com/bedrock/cohere-command-embed/) models via [Amazon Bedrock](https://aws.amazon.com/bedrock/)

```python
import json
import boto3

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
```

and use it to generate embeddings for a small subset of our data (answers only, for now)

```python
from tqdm import tqdm

# Text call limit for Cohere Embed models via Amazon Bedrock
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
MAX_TEXTS_PER_CALL = 96

embeddings = []
for idx in tqdm(range(0, len(answers), MAX_TEXTS_PER_CALL), "Generating embeddings"):
    embeddings += embed(answers[idx:idx+MAX_TEXTS_PER_CALL])
```

We are now ready to initialize VektorDB and start loading data

```python
from vektordb import ANNVectorDatabase
from vektordb.types import Vector

# Initialize database
vector_db = ANNVectorDatabase()

# Load embeddings into the database
for idx in tqdm(range(len(embeddings)), "Loading embeddings"):
    vector_db.insert(idx, Vector(embeddings[idx], {'answer': answers[idx][:20]}))
```

As a sanity check, we can print a small sample of our database

```python
vector_db.display(
    np_format={
        'edgeitems': 1,
        'precision': 5,
        'threshold': 3,
        'suppress': True
    },
    keys=range(10)
)
```

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
```

Our VektorDB instance is backed by an implementation of [Approximate Nearest Neighbors](https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6) (ANN) search that uses [binary trees](https://en.wikipedia.org/wiki/Binary_tree) to represent different partitions/splits of the hyperspace.

These partitions are generated by picking two vectors at random, finding the hyperplane equidistant between the two and then splitting the other points into `left` and `right` depending on which side they're on

<img src="https://erikbern.com/assets/2015/09/tree-1-1024x793.png" width="50%"/>

This process is repeated until we have *at most* `k` items in each node (partition)

<img src="https://erikbern.com/assets/2015/09/tree-full-K-1024x793.png" width="50%"/>

We can get better results by generating a *forest of trees** 🌳 and searching all of them, so let's do that:

```python
import random

# Set seed value for replication
random.seed(42)

# Plant a bunch of trees 🏞️
vector_db.build(n_trees=3, k=3)
print(vector_db.trees[0], "\n")
```

Here's a representation of the first tree in our *forest* (the nodes show the number of instances in each partition)

```
                                                       __________100______________
                                                      /                           \
     ________________________________________________63______          __________37___________
    /                                                        \        /                       \
  _51__                                                    _12_      16____               ___21____
 /     \                                                  /    \    /      \             /         \
 6    45____________________                             _7    5    2    _14___        _10_      _11_____
/ \  /                      \                           /  \  / \       /      \      /    \    /        \
3 3  3           __________42_____________              4  3  2 3       5     _9     _5    5    4     ___7
                /                         \            / \             / \   /  \   /  \  / \  / \   /    \
            ___18____               _____24____        2 2             3 2   6  3   4  1  2 3  3 1   6_   1
           /         \             /           \                            / \    / \              /  \
          _8_      _10_        ___11___      _13_                           3 3    1 3              2  4
         /   \    /    \      /        \    /    \                                                    / \
         4   4    4    6_     6_      _5    5    8_                                                   3 1
        / \ / \  / \  /  \   /  \    /  \  / \  /  \
        2 2 3 1  1 3  2  4   2  4    4  1  2 3  3  5
                        / \    / \  / \           / \
                        1 3    3 1  2 2           2 3
```

Finally, we can run a query by simply searching the database for answers *similar* to a target question.

We use [distance functions](https://weaviate.io/blog/distance-metrics-in-vector-search) like the ones shown below to quantify how similar two vectors are to one another.

<img src="https://miro.medium.com/v2/resize:fit:1200/1*FTVRr_Wqz-3_k6Mk6G4kew.png" width="30%"/>

For instance, if we ask the first question in our training dataset

```python
from vektordb.utils import print_similarity_scores

# Extract first question
query = questions[0]
print("\nQuery:", query, "\n")

# Run search and display similarity scores
results = vector_db.search(embed([query])[0], 3)
print_similarity_scores(results)
```

we expect the answer with the same index (`0`) to be the top result:

```
Query: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.
How many clips did Natalia sell altogether in April and May?

+-----+---------------------+
| Key |        Score        |
+-----+---------------------+
|  0  | 0.15148634752350043 |
|  15 |  0.6105711817572272 |
|  83 |  0.6823805943068366 |
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