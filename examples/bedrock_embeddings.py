"""
Amazon Bedrock Embeddings
"""

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
