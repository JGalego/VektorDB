"""
Amazon Bedrock Embeddings
"""

# Standard imports
import json
import random

# Library imports
import boto3

from datasets import load_dataset
from tqdm import tqdm

# VektorDB imports
from vektordb import ANNVectorDatabase
from vektordb.types import Vector
from vektordb.utils import print_similarity_scores

#############
# Constants #
#############

# Text call limit for Cohere Embed models via Amazon Bedrock
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
MAX_TEXTS_PER_CALL = 96

# Number of samples we want to process
N_SAMPLES = 100

##################
# Amazon Bedrock #
##################

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

########
# Main #
########

# Set the seed value
random.seed(42)

# Load dataset
# https://huggingface.co/datasets/openai/gsm8k
ds = load_dataset("openai/gsm8k", "main", split="train")[:N_SAMPLES]
questions = ds['question']
answers = ds['answer']

# Initialize database
vector_db = ANNVectorDatabase()

# Generate embeddings
embeddings = []
for idx in tqdm(range(0, len(answers), MAX_TEXTS_PER_CALL), "Generating embeddings"):
    embeddings += embed(answers[idx:idx+MAX_TEXTS_PER_CALL])

# Load embeddings into the database
for idx in tqdm(range(len(embeddings)), "Loading embeddings"):
    vector_db.insert(idx, Vector(embeddings[idx], {'answer': answers[idx][:20]}))

# Print a small sample of the database
vector_db.display(
    np_format={
        'edgeitems': 1,
        'precision': 5,
        'threshold': 3,
        'suppress': True
    },
    keys=range(10)
)

# Build inner tree structure for ANN search
vector_db.build(n_trees=3, k=3)
print(vector_db.trees[0], "\n")

# Run a query
query = questions[0]
print("\nQuery:", query, "\n")
results = vector_db.search(embed([query])[0], 3)
print_similarity_scores(results)
