from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
print(torch.__version__)

# Load the dataset
dataset = load_dataset("conceptofmind/facebook_ads_160k", split="train")

# Save the original length
original_length = len(dataset)

# Get the number of GPUs available
num_gpus = torch.cuda.device_count()

# Initialize SentenceTransformer models for each GPU
models = [SentenceTransformer("sentence-transformers/sentence-t5-xxl").to(f'cuda:{i}') for i in range(num_gpus)]

def embed_text(examples):
    # Determine which GPU this batch will be sent to
    gpu_id = torch.tensor(range(len(examples))).fmod_(num_gpus)

    embeddings = []
    for ex, id in zip(examples["text"], gpu_id):
        # Get the embeddings for the text
        embeddings.append(models[id].encode(ex).tolist())

    return {"embeddings": embeddings}

dataset = dataset.map(embed_text, batched=True, batch_size=16)

# Compare the lengths
assert original_length == len(dataset), "The datasets do not have the same length."