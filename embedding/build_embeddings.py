from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
from tqdm import tqdm
print(torch.__version__)

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("conceptofmind/facebook_ads", split="train")

    model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')

    sentences = dataset["text"]

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    print("Start Embedding")
    #embeddings = model.encode(sentences)

    #Compute the embeddings using the multi-process pool
    embeddings = model.encode_multi_process(sentences, pool, batch_size=16)
    print("Embeddings computed. Shape:", embeddings.shape)

    print("Finished Embedding")

    embeddings_list = embeddings.tolist()

    fdataset = dataset.add_column("embeddings", embeddings_list)
