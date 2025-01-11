import argparse
import logging

import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity matrix from a tensor of embeddings.
    embeddings: shape (N, dim)
    Returns a tensor (N, N).
    """
    # L2 normalization
    norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return torch.mm(norm, norm.T)  # matrix multiplication

def get_top_k_similar(sim_matrix: torch.Tensor, data: pd.DataFrame, k=5) -> pd.DataFrame:
    """
    For each row (article), find the k articles with the highest cosine similarity,
    ignoring the article itself.
    Returns a copy of 'data' with a new column 'similar_titles'.
    """
    titles = data["Titre"].tolist()
    sim_np = sim_matrix.cpu().numpy()  # move to CPU / NumPy

    top_k_list = []
    for i in range(len(titles)):
        row = sim_np[i]
        # Sort descending
        sorted_indices = row.argsort()[::-1]
        # Exclude the item itself
        sorted_indices = sorted_indices[sorted_indices != i]
        # Top k
        k_indices = sorted_indices[:k]
        similar_titles = [titles[idx] for idx in k_indices]
        top_k_list.append(similar_titles)
    
    out_df = data.copy()
    out_df["similar_titles"] = top_k_list
    return out_df

def find_similar_sbert(data: pd.DataFrame, k=5) -> pd.DataFrame:
    """
    Generates embeddings with SBERT, computes the cosine similarity,
    and returns a DataFrame with the column 'similar_titles'.
    """
    logging.info("[SBERT] Loading the SBERT model...")
    model_name = "all-MiniLM-L6-v2"  # or any other Sentence-BERT model
    model_sbert = SentenceTransformer(model_name)
    
    texts = data["combined_text"].tolist()
    logging.info("[SBERT] Encoding texts...")
    embeddings = model_sbert.encode(texts, convert_to_tensor=True)
    
    if torch.cuda.is_available():
        embeddings = embeddings.to("cuda")
    
    logging.info("[SBERT] Computing the cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    result_df = get_top_k_similar(sim_matrix, data, k)
    return result_df

def main():
    parser = argparse.ArgumentParser(description="SBERT script for computing article similarity.")
    parser.add_argument("--input_csv", type=str, default="data_set_articles_list.csv",
                        help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, default="output_sbert.csv",
                        help="Name of the output CSV file.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of similar articles to retrieve.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG mode.")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(message)s")

    logging.info(f"Reading CSV file: {args.input_csv}")
    data = pd.read_csv(args.input_csv)
    
    data["combined_text"] = data["Titre"].astype(str) + ". " + data["Meta"].astype(str)
    
    logging.info("=== SBERT ===")
    df_sbert = find_similar_sbert(data, k=args.k)
    df_sbert.to_csv(args.output_csv, index=False)
    logging.info(f"SBERT file generated: {args.output_csv}")
    
    logging.info("Done. Check the output CSV to see the 'similar_titles' column.")

if __name__ == "__main__":
    main()
