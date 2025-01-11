
# SBERT Article Similarity Script

This repository contains a Python script (`sbert_script.py`) that computes semantic similarity between a set of articles using **Sentence-BERT (SBERT)**. Each article is described by a *title* (`Titre`) and a *meta description* (`Meta`), which are combined into a single text input for embedding generation.

## How It Works

1. **Reading the Dataset**  
   The script reads a CSV file (`data_set_articles_list.csv` by default) that must include the columns:
   - `Titre` (title of the article)  
   - `Meta` (short description of the article)  
   - `Slug` (optional unique identifier)  
   - `Groupe` (optional category or theme)

2. **Combining Title and Meta**  
   For each article, the script concatenates `Titre` and `Meta` into a new column called `combined_text`, such as:
   ```python
   combined_text = Titre + ". " + Meta
   ```
   This text is what SBERT will encode.

3. **Generating Embeddings with SBERT**  
   The script uses the **sentence-transformers** library and a default model, e.g. `"all-MiniLM-L6-v2"`, to encode each `combined_text` into a numerical vector (embedding).

4. **Computing Similarities**  
   After generating embeddings, the script calculates a cosine similarity matrix among all articles. For each article, it retrieves the top `k` similar articles (by default, `k=5`).

5. **Output**  
   The script writes a new CSV file (by default `output_sbert.csv`) that keeps all original columns plus an additional column named **`similar_titles`**, which contains the top `k` recommended articles for each row.

## Why SBERT?

There are multiple approaches to semantic similarity (e.g., traditional BERT, RoBERTa, LLaMA with embeddings, etc.). Here’s why this script focuses on SBERT:

1. **Sentence-Level Focus:**  
   SBERT is optimized for producing embeddings at the sentence or short-text level, perfect for titles and brief descriptions.

2. **Efficiency & Quality:**  
   SBERT typically runs faster and yields strong performance for similarity tasks, making it a good balance of speed and accuracy compared to larger language models.

3. **Ease of Use:**  
   With the **sentence-transformers** library, it’s straightforward to install, load a model, and quickly obtain embeddings without extensive configurations or specialized hardware.

In contrast:

- **Traditional BERT** requires special pooling strategies to derive sentence embeddings.  
- **RoBERTa** offers strong performance but often has larger models and may be slower.  
- **LLaMA or GPT-based embeddings** can provide context-aware vectors but require specialized set-ups (like Ollama or Hugging Face Transformers with large model weights). They may also be slower or have more complex licensing.

For many projects, SBERT provides an excellent combination of **ease-of-use**, **efficiency**, and **quality**, especially if you need quick sentence or short-paragraph embeddings in a multi-lingual context.

## Quick Start

### Install Dependencies

```bash
pip install pandas torch numpy scikit-learn sentence-transformers
```

### Run the Script

```bash
python sbert_script.py --input_csv data_set_articles_list.csv --output_csv output_sbert.csv --k 5
```

- `--input_csv`: Path to your CSV file  
- `--output_csv`: Name of the resulting CSV file  
- `--k`: Number of similar articles to retrieve per article

### Check the Output

Open `output_sbert.csv` and review the `similar_titles` column to see which articles are recommended for each row.

## Contributing

Feel free to open issues or submit pull requests if you have ideas for improvements.

- **Adding new models**: You can change the model in `sbert_script.py` by modifying the line:
  ```python
  model_name = "all-MiniLM-L6-v2"
  ```
  to another pre-trained model from the [Sentence-Transformers Model Hub](https://www.sbert.net/docs/pretrained_models.html).

- **Batch size**: To handle larger datasets more efficiently, you might consider modifying the script to encode texts in batches rather than all at once.

## License

This project is released under the MIT License. Feel free to use it or adapt it for your own needs.
# sbert_similars_articles
