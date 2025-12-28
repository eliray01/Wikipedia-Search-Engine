# Wikipedia RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot powered by English Wikipedia that combines classical **BM25 retrieval** with modern **large-language-model (LLM)** answer generation.  
The system retrieves relevant Wikipedia documents and uses **[Qwen2.5‑3B‑Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)** to generate accurate answers based solely on the retrieved context.

<p align="center">
  <img src="https://github.com/eliray01/Wikipedia-Search-Engine/blob/main/.github/assets/demo.gif" alt="demo" width="600">
</p>

---

## Features

|                                       | Details |
|---------------------------------------|---------|
| **Retrieval model**                   | BM25 with adjustable *k₁* & *b* (defaults 1.5 / 0.75) |
| **Corpus**                            | 2023‑11‑01 English Wikipedia dump (6.4 M articles) |
| **Query augmentation**                | 5 paraphrases per query from Qwen2.5‑3B‑Instruct |
| **RAG pipeline**                      | Retrieves top 5 documents and generates context-based answers |
| **Web interface**                     | Single-page chatbot interface with real-time progress |
| **Offline usage**                     | Pre‑computed toy index ships in `precomputed_data/` |

---

## Quick start (toy demo)

```bash
git clone https://github.com/eliray01/Wikipedia-Search-Engine.git
cd Wikipedia-Search-Engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # ~2 GB incl. PyTorch + Transformers
python app3_sep.py               # ▶︎ open http://localhost:5000
```

The bundled `precomputed_data/` folder contains an **extremely small subset** of Wikipedia so that the demo starts instantly on a laptop. Expect only a handful of results per query.  
To index the **full 6.4 M‑article dump** follow the steps below.

---

## Building the full BM25 index

> **Hardware**: 16 GB RAM minimum (32 GB recommended) – the script streams the dump but stores all tokenised articles in RAM before serialising.

```bash
# (inside the repo root)
python bm25_prepare_data.py   # Note: dataset only loads if precomputed_data doesn't exist
```

The script will
1. download the 2023‑11‑01 English Wikipedia split via 🤗 `datasets` (only if `precomputed_data/` is empty),
2. clean & tokenise each article,
3. compute per‑document term frequencies and global IDF values,
4. save three pickles in `precomputed_data/`:
   - `docs.pkl` (article metadata, text content, and term counts)
   - `avgdl.pkl` (average document length)
   - `idf_values.pkl` (global IDF table)

All later components (Flask UI) will automatically load these files on startup.  ([github.com](https://github.com/eliray01/Wikipedia-Search-Engine/blob/main/bm25_prepare_data.py))

---

## How the RAG pipeline works

When a user asks a question, the system:

1. **Query Augmentation**: Generates *N = 5* semantically‑equivalent rewrites with Qwen2.5‑3B‑Instruct using an instruct‑style prompt.  ([github.com](https://github.com/eliray01/Wikipedia-Search-Engine/blob/main/augment.py))
2. **Tokenization**: Tokenises the original query **and** every rewrite.
3. **Document Retrieval**: Forms the union of all tokens to retrieve a **candidate set** of documents from the inverted index (built at launch).
4. **BM25 Scoring**: Scores each candidate with BM25 and retrieves the **top 5 documents**.
5. **Answer Generation**: Uses Qwen2.5‑3B‑Instruct with a RAG prompt that includes the retrieved documents as context to generate an accurate answer based solely on the provided documents.

The RAG approach ensures answers are grounded in the retrieved Wikipedia documents, preventing hallucination and providing source attribution.

---

## Running the Flask web app

```bash
export PYTHONPATH=$PWD  # so that augment.py can be imported
python app3_sep.py      # serves on http://127.0.0.1:5000
```

Key routes  ([github.com](https://github.com/eliray01/Wikipedia-Search-Engine/blob/main/app3_sep.py))

| Endpoint            | Method | Purpose                            |
|---------------------|--------|------------------------------------|
| `/`                 |  GET   | Chatbot interface                  |
| `/start_search`     |  POST  | Kick‑off backend retrieval & generation thread  |
| `/progress`         |  GET   | Server‑sent events for progress    |
| `/get_answer`       |  GET   | JSON response with answer and sources |

The UI provides a single-page chatbot interface with a progress bar while the backend retrieves documents and generates answers. Answers appear in the chat with source document links.

---

## Repository layout

```
.
├── app3_sep.py           # Flask web front‑end, BM25 logic & RAG pipeline
├── augment.py            # Qwen‑powered query augmentation & answer generation
├── bm25_prepare_data.py  # Wikipedia dump ➜ pickled BM25 index
├── precomputed_data/     # ⬅︎ toy example index (replace for full)
├── templates/            # Jinja2 HTML templates (chatbot interface)
└── requirements.txt      # Python dependencies
```

---

## Tips & troubleshooting

* **GPU acceleration** – `augment.py` automatically places the Qwen model on the first CUDA device if available. Without a GPU expect ~2–5 s per query (augmentation + generation).
* **Cold‑start** – The model loads once at startup; subsequent queries are faster.
* **RAM errors while indexing** – Set `subset_size` in `bm25_prepare_data.py` to a smaller value or increase swap.
* **Windows** – Replace the `source .venv/bin/activate` command with `.venv\Scripts\activate`.
* **Double process issue** – The reloader is disabled (`use_reloader=False`) to prevent loading the model twice. Restart manually after code changes.

---

## License

Code is released under the MIT license (see `LICENSE`). Wikipedia content is available under **CC BY‑SA 3.0** & **GFDL**. Qwen models are released under their respective licenses

---

## Acknowledgements

* [Wikipedia](https://www.wikipedia.org/) contributors for the open corpus.
* The [🤗 Transformers](https://github.com/huggingface/transformers) and [🤗 Datasets](https://github.com/huggingface/datasets) teams.
* [Qwen](https://huggingface.co/Qwen) authors for open‑sourcing the 2.5‑series models.
