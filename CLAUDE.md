# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Outfit Vibe Recommender: takes a Pinterest board (a folder of images the user saves manually), extracts a
"vibe" embedding with FashionCLIP, recommends visually similar clothing items from a Kaggle dataset, predicts
a style/usage category with a trained classifier, and generates specific Depop search links (e.g.
"sheer halter satin top" rather than "y2k") so the look is actually shoppable.

The full pipeline design, build steps, and milestone checklist live in `outfit_vibe_project_scope.md` — read
it before making architectural changes, since it's the source of truth for what each stage is supposed to do
and why (e.g. why Pinterest/Depop APIs aren't used — see the Non-Goals section).

Explicit non-goals: no live Pinterest API pull (boards are uploaded manually), no live/scraped Depop data
(links are constructed search URLs only, to respect Depop's ToS), no auth/multi-user/production deployment,
no training CLIP from scratch (fine-tuning it is an optional stretch goal, not required).

## Pipeline architecture

```
Board images (user upload)
        |
        v
  FashionCLIP embeddings  --average-->  Vibe vector
        |                                    |
        v                                    v
Item dataset embeddings            +----------------+----------------+
(Kaggle, precomputed once)         |                |                |
        |                    Cosine similarity   Zero-shot      Trained
        +------------------> ranking (item      aesthetic      "usage"
                              recommendations)    tags          classifier
                                                     |                |
                                                     v                v
                                          Garment-detail term extraction
                                          (zero-shot, specific vocabulary)
                                                     |
                                                     v
                                          Depop search link generation
                                                     |
                                                     v
                                            Streamlit interface
```

Key implementation notes that don't show up from file structure alone:

- **Item pool embeddings are computed once and cached** (Kaggle dataset is ~44k images) — never recompute
  them on every run; cache to `data/embeddings/`.
- **Two different uses of CLIP's text encoder**: zero-shot aesthetic tags use a small vocabulary of broad
  style words (cottagecore, y2k, streetwear, ...); garment-detail extraction (for Depop queries) uses a much
  larger, concrete vocabulary of silhouette/neckline/fabric/pattern terms. These are separate steps with
  separate vocabularies — don't conflate them.
- **The trained classifier is real supervised learning**, not a CLIP zero-shot call: it trains on the Kaggle
  `usage` column (Casual, Formal, Party, Sports, Ethnic) on top of the embeddings, with a proper train/val/test
  split. This is distinct from the zero-shot aesthetic tagging step, which needs no training.
- **Depop queries are constructed strings**, not results from any live Depop query — the tool never calls
  Depop or scrapes it.

## Repo layout

- `notebooks/` — exploratory, notebook-first development (numbered `NN_description.ipynb`).
- `src/` — intended home for code once logic graduates out of notebooks (currently empty).
- `app/` — intended home for the Streamlit interface (currently empty).
- `data/board_images/`, `data/kaggle_fashion/`, `data/embeddings/` — gitignored except for `.gitkeep`; datasets
  and cached embeddings are downloaded/generated locally, never committed.

## Commands

```bash
# Environment setup (venv already exists at ./venv)
source venv/bin/activate
pip install -r requirements.txt

# Notebook-driven development
jupyter lab              # or: jupyter notebook

# Run the Streamlit app (once app/ is populated)
streamlit run app/<entrypoint>.py
```

There is no lint/test/build tooling configured yet — development is currently notebook-first
(`notebooks/01_load_data.ipynb` downloads the Kaggle dataset via `kagglehub` and does the first look at both
datasets).

## Data

- Kaggle "Fashion Product Images (Small)" (`paramaggarwal/fashion-product-images-small`, via `kagglehub`)
  — ~44k labeled clothing images with `styles.csv` metadata (category, colour, season, usage, gender,
  product description). This is both the recommendation candidate pool and the classifier's training data.
  Note: the CSV has some malformed rows (extra commas from product names) — load with `on_bad_lines="skip"`.
- Board images are manually saved into `data/board_images/`.
