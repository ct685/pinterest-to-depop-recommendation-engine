# Transpose

*A Pinterest to Depop recommendation engine.*

Turn a Pinterest style board into **shoppable** outfit recommendations.

You save a folder of outfit images the way you'd pin them. The tool embeds them with
[FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip), averages them into a single
**vibe vector**, and uses that to:

- recommend visually similar clothing items from a ~44k-item labeled fashion dataset,
- tag the board's broad aesthetic (zero-shot: *clean girl*, *y2k*, *cottagecore*, …),
- predict a usage category with a **trained** classifier (*Casual / Formal / Party / Sports / Ethnic*),
- and generate concrete [Depop](https://www.depop.com) search links — `sheer halter satin top`
  rather than `y2k` — so the look is actually findable on a resale marketplace.

> **Demo:** _add a short screen recording / GIF of the Streamlit app here._

---

## Why it's built this way

- **No live Pinterest pull.** Pinterest's API doesn't expose board contents; boards are saved manually into `data/board_images/`.
- **No Depop data.** Depop has no public API and scraping violates its ToS. The tool builds *search URLs* only — it never queries or scrapes Depop.
- **No CLIP training from scratch.** FashionCLIP is used pretrained for all embedding work. The one piece of real supervised training is a small classifier head on top of frozen embeddings.
- Single-user, local, notebook-first. No auth, no deployment.

Full rationale and build log: [`project_scope.md`](project_scope.md).

---

## Pipeline

```
Board images (manual upload)
        │
        ▼
  FashionCLIP image encoder ── average + L2-normalize ──▶  Vibe vector (512-d)
        │                                                       │
        ▼                                    ┌──────────────────┼──────────────────┐
Item-pool embeddings                         ▼                  ▼                  ▼
(Kaggle ~44k, cached once)          Cosine-similarity      Zero-shot          Trained usage
        │                          ranking → top items   aesthetic tags     classifier (512→128→5)
        └────────────────────────▶       │                    │                  │
                                         │                    ▼                  ▼
                                         │        Garment-detail term extraction (zero-shot,
                                         │        concrete silhouette/fabric/neckline vocab)
                                         │                    │
                                         └──────────┬─────────┘
                                                    ▼
                                    Depop search-link construction
                                    "{aesthetic} {detail} {noun}"  →  depop.com/search/?q=…
                                                    │
                                                    ▼
                                          Streamlit interface
```

Two **separate** uses of CLIP's text encoder, with different vocabularies:

| Step | Vocabulary | Purpose |
|---|---|---|
| Aesthetic tags | ~19 broad style words (*minimalist*, *grunge*, *old money*) | Label the board's overall vibe |
| Garment-detail terms | ~27 concrete attributes (*cropped*, *sheer*, *off the shoulder*) | Build Depop queries that return real listings |

The usage classifier is **not** a CLIP zero-shot call — it's supervised learning on the Kaggle
`usage` column with a train/val/test split, class weighting for imbalance, early stopping, and a
confusion-matrix evaluation (`notebooks/06_usage_classifier.ipynb`).

---

## Setup

Requires Python 3.11+ (developed on 3.13) and the Kaggle "Fashion Product Images (Small)" dataset.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # app + pipeline only
pip install -r requirements-dev.txt    # add this for the notebooks / data prep
```

### Kaggle credentials

`notebooks/01_load_data.ipynb` pulls the dataset with `kagglehub`, which needs a (free)
Kaggle account and API token. Create one at *Kaggle → Account → Create New Token*, then either:

- save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`, or
- export `KAGGLE_USERNAME` and `KAGGLE_KEY` in your shell.

Already have the dataset on disk? Skip the token and point the pipeline at it with
`KAGGLE_FASHION_DIR=/path/to/fashion-product-images-small` (the folder holding `styles.csv`
and `images/`).

### One-time data prep (notebooks)

The app depends on cached artifacts that the notebooks generate into `data/embeddings/`
(all git-ignored). Run these once, in order:

| Notebook | Produces | Notes |
|---|---|---|
| `01_load_data.ipynb` | downloads the Kaggle dataset via `kagglehub` | needs Kaggle credentials (above); first look at both datasets |
| `03_item_embeddings.ipynb` | `item_embeddings.npy`, `item_ids.npy` | ~44k images, ≈10 min on Apple Silicon; runs once |
| `06_usage_classifier.ipynb` | `usage_classifier.pt` | trains + evaluates the classifier |

Notebooks `02`, `04`, `05`, `07`, `08` are the exploratory build-up of each pipeline stage and
are self-contained — useful to read, not required for the app.

```bash
jupyter lab
```

---

## Running it

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Upload board images, or tick "use the sample board" (10 images ship with the repo in
`data/board_images/`), then click **Analyze board**. Results are split into tabs:

- **Overview**: a one-line read of the board ("Clean Girl, *mostly casual*") with highlights
- **The vibe**: aesthetic tags and the usage classifier's probabilities
- **Details**: the garment-detail terms, ranked and grouped by fit, neckline, fabric and pattern
- **Pieces**: the recommended items, each with its own Depop link
- **Shop**: one Depop search per generated query, with the items it covers
- **How it works**: the pipeline and the classifier's test-set results

### Command line

```bash
python -m src.pipeline        # analyzes data/board_images/, prints the board summary
```

---

## Code layout

```
src/pipeline.py        Importable end-to-end inference pipeline (shared by the app + CLI).
app/streamlit_app.py   Streamlit UI; owns Streamlit-level caching of the model and item pool.
notebooks/             Numbered, notebook-first development of each stage (01–08).
data/board_images/     Board images. A 10-image sample board is committed; your own additions are git-ignored.
data/kaggle_fashion/   Reserved for the Kaggle dataset (downloaded to the kagglehub cache).
data/embeddings/       Cached embeddings + trained classifier (git-ignored).
project_scope.md       Source-of-truth design doc: goals, non-goals, build steps.
```

`src/pipeline.py` mirrors the notebooks stage for stage: `board_vibe_vector` (nb 02),
`recommend_items` (nb 04), `aesthetic_tags` (nb 05), `predict_usage` (nb 06),
`build_category_gate` + garment-term ranking (nb 07), `generate_depop_links` (nb 08).

---

## Data

**Kaggle "Fashion Product Images (Small)"** (`paramaggarwal/fashion-product-images-small`,
via `kagglehub`) — ~44k labeled clothing images with `styles.csv` metadata (category, colour,
season, `usage`, gender, product description). Serves as both the recommendation candidate pool
and the classifier's training data. The CSV has a few comma-mangled rows; it's loaded with
`on_bad_lines="skip"`.

Point the pipeline at a non-default dataset location with `KAGGLE_FASHION_DIR`.

---

## Evaluation

- **Usage classifier:** accuracy + per-class precision/recall/F1 + confusion matrix on a
  held-out test set (`notebooks/06`). Class support is very uneven (Casual dominates, Party is
  tiny), so overall accuracy is reported alongside the per-class breakdown, not on its own.
- **Recommendations / tags / terms:** assessed qualitatively against boards of obvious style
  (e.g. an all-denim board should surface denim and *cropped* / *high-waisted* / *distressed*).

---

## Skills demonstrated

Transfer learning with a pretrained vision–language model · multimodal (image + text) embeddings ·
similarity search / retrieval · zero-shot classification · supervised training done properly
(train/val/test, cross-entropy, Adam, class weighting, early stopping, confusion matrix) ·
turning notebooks into a shared module + Streamlit app · designing around a data-access
constraint (ToS) instead of scraping.

---

## Stretch goals (not done)

Fine-tune FashionCLIP itself · validate the garment vocabulary against real Depop result
relevance · let "more like this" interactions nudge the vibe vector.
