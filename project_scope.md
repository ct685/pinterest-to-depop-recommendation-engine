# Pinterest to Depop Recommendation Engine — Project Scope & Build Plan

## 1. Problem Statement

Pinterest is great for gathering visual inspiration but bad at telling you where to actually buy pieces that match a look. This project takes a Pinterest board, figures out its aesthetic ("vibe") using a pretrained computer vision model, recommends similar clothing items from a public dataset, and generates specific, ready-to-use Depop search links so the user can go shop the look.

## 2. Goals

- Extract a numeric "vibe" representation from a set of uploaded board images.
- Recommend visually/stylistically similar clothing items from a public dataset.
- Train a real classifier (not just a pretrained model call) to predict a style category from the vibe.
- Generate Depop search queries using **specific garment-detail terms** (e.g. "off the shoulder," "sheer," "cropped," "high-waisted") rather than vague aesthetic labels (e.g. "y2k"), so the links actually surface relevant listings.
- Ship a simple, demoable interface.

## 3. Non-Goals (explicitly out of scope)

- Pulling a Pinterest board live via API — Pinterest's API doesn't expose this; images are uploaded manually by the user instead.
- Reading live Depop listing data — Depop has no public API and scraping violates its ToS. The tool generates *search links*, not live product recommendations.
- Multi-user accounts, authentication, or production deployment.
- Training CLIP itself from scratch (using a pretrained version is the correct engineering call here — fine-tuning it is a possible stretch goal, not required scope).

## 4. Pipeline Overview

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

## 5. Tech Stack

- **VSCode** — local development, using the Python + Jupyter extensions for notebook-style exploration and a venv for dependency isolation. (Optional fallback: run the one-time item-embedding step, 5, in Google Colab for free GPU speed if it's too slow locally, then download the cached embeddings into the local project.)
- **Hugging Face `transformers`** — loads pretrained FashionCLIP.
- **scikit-learn / PyTorch** — the trained classifier.
- **pandas** — data handling.
- **Streamlit** — the demo interface.
- **GitHub** — hosting the finished code/portfolio piece.

## 6. Data Sources

- **Your Pinterest board images** — manually saved/downloaded into a folder. This is your "vibe input."
- **Kaggle "Fashion Product Images (Small)"** — ~44,000 labeled clothing images (category, colour, season, usage, gender, product description). This is your recommendation candidate pool *and* your classifier training data.

## 7. Detailed Build Steps

**Step 1 — Setup.** In VSCode: create a project folder, `git init`, create and activate a virtual environment, install `transformers`, `torch`, `scikit-learn`, `pandas`, `streamlit`. Add the Python and Jupyter extensions for notebook-style exploration. Connect the folder to a GitHub repo via VSCode's Source Control panel.

**Step 2 — Get data.** Save board images to a folder. Download the Kaggle dataset (images + metadata CSV).

**Step 3 — Load FashionCLIP.** A few lines via `transformers`; this is a pretrained model, so no training needed to use it for embeddings.

**Step 4 — Vibe vector.** Embed each board image, average into a single vector representing the board's overall style.

**Step 5 — Embed the item pool (once).** Run all ~44,000 dataset images through FashionCLIP, cache the resulting vectors to a file so you never recompute them.

**Step 6 — Item recommendations.** Cosine similarity between the vibe vector and every cached item vector; sort and take the top 10–20 as recommended pieces.

**Step 7 — Zero-shot aesthetic tags.** Compare the vibe vector against a fixed list of style words (cottagecore, dark academia, y2k, minimalist, streetwear, etc.) using CLIP's shared text/image space. Top matches become broad "vibe tags" — no training required, since this leans on CLIP's existing pretrained knowledge.

**Step 8 — Trained "usage" classifier.** The dataset's `usage` column (Casual, Formal, Party, Sports, Ethnic) is the closest real, trainable style label available. Train a small classifier (logistic regression or a couple of dense layers) on top of the embeddings:
- Split into train / validation / test sets.
- Loss function: cross-entropy.
- Optimizer: Adam; train over multiple epochs.
- Watch train vs. validation accuracy to catch overfitting; stop early if validation stalls.
- Evaluate on the held-out test set with accuracy, precision/recall/F1, and a confusion matrix.
- Use the trained model's prediction as an additional tag at inference time.

**Step 9 — Garment-detail term extraction (for Depop).** This is the step that makes the shopping links actually specific. Build a curated vocabulary of concrete garment-attribute terms, grouped by type:
- *Silhouette/fit:* oversized, cropped, fitted, high-waisted, wide-leg, bodycon, baggy
- *Neckline/sleeve:* off the shoulder, halter, puff sleeve, spaghetti strap, turtleneck, square neck
- *Fabric/texture:* sheer, satin, corduroy, mesh, knit, denim, leather, velvet
- *Pattern/finish:* pastel, plaid, floral, metallic, distressed, ribbed

Embed this vocabulary once with CLIP's text encoder (same trick as step 7, just a different, much more specific word list). Score each term against the vibe vector, take the top 3–5 matches — these become the actual search keywords, since "sheer halter satin top" is something Depop's own search can act on, unlike "cottagecore."

**Step 10 — Depop link generation.** Combine the top garment-detail terms with the top item category (from step 6/8) into a query string, and build a URL like `depop.com/search/?q=sheer+halter+top`. This requires no Depop data access at all — it's just a constructed link.

**Step 11 — Streamlit interface.** Upload a folder of board images → display: aesthetic tags, predicted usage category, a grid of recommended items with images, and "Shop this vibe on Depop" buttons per generated query.

**Step 12 — Evaluation.** Sanity-check with boards of known, obvious style (e.g. an all-denim board) and confirm tags/terms/recommendations look reasonable. Report the classifier's test-set metrics formally; treat the recommendation/tagging quality qualitatively and say so honestly in the writeup.

**Step 13 — Documentation.** README covering the problem, approach, architecture diagram, results, and a short demo GIF/video of the Streamlit app.

## 8. Build Order / Milestones Checklist

- [x] Environment set up, both datasets downloaded
- [x] FashionCLIP loaded, board vibe vector computed
- [x] Item pool embedded and cached
- [x] Cosine-similarity item recommendations working
- [x] Zero-shot aesthetic tags working
- [x] Usage classifier trained and evaluated (metrics + confusion matrix recorded)
- [x] Garment-detail term extraction working
- [x] Depop links generating correctly from those terms
- [x] Streamlit app assembled end-to-end
- [x] Sanity-check evaluation done
- [ ] README + demo recorded, pushed to GitHub  *(README done; demo GIF still to record)*

## 9. Resume Skills This Project Demonstrates

Python; computer vision; transfer learning with pretrained models (FashionCLIP); multimodal embeddings (text + image); similarity search/retrieval systems; zero-shot classification; supervised model training (train/val/test methodology, loss functions, gradient descent, overfitting mitigation, evaluation metrics, confusion matrices); working with public datasets; Streamlit app development; API/ToS-aware system design (recognizing a data-access constraint and engineering around it instead of scraping).

## 10. Stretch Goals (optional, after core scope is done)

- Fine-tune FashionCLIP itself on the Kaggle dataset instead of only training a classifier on top of it.
- Expand the garment-detail vocabulary and validate it against real Depop search result relevance (manually, by clicking through).
- Let interaction with recommended items (e.g. a "more like this" click) nudge the vibe vector, similar to how a real feed adapts.
