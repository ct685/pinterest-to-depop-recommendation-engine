"""Pinterest to Depop Recommendation Engine — Streamlit interface.

Upload a Pinterest board (a handful of saved outfit images) and get back:
  - the board's aesthetic tags (zero-shot) and predicted usage category (trained classifier)
  - a grid of visually similar clothing items from the Kaggle fashion dataset
  - constructed Depop search links so the look is actually shoppable

Run from the project root:  streamlit run app/streamlit_app.py

Requires the cached artifacts produced by the notebooks (item embeddings + trained
classifier). See the README for the one-time setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline  # noqa: E402
from src.pipeline import BoardAnalysis  # noqa: E402

st.set_page_config(page_title="Pinterest to Depop Recommendation Engine", page_icon="🧥", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources — loaded once per session, reused across boards
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading FashionCLIP + cached item pool…")
def load_resources():
    model, processor, device = pipeline.load_fashion_clip()
    item_embeddings, item_ids = pipeline.load_item_pool()
    dataset_root = pipeline.find_dataset_root()
    styles_df = pipeline.load_styles(dataset_root)
    classifier = pipeline.load_usage_classifier()

    terms, groups = pipeline.flat_garment_terms()
    term_embeddings = pipeline.embed_text_terms(
        terms, pipeline.GARMENT_PROMPT_TEMPLATE, model, processor, device
    )
    category_gate = pipeline.build_category_gate(
        item_embeddings, item_ids, styles_df, term_embeddings, terms, groups
    )

    return {
        "model": model, "processor": processor, "device": device,
        "item_embeddings": item_embeddings, "item_ids": item_ids,
        "styles_df": styles_df, "classifier": classifier,
        "category_gate": category_gate, "dataset_root": dataset_root,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_analysis(analysis: BoardAnalysis, n_item_cols: int) -> None:
    top_col, usage_col = st.columns(2)

    with top_col:
        st.subheader("Aesthetic tags")
        st.caption("Zero-shot: the board's vibe vector vs. a fixed list of style words.")
        chips = "  ".join(f"`{tag}`" for tag, _ in analysis.aesthetic_tags[:5])
        st.markdown(chips)
        st.bar_chart(
            {tag: score for tag, score in analysis.aesthetic_tags[:8]},
            horizontal=True, height=260,
        )

    with usage_col:
        st.subheader("Predicted usage")
        st.caption("Supervised classifier trained on the Kaggle `usage` column.")
        st.metric("Most likely", analysis.predicted_usage)
        st.bar_chart(
            {name: prob for name, prob in analysis.usage_prediction},
            horizontal=True, height=260,
        )

    st.divider()
    st.subheader("Recommended items")
    st.caption("Top matches from the ~44k-item Kaggle pool by cosine similarity to the vibe vector.")
    recs = analysis.recommendations
    cols = st.columns(n_item_cols)
    for i, (item_id, row) in enumerate(recs.iterrows()):
        with cols[i % n_item_cols]:
            img_path = analysis.item_image_path(item_id)
            if img_path.is_file():
                st.image(str(img_path), width="stretch")
            st.caption(f"**{row.productDisplayName}**  \n{row.articleType} · sim {row.similarity:.3f}")

    st.divider()
    st.subheader("Shop this vibe on Depop")
    st.caption(
        "Constructed `depop.com/search` links — `{aesthetic} {garment detail} {noun}`. "
        "The tool never queries or scrapes Depop."
    )
    for row in analysis.depop_links.itertuples(index=False):
        link_col, covers_col = st.columns([1, 2])
        with link_col:
            st.link_button(f'🔍  {row.query}', row.url, width="stretch")
        with covers_col:
            st.caption("covers: " + ", ".join(row.covers))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("🧥 Pinterest to Depop Recommendation Engine")
st.write(
    "Turn a Pinterest style board into shoppable outfit recommendations. "
    "FashionCLIP embeds your saved images into one *vibe vector*, which drives "
    "similarity search, zero-shot tagging, a trained usage classifier, and Depop search links."
)

with st.sidebar:
    st.header("Board")
    uploaded = st.file_uploader(
        "Upload board images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )
    sample_paths = pipeline.list_board_images()
    use_sample = st.checkbox(
        f"Use the sample board in data/board_images/ ({len(sample_paths)} images)",
        value=not uploaded and bool(sample_paths),
        disabled=not sample_paths,
    )
    st.divider()
    st.header("Options")
    top_n_items = st.slider("Recommended items", min_value=6, max_value=24, value=15, step=3)
    n_item_cols = st.slider("Grid columns", min_value=3, max_value=6, value=5)
    run = st.button("Analyze board", type="primary", width="stretch")

# Resolve the input images
images, source_label = [], ""
if uploaded:
    from PIL import Image

    images = [Image.open(f) for f in uploaded]
    source_label = f"{len(images)} uploaded image(s)"
elif use_sample and sample_paths:
    from PIL import Image

    images = [Image.open(p) for p in sample_paths]
    source_label = f"{len(images)} sample image(s)"

if images:
    st.caption(f"Board: {source_label}")
    thumbs = st.columns(min(len(images), 8))
    for i, img in enumerate(images[:8]):
        thumbs[i].image(img, width="stretch")

if run:
    if not images:
        st.warning("Add board images (upload, or tick the sample-board box) first.")
        st.stop()
    try:
        resources = load_resources()
    except FileNotFoundError as err:
        st.error(str(err))
        st.info(
            "One-time setup: run `notebooks/01_load_data.ipynb`, `03_item_embeddings.ipynb`, "
            "and `06_usage_classifier.ipynb` to generate the cached artifacts."
        )
        st.stop()

    with st.spinner("Embedding the board and ranking items…"):
        analysis = pipeline.analyze_board(
            images,
            model=resources["model"],
            processor=resources["processor"],
            device=resources["device"],
            item_embeddings=resources["item_embeddings"],
            item_ids=resources["item_ids"],
            styles_df=resources["styles_df"],
            classifier=resources["classifier"],
            category_gate=resources["category_gate"],
            dataset_root=resources["dataset_root"],
            top_n_items=top_n_items,
        )
    render_analysis(analysis, n_item_cols)
else:
    st.info("Configure the board in the sidebar, then click **Analyze board**.")
