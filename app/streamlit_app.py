"""Pinterest to Depop Recommendation Engine — Streamlit interface.

Upload a Pinterest board (a handful of saved outfit images) and get back:
  - the board's aesthetic tags (zero-shot) and predicted usage category (trained classifier)
  - a grid of visually similar clothing items from the Kaggle fashion dataset
  - constructed Depop search links so the look is actually shoppable

Run from the project root:  streamlit run app/streamlit_app.py

Requires the cached artifacts produced by the notebooks (item embeddings + trained
classifier). See the README for the one-time setup.

Look & feel: the theme lives in ``.streamlit/config.toml`` (warm bone ground, one
terracotta accent, Inter throughout). This module adds structure — a hero, tabbed
results, uniform image tiles — with the stock Streamlit API, ``streamlit-shadcn-ui``
for the stat cards / badges / tabs, and a little scoped CSS for the recommendation grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit_shadcn_ui as ui
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline  # noqa: E402
from src.pipeline import BoardAnalysis  # noqa: E402

st.set_page_config(
    page_title="Pinterest to Depop Recommendation Engine",
    page_icon=":material/checkroom:",
    layout="wide",
)

TILE_PX = 480  # square-crop size for every thumbnail — keeps grids visually even


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
# Small view helpers
# ---------------------------------------------------------------------------

def square(img: Image.Image, px: int = TILE_PX) -> Image.Image:
    """Center-crop to a square and resize — uniform tiles without any CSS."""
    return ImageOps.fit(img.convert("RGB"), (px, px), method=Image.LANCZOS)


def hairline() -> None:
    """A thin terracotta rule — the editorial section break."""
    st.markdown(
        "<hr style='border:none;border-top:1px solid #a1553f;opacity:.35;margin:.6rem 0 1.2rem'>",
        unsafe_allow_html=True,
    )


def kicker(text: str) -> None:
    """Small uppercase label above a section, magazine-style."""
    st.markdown(
        f"<div style='font-size:.72rem;letter-spacing:.16em;"
        f"text-transform:uppercase;color:#a1553f;font-weight:600;margin-bottom:.25rem'>{text}</div>",
        unsafe_allow_html=True,
    )


def meter_row(label: str, frac: float, right_text: str) -> None:
    """One label · bar · value line — the shared measurement motif on the vibe tab."""
    frac = max(0.0, min(1.0, frac))
    st.markdown(
        f"<div style='font-size:.8rem;display:flex;"
        f"justify-content:space-between;margin:.55rem 0 .2rem'>"
        f"<span>{label}</span><span style='color:#8a6f52'>{right_text}</span></div>"
        f"<div style='height:6px;background:#ece3d3;border-radius:3px;overflow:hidden'>"
        f"<div style='height:100%;width:{frac * 100:.1f}%;background:#a1553f'></div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Result sections
# ---------------------------------------------------------------------------

def render_vibe(analysis: BoardAnalysis) -> None:
    top_tag, top_tag_score = analysis.aesthetic_tags[0]
    top_usage, top_usage_prob = analysis.usage_prediction[0]

    c1, c2 = st.columns(2)
    with c1:
        ui.metric_card(
            label="Top aesthetic",
            value=top_tag.title(),
            description=f"cosine {top_tag_score:.3f} · zero-shot vs. 19 style words",
            key="mc_aesthetic",
        )
    with c2:
        ui.metric_card(
            label="Predicted usage",
            value=top_usage,
            description=f"{top_usage_prob:.0%} confidence · trained classifier",
            key="mc_usage",
        )

    st.write("")
    kicker("The board reads as")
    ui.badges(
        [(tag.title(), "outline") for tag, _ in analysis.aesthetic_tags[:5]],
        key="b_aesthetic",
    )
    st.caption("Zero-shot: the board's vibe vector scored against a fixed list of style words.")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        kicker("Aesthetic spectrum")
        with st.container(border=True):
            top8 = analysis.aesthetic_tags[:8]
            hi = top8[0][1]
            lo = min(s for _, s in top8)
            span = (hi - lo) or 1.0
            for tag, score in top8:
                # normalize within the visible range so ordering is legible
                meter_row(tag.title(), 0.12 + 0.88 * (score - lo) / span, f"{score:.3f}")
        st.caption("Zero-shot cosine similarity vs. a 19-word style vocabulary.")
    with right:
        kicker("Usage breakdown")
        with st.container(border=True):
            for name, prob in analysis.usage_prediction:
                meter_row(name, float(prob), f"{prob:.0%}")
        st.caption("Supervised classifier trained on the Kaggle `usage` column.")


def render_pieces(analysis: BoardAnalysis, n_cols: int) -> None:
    kicker("Closest pieces in the ~44k-item pool")
    st.caption("Ranked by cosine similarity between each item embedding and the board's vibe vector.")
    st.write("")

    recs = analysis.recommendations
    rows = list(recs.iterrows())

    # Scoped polish for the grid: rounded thumbnails + a lift on hover.
    # Targets the stable `.st-key-*` class Streamlit puts on keyed containers.
    st.markdown(
        """
        <style>
        .st-key-pieces_grid [data-testid="stImage"] img { border-radius: 4px; }
        .st-key-pieces_grid [data-testid="stVerticalBlockBorderWrapper"] {
            background: #fffdf9;
            transition: box-shadow .18s ease, transform .18s ease;
        }
        .st-key-pieces_grid [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 6px 22px rgba(43, 38, 34, .10);
            transform: translateY(-2px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="pieces_grid"):
        for start in range(0, len(rows), n_cols):
            cols = st.columns(n_cols, gap="medium")
            for col, (item_id, row) in zip(cols, rows[start:start + n_cols]):
                with col, st.container(border=True):
                    img_path = analysis.item_image_path(item_id)
                    if img_path.is_file():
                        st.image(square(Image.open(img_path)), width="stretch")
                    st.markdown(
                        f"<div style='font-size:.82rem;line-height:1.25;margin-top:.35rem'>"
                        f"<b>{row.productDisplayName}</b></div>"
                        f"<div style='font-size:.7rem;color:#8a6f52;margin-top:.2rem'>"
                        f"{row.articleType} · sim {row.similarity:.3f}</div>",
                        unsafe_allow_html=True,
                    )


def render_shop(analysis: BoardAnalysis) -> None:
    kicker("Constructed Depop searches")
    st.caption(
        "Links follow `{aesthetic} {garment detail} {noun}` and open `depop.com/search`. "
        "The tool never queries or scrapes Depop."
    )
    st.write("")

    for i, row in enumerate(analysis.depop_links.itertuples(index=False)):
        with st.container(border=True):
            a, b = st.columns([2, 1], vertical_alignment="center")
            with a:
                st.markdown(
                    f"<div style='font-size:1.05rem'><b>{row.query}</b></div>"
                    f"<div style='font-size:.72rem;color:#8a6f52;margin-top:.25rem'>"
                    f"covers: {', '.join(row.covers)}</div>",
                    unsafe_allow_html=True,
                )
            with b:
                st.link_button("Search on Depop", row.url, width="stretch")


def render_analysis(analysis: BoardAnalysis, n_item_cols: int) -> None:
    tab = ui.tabs(
        ["The vibe", "The pieces", "Shop the look"],
        value="The vibe",
        key="results_tabs",
    )
    st.write("")
    if tab == "The vibe":
        render_vibe(analysis)
    elif tab == "The pieces":
        render_pieces(analysis, n_item_cols)
    else:
        render_shop(analysis)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

kicker("Pinterest board to resale marketplace")
st.markdown(
    "<h1 style='font-weight:700;font-size:2.8rem;line-height:1.08;letter-spacing:-.02em;"
    "color:#2b2622;margin:.1rem 0 .5rem'>The Vibe Translator</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:1.05rem;line-height:1.5;color:#5c534a;max-width:44rem;margin:0'>"
    "Save a folder of outfits the way you'd pin them. FashionCLIP blends them into one "
    "vibe vector that drives similarity search, zero-shot tagging, a trained usage "
    "classifier, and shoppable Depop links.</p>",
    unsafe_allow_html=True,
)

with st.expander("How it works"):
    st.markdown(
        "- **Vibe vector** — every board image is embedded with "
        "[FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip); the averaged, "
        "L2-normalized result summarizes the board.\n"
        "- **The pieces** — cosine similarity against ~44k cached Kaggle item embeddings.\n"
        "- **The vibe** — the same vector scored against a broad style vocabulary (zero-shot) "
        "and run through a small classifier trained on the Kaggle `usage` column "
        "(train/val/test, class weighting, early stopping).\n"
        "- **Shop the look** — top garment-detail terms (`cropped`, `sheer`, `square neck`) "
        "are combined into `depop.com/search` URLs. No Depop data is ever queried or scraped."
    )

hairline()


# ---------------------------------------------------------------------------
# Sidebar — board input + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    kicker("The board")
    uploaded = st.file_uploader(
        "Upload board images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    sample_paths = pipeline.list_board_images()
    use_sample = st.checkbox(
        f"Use the sample board ({len(sample_paths)} images)",
        value=not uploaded and bool(sample_paths),
        disabled=not sample_paths,
        help="Ten outfit images bundled with the repo in data/board_images/.",
    )

    st.divider()
    kicker("Settings")
    top_n_items = st.slider("Recommended pieces", min_value=6, max_value=24, value=15, step=3)
    n_item_cols = st.slider("Grid columns", min_value=3, max_value=6, value=5)

    run = st.button("Analyze board", type="primary", width="stretch")
    st.caption("Embeddings: FashionCLIP (frozen). Classifier: trained locally. Data: Kaggle.")


# ---------------------------------------------------------------------------
# Resolve input images
# ---------------------------------------------------------------------------

images, source_label = [], ""
if uploaded:
    images = [Image.open(f) for f in uploaded]
    source_label = f"{len(images)} uploaded image(s)"
elif use_sample and sample_paths:
    images = [Image.open(p) for p in sample_paths]
    source_label = f"{len(images)} sample image(s)"

if images:
    kicker(f"Your board · {source_label}")
    n_show = min(len(images), 8)
    for col, img in zip(st.columns(n_show, gap="small"), images[:n_show]):
        col.image(square(img, 300), width="stretch")
    st.write("")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

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

    with st.status("Analyzing the board…", expanded=True) as status:
        st.write(f"Embedding {len(images)} board image(s) with FashionCLIP…")
        st.write("Ranking ~44k items by cosine similarity…")
        st.write("Tagging the aesthetic and building Depop queries…")
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
        status.update(label="Board analyzed", state="complete", expanded=False)

    # Persist so tab switches (which rerun the script) keep showing the result.
    st.session_state["analysis"] = analysis

# Grid-column count is a live control — always read the current slider value.
if "analysis" in st.session_state:
    render_analysis(st.session_state["analysis"], n_item_cols)
else:
    with st.container(border=True):
        kicker("Start here")
        st.markdown(
            "1. Pick a board in the sidebar — **upload** your own pins or tick **the sample board**.\n"
            "2. Set how many pieces to surface.\n"
            "3. Hit **Analyze board**."
        )
