"""Streamlit interface for Transpose (the Pinterest to Depop Recommendation Engine).

Upload a Pinterest board (a handful of saved outfit images) and get back:
  - a one-line read of the board: its aesthetic (zero-shot) and usage (trained classifier)
  - the concrete garment details that define it (cropped, square neck, sheer, ...)
  - a grid of visually similar clothing items from the Kaggle fashion dataset
  - constructed Depop search links so the look is actually shoppable

Run from the project root:  streamlit run app/streamlit_app.py

Requires the cached artifacts produced by the notebooks (item embeddings + trained
classifier). See the README for the one-time setup.

Look & feel: the theme lives in ``.streamlit/config.toml`` (white and paper gray, black as
the only accent, Cormorant Garamond headings over EB Garamond). A muted red is reserved for
the Depop buttons. The result views are small HTML blocks styled by the scoped CSS below.
"""

from __future__ import annotations

import base64
import io
import sys
from html import escape
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline  # noqa: E402
from src.pipeline import BoardAnalysis  # noqa: E402

st.set_page_config(
    page_title="Transpose",
    page_icon=":material/checkroom:",
    layout="wide",
)

BOARD_THUMB_PX = 360
ITEM_THUMB_PX = 400

# Usage-classifier test-set results, copied from notebooks/06_usage_classifier.ipynb.
CLASSIFIER_REPORT = [
    # class,   precision, recall, f1,   support
    ("Casual", 0.97, 0.91, 0.94, 5161),
    ("Ethnic", 0.86, 0.96, 0.91, 481),
    ("Formal", 0.70, 0.90, 0.79, 352),
    ("Sports", 0.67, 0.87, 0.76, 604),
    ("Party", 0.17, 0.25, 0.20, 4),
]
CLASSIFIER_ACCURACY = 0.909
CLASSIFIER_MACRO_F1 = 0.72

# ---------------------------------------------------------------------------
# Cached resources — loaded once per session, reused across boards
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading FashionCLIP and the item pool…")
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
# Styles
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
  --ink: #111111; --graphite: #6b6b68; --rule: #e3e3e0; --paper: #f4f4f2;
  --depop: #b3261e; --depop-dark: #8c1d17;
  --display: 'Cormorant Garamond', 'EB Garamond', Garamond, serif;
}
.block-container { padding-top: 2.6rem; max-width: 1180px; }

.tp-eyebrow { font-variant: all-small-caps; letter-spacing: .12em; font-size: 1rem; color: var(--graphite); }
.tp-caption { color: var(--graphite); font-size: .95rem; margin-top: .35rem; }
.tp-caption i { color: var(--ink); }

.tp-hero { display: grid; gap: .5rem; border-bottom: 1px solid var(--ink); padding-bottom: 1.1rem; margin-bottom: 1.6rem; }
.tp-wordmark { font-family: var(--display); font-weight: 700; font-size: 4.4rem; line-height: .95; letter-spacing: -.01em; }
.tp-tagline { font-style: italic; font-size: 1.25rem; color: var(--graphite); }
.tp-lede { max-width: 40rem; font-size: 1.1rem; line-height: 1.5; margin-top: .3rem; }
.tp-hero.compact { grid-template-columns: auto 1fr; align-items: baseline; gap: 1.1rem; padding-bottom: .7rem; margin-bottom: 1rem; }
.tp-hero.compact .tp-wordmark { font-size: 2.4rem; }
.tp-hero.compact .tp-tagline { font-size: 1.05rem; }

.tp-verdict { font-family: var(--display); font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.02; font-weight: 500; margin: .15rem 0 .5rem; }
.tp-verdict b { font-weight: 700; }
.tp-verdict i { font-weight: 500; }
.tp-also { font-size: 1.1rem; color: var(--graphite); }
.tp-also i { color: var(--ink); }

.tp-board { display: grid; grid-template-columns: repeat(var(--n, 8), minmax(0, 1fr)); gap: .5rem; margin: 1.4rem 0 1.8rem; }
@media (max-width: 700px) { .tp-board { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
.tp-board img { width: 100%; aspect-ratio: 3 / 4; object-fit: cover; display: block; }

.tp-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 2.2rem; align-items: start; }
.tp-cols.four { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1.8rem; }
.tp-caption + .tp-section-title { margin-top: 2rem; }
.tp-section-title { font-family: var(--display); font-size: 1.7rem; font-weight: 600; line-height: 1.1; border-top: 1px solid var(--ink); padding-top: .6rem; margin-bottom: .8rem; }

.tp-meter { display: grid; gap: .25rem; margin-bottom: .75rem; }
.tp-meter .row { display: flex; justify-content: space-between; gap: .5rem; font-size: 1.05rem; }
.tp-meter .row span:last-child { color: var(--graphite); font-style: italic; }
.tp-meter.top .row span:first-child { font-weight: 600; }
.tp-track { height: 3px; background: var(--rule); }
.tp-fill { height: 100%; background: var(--ink); }

.tp-terms { font-family: var(--display); font-size: 2rem; line-height: 1.2; font-style: italic; font-weight: 500; }
.tp-terms span + span::before { content: " · "; font-style: normal; color: var(--graphite); }

.tp-highlight { display: grid; gap: .5rem; align-content: start; }
.tp-highlight .big { font-family: var(--display); font-size: 1.6rem; font-weight: 600; line-height: 1.15; }
.tp-highlight .big i { font-weight: 500; }
.tp-highlight img { width: 100%; max-width: 220px; aspect-ratio: 1; object-fit: contain; background: #fff; border: 1px solid var(--rule); }

.tp-grid { display: grid; grid-template-columns: repeat(var(--cols, 5), minmax(0, 1fr)); gap: 1.6rem 1.1rem; }
@media (max-width: 900px) { .tp-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.tp-piece { display: grid; grid-template-rows: auto auto 1fr auto; gap: .35rem; }
.tp-piece .rank { font-variant: all-small-caps; letter-spacing: .1em; color: var(--graphite); font-size: .95rem; }
.tp-piece img { width: 100%; aspect-ratio: 1; object-fit: contain; background: #fff; border: 1px solid var(--rule); transition: border-color .15s ease; }
.tp-piece:hover img { border-color: var(--ink); }
.tp-piece .name { font-size: 1rem; line-height: 1.25; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2.5em; }
.tp-piece .meta { font-size: .9rem; color: var(--graphite); font-style: italic; }

.tp-link-depop { color: var(--depop) !important; text-decoration: none !important; font-size: .95rem; border-bottom: 1px solid currentColor; justify-self: start; }
.tp-link-depop:hover { color: var(--depop-dark) !important; }
.tp-btn-depop { display: inline-block; background: var(--depop); color: #fff !important; text-decoration: none !important; padding: .55rem 1.2rem; font-size: 1rem; letter-spacing: .02em; border: 1px solid var(--depop); transition: background .15s ease; white-space: nowrap; }
.tp-btn-depop:hover { background: var(--depop-dark); border-color: var(--depop-dark); }
.tp-btn-depop:focus-visible, .tp-link-depop:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }

.tp-shop { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 1rem 1.6rem; align-items: center; border-top: 1px solid var(--rule); padding: 1.1rem 0; }
.tp-shop:first-of-type { border-top-color: var(--ink); }
.tp-shop .q { font-family: var(--display); font-size: 1.75rem; font-weight: 700; line-height: 1.1; }
.tp-shop .count { color: var(--graphite); font-style: italic; margin-top: .15rem; }
.tp-shop .thumbs { display: flex; gap: .4rem; margin-top: .6rem; flex-wrap: wrap; }
.tp-shop .thumbs img { width: 64px; height: 64px; object-fit: contain; background: #fff; border: 1px solid var(--rule); }
@media (max-width: 640px) { .tp-shop { grid-template-columns: 1fr; } }

.tp-steps { counter-reset: step; display: grid; gap: 1rem; margin: 0; padding: 0; list-style: none; }
.tp-steps li { display: grid; grid-template-columns: 2.4rem 1fr; gap: .6rem; }
.tp-steps li::before { counter-increment: step; content: counter(step); font-family: var(--display); font-size: 1.9rem; font-weight: 600; line-height: 1; }
.tp-steps b { font-weight: 600; }
.tp-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.tp-table th { text-align: left; font-weight: 400; font-variant: all-small-caps; letter-spacing: .1em; color: var(--graphite); border-bottom: 1px solid var(--ink); padding: .3rem .4rem; }
.tp-table td { border-bottom: 1px solid var(--rule); padding: .35rem .4rem; }
.tp-table td + td, .tp-table th + th { text-align: right; }
.tp-stat { font-family: var(--display); font-size: 2.6rem; font-weight: 600; line-height: 1; }

.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
.stTabs [data-baseweb="tab"] p { font-size: 1.1rem; }
.stTabs [aria-selected="true"] p { font-weight: 600; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.6rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def html(markup: str) -> None:
    """Render an HTML block. Collapsed to one line so Markdown never treats indented
    lines as code or splits the block at a blank line."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def to_data_uri(img: Image.Image, px: int, fit_square: bool = False) -> str:
    img = img.convert("RGB")
    if fit_square:
        img = ImageOps.pad(img, (px, px), color=(255, 255, 255))
    else:
        img.thumbnail((px, px * 2))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def item_thumb(path: str, px: int = ITEM_THUMB_PX) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return to_data_uri(Image.open(p), px, fit_square=True)


def board_strip(thumbs: list[str], style: str = "") -> str:
    """Board thumbnails in one row, or two even rows once there are more than ten."""
    n = len(thumbs) if len(thumbs) <= 10 else -(-len(thumbs) // 2)
    imgs = "".join(f"<img src='{uri}' alt='Board image'>" for uri in thumbs)
    return f"<div class='tp-board' style='--n:{max(n, 1)};{style}'>{imgs}</div>"


def strength(rel: float) -> str:
    if rel >= 0.85:
        return "strong"
    if rel >= 0.6:
        return "moderate"
    return "light"


def meter(label: str, rel: float, right: str, tooltip: str, top: bool = False) -> str:
    width = 4 + 96 * max(0.0, min(1.0, rel))
    return (
        f"<div class='tp-meter{' top' if top else ''}' title='{escape(tooltip)}'>"
        f"<div class='row'><span>{escape(label)}</span><span>{escape(right)}</span></div>"
        f"<div class='tp-track'><div class='tp-fill' style='width:{width:.1f}%'></div></div></div>"
    )


def relative(scores: list[tuple[str, float]]) -> dict[str, float]:
    """Each score's position between the lowest and highest in the list (0..1)."""
    values = [s for _, s in scores]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return {name: (s - lo) / span for name, s in scores}


def item_depop_urls(analysis: BoardAnalysis) -> dict:
    urls = {}
    for row in analysis.depop_links.itertuples(index=False):
        for item_id in row.item_ids:
            urls[item_id] = row.url
    return urls


def depop_button(url: str, label: str = "Search on Depop") -> str:
    return f"<a class='tp-btn-depop' href='{escape(url)}' target='_blank' rel='noopener'>{label}</a>"


def section_title(text: str) -> str:
    return f"<div class='tp-section-title'>{text}</div>"


# ---------------------------------------------------------------------------
# Result views (one per tab)
# ---------------------------------------------------------------------------


def render_overview(analysis: BoardAnalysis, board_thumbs: list[str]) -> None:
    top_tag = analysis.top_aesthetic
    usage = analysis.predicted_usage
    runners_up = ", ".join(f"<i>{escape(t.title())}</i>" for t, _ in analysis.aesthetic_tags[1:4])
    top_terms = "".join(f"<span>{escape(t)}</span>" for t, _ in analysis.garment_terms[:3])

    best_id, best = next(analysis.recommendations.iterrows())
    best_img = item_thumb(str(analysis.item_image_path(best_id)))
    best_img_tag = f"<img src='{best_img}' alt='{escape(best.productDisplayName)}'>" if best_img else ""

    first = analysis.depop_links.iloc[0]

    html(f"""
<div>
<div class='tp-eyebrow'>Your board reads as</div>
<div class='tp-verdict'><b>{escape(top_tag.title())}</b>, <i>mostly {escape(usage.lower())}</i></div>
<div class='tp-also'>with notes of {runners_up}</div>
{board_strip(board_thumbs)}
<div class='tp-cols'>
<div class='tp-highlight'>
<div class='tp-eyebrow'>Defining details</div>
<div class='tp-terms'>{top_terms}</div>
<div class='tp-caption'>The garment attributes closest to the board. See <i>Details</i>.</div>
</div>
<div class='tp-highlight'>
<div class='tp-eyebrow'>Closest piece</div>
{best_img_tag}
<div>{escape(best.productDisplayName)}</div>
<div class='tp-caption'>First of {len(analysis.recommendations)} in <i>Pieces</i>.</div>
</div>
<div class='tp-highlight'>
<div class='tp-eyebrow'>Start shopping</div>
<div class='big'>{escape(first.query)}</div>
<div>{depop_button(first.url)}</div>
<div class='tp-caption'>{len(analysis.depop_links)} searches in <i>Shop</i>.</div>
</div>
</div>
</div>
""")


def render_vibe(analysis: BoardAnalysis) -> None:
    rel = relative(analysis.aesthetic_tags)
    aesthetic_rows = "".join(
        meter(tag.title(), rel[tag], strength(rel[tag]), f"cosine similarity {score:.3f}", top=(i == 0))
        for i, (tag, score) in enumerate(analysis.aesthetic_tags[:8])
    )
    usage_rows = "".join(
        meter(name, prob, f"{prob:.0%}", f"probability {prob:.3f}", top=(i == 0))
        for i, (name, prob) in enumerate(analysis.usage_prediction)
    )
    html(f"""
<div class='tp-cols'>
<div>
{section_title("Aesthetic")}
{aesthetic_rows}
<div class='tp-caption'>The board compared against {len(pipeline.AESTHETIC_LABELS)} style words (zero-shot).
Bars run from the weakest to the strongest of the {len(pipeline.AESTHETIC_LABELS)}. Hover a row for the exact score.</div>
</div>
<div>
{section_title("Usage")}
{usage_rows}
<div class='tp-caption'>From a small classifier trained on the dataset's labeled <i>usage</i> column.</div>
</div>
</div>
""")


def render_details(analysis: BoardAnalysis) -> None:
    scores = dict(analysis.garment_terms)
    rel = relative(analysis.garment_terms)
    top_terms = "".join(f"<span>{escape(t)}</span>" for t, _ in analysis.garment_terms[:5])

    columns = []
    for group, words in pipeline.GARMENT_VOCAB.items():
        ranked = sorted(words, key=lambda w: -scores[w])
        rows = "".join(
            meter(w, rel[w], strength(rel[w]), f"cosine similarity {scores[w]:.3f}", top=(i == 0))
            for i, w in enumerate(ranked)
        )
        columns.append(f"<div>{section_title(escape(group.replace('/', ' & ')))}{rows}</div>")

    html(f"""
<div>
<div class='tp-eyebrow'>The five details that define this board</div>
<div class='tp-terms' style='font-size:2.4rem;margin:.2rem 0 .4rem'>{top_terms}</div>
<div class='tp-caption' style='margin-bottom:2rem'>Concrete attributes a seller would put in a listing title.
They become the middle word of each Depop search.</div>
<div class='tp-cols four'>{''.join(columns)}</div>
</div>
""")


def render_pieces(analysis: BoardAnalysis, n_cols: int) -> None:
    urls = item_depop_urls(analysis)
    cards = []
    for rank, (item_id, row) in enumerate(analysis.recommendations.iterrows(), start=1):
        uri = item_thumb(str(analysis.item_image_path(item_id)))
        img = f"<img src='{uri}' alt='{escape(row.productDisplayName)}'>" if uri else "<div></div>"
        link = (
            f"<a class='tp-link-depop' href='{escape(urls[item_id])}' target='_blank' rel='noopener'>Find on Depop</a>"
            if item_id in urls else ""
        )
        cards.append(
            f"<div class='tp-piece' title='cosine similarity {row.similarity:.3f}'>"
            f"<div class='rank'>No. {rank}</div>{img}"
            f"<div class='name'>{escape(row.productDisplayName)}</div>"
            f"<div class='meta'>{escape(str(row.articleType))} · {escape(str(row.baseColour)).lower()}</div>"
            f"{link}</div>"
        )
    html(f"""
<div>
<div class='tp-caption' style='margin:0 0 1.4rem'>The closest items in the ~44,000-piece catalogue, in order.
Hover a piece for its similarity score.</div>
<div class='tp-grid' style='--cols:{n_cols}'>{''.join(cards)}</div>
</div>
""")


def render_shop(analysis: BoardAnalysis) -> None:
    rows = []
    for row in analysis.depop_links.itertuples(index=False):
        thumbs = "".join(
            f"<img src='{uri}' alt='{escape(name)}' title='{escape(name)}'>"
            for item_id, name in zip(row.item_ids[:5], row.covers[:5])
            if (uri := item_thumb(str(analysis.item_image_path(item_id)), 160))
        )
        n = len(row.covers)
        rows.append(
            f"<div class='tp-shop'><div><div class='q'>{escape(row.query)}</div>"
            f"<div class='count'>{n} matching piece{'s' if n != 1 else ''}</div>"
            f"<div class='thumbs'>{thumbs}</div></div>"
            f"<div>{depop_button(row.url)}</div></div>"
        )
    html(f"""
<div>
<div class='tp-caption' style='margin:0 0 1rem'>Each search reads <i>aesthetic · detail · garment</i> and opens Depop's own search page.
Transpose never queries or scrapes Depop.</div>
{''.join(rows)}
</div>
""")


def render_how_it_works() -> None:
    table_rows = "".join(
        f"<tr><td>{name}</td><td>{p:.2f}</td><td>{r:.2f}</td><td>{f:.2f}</td><td>{n:,}</td></tr>"
        for name, p, r, f, n in CLASSIFIER_REPORT
    )
    html(f"""
<div class='tp-cols' style='grid-template-columns:repeat(auto-fit,minmax(320px,1fr))'>
<div>
{section_title("The pipeline")}
<ol class='tp-steps'>
<li><div><b>Embed the board.</b> Every image goes through FashionCLIP, a vision–language model pretrained on fashion. Its weights stay frozen.</div></li>
<li><div><b>Blend into one vibe vector.</b> The image embeddings are averaged and normalized into a single 512-number summary of the board.</div></li>
<li><div><b>Find the closest pieces.</b> Cosine similarity ranks ~44,000 pre-embedded catalogue items against the vibe vector.</div></li>
<li><div><b>Name the aesthetic.</b> The same vector is compared with text embeddings of 19 style words (zero-shot, no training).</div></li>
<li><div><b>Predict usage.</b> A small neural network trained on the dataset's labels classifies it as casual, formal, sports, ethnic or party.</div></li>
<li><div><b>Pull out garment details.</b> 27 concrete attributes are ranked, then filtered per garment type so shorts never get a “square neck”.</div></li>
<li><div><b>Build Depop searches.</b> <i>aesthetic · detail · garment</i>, for example “clean girl cropped tee”.</div></li>
</ol>
</div>
<div>
{section_title("The usage classifier")}
<div style='display:flex;gap:2.4rem;margin-bottom:1rem;flex-wrap:wrap'>
<div><div class='tp-stat'>{CLASSIFIER_ACCURACY:.1%}</div><div class='tp-caption'>test accuracy</div></div>
<div><div class='tp-stat'>{CLASSIFIER_MACRO_F1:.2f}</div><div class='tp-caption'>macro F1</div></div>
</div>
<div style='overflow-x:auto'><table class='tp-table'>
<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test items</th></tr></thead>
<tbody>{table_rows}</tbody></table></div>
<div class='tp-caption'>512 → 128 → 5 network on frozen embeddings; class-weighted loss, early stopping,
held-out test set. Casual makes up ~78% of the data, so per-class scores matter more than accuracy;
Party has only 29 examples in total.</div>
{section_title("Design decisions")}
<ul style='margin:0;padding-left:1.1rem;display:grid;gap:.4rem'>
<li>No Depop scraping: its terms forbid it, so Transpose only builds search links.</li>
<li>No Pinterest API: it doesn't expose board contents, so boards are uploaded as images.</li>
<li>Catalogue embeddings are computed once and cached, so each analysis takes seconds.</li>
</ul>
</div>
</div>
""")


# ---------------------------------------------------------------------------
# Sidebar — board input + settings
# ---------------------------------------------------------------------------

with st.sidebar:
    html("<div class='tp-eyebrow'>The board</div>")
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
    html("<div class='tp-eyebrow'>Settings</div>")
    top_n_items = st.slider(
        "Pieces to recommend", min_value=6, max_value=24, value=15, step=3,
        help="Takes effect the next time you analyze the board.",
    )
    n_item_cols = st.slider("Grid columns", min_value=3, max_value=6, value=5)

    run = st.button("Analyze board", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# Resolve input images
# ---------------------------------------------------------------------------

images, source_label = [], ""
if uploaded:
    images = [Image.open(f) for f in uploaded]
    source_label = f"{len(images)} uploaded image{'s' if len(images) != 1 else ''}"
elif use_sample and sample_paths:
    images = [Image.open(p) for p in sample_paths]
    source_label = f"{len(images)} sample images"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run:
    if not images:
        st.warning("Add board images first: upload your own, or tick the sample board in the sidebar.")
    else:
        try:
            resources = load_resources()
        except FileNotFoundError as err:
            st.error(str(err))
            st.info(
                "One-time setup: run `notebooks/01_load_data.ipynb`, `03_item_embeddings.ipynb`, "
                "and `06_usage_classifier.ipynb` to generate the cached files."
            )
            st.stop()

        with st.spinner(f"Reading {source_label}…"):
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
        # Persist so later reruns (slider changes, etc.) keep showing the result.
        st.session_state["analysis"] = analysis
        st.session_state["board_thumbs"] = [to_data_uri(img, BOARD_THUMB_PX) for img in images]

analysis = st.session_state.get("analysis")

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

if analysis is None:
    html("""
<div class='tp-hero'>
<div class='tp-wordmark'>Transpose</div>
<div class='tp-tagline'>from Pinterest board to Depop search</div>
<div class='tp-lede'>Save a folder of outfits the way you'd pin them. Transpose reads the board's aesthetic,
names the details that define it, finds the closest pieces, and turns them into Depop searches.</div>
</div>
""")
    if images:
        strip = board_strip([to_data_uri(img, BOARD_THUMB_PX) for img in images], "margin-top:.5rem")
        html(f"<div><div class='tp-eyebrow'>Your board · {escape(source_label)}</div>{strip}</div>")
        html("<div class='tp-caption'>Ready. Press <i>Analyze board</i> in the sidebar.</div>")
    else:
        html("<div class='tp-caption'>Start in the sidebar: upload your own pins, or tick the sample board.</div>")
else:
    html("""
<div class='tp-hero compact'>
<div class='tp-wordmark'>Transpose</div>
<div class='tp-tagline'>from Pinterest board to Depop search</div>
</div>
""")
    tabs = st.tabs(["Overview", "The vibe", "Details", "Pieces", "Shop", "How it works"])
    with tabs[0]:
        render_overview(analysis, st.session_state.get("board_thumbs", []))
    with tabs[1]:
        render_vibe(analysis)
    with tabs[2]:
        render_details(analysis)
    with tabs[3]:
        render_pieces(analysis, n_item_cols)
    with tabs[4]:
        render_shop(analysis)
    with tabs[5]:
        render_how_it_works()
