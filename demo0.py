# ===============================================================
#  ESAG Art Hub – Streamlit (AI-ranked Gallery + Keyword Fallback)
# ===============================================================

import streamlit as st
import pandas as pd
import openai, os, base64, re, difflib
from dotenv import load_dotenv
from PIL import Image
from collections import defaultdict

# ---------------------------------------------------------------
# 1) API key
# ---------------------------------------------------------------
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------
# 2) Page + Styles
# ---------------------------------------------------------------
st.set_page_config(page_title="ESAG Art Hub", page_icon="🎨", layout="wide")
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg,#e0f7fa 0%,#80deea 100%);
  color:#333;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#e0f7fa 0%,#80deea 100%);
}
h1,h2,h3{color:#4a2600;text-shadow:1px 1px 2px rgba(255,255,255,0.7);}
.card{border-radius:14px;background:rgba(255,255,255,0.9);padding:16px;
      box-shadow:0 4px 12px rgba(0,0,0,0.1); position: relative;}
.badge{
  position:absolute; top:10px; left:10px; padding:4px 8px; border-radius:999px;
  background:#ffd700; color:#4a2600; font-weight:700; font-size:12px;
  box-shadow:0 2px 8px rgba(0,0,0,0.15)
}
[data-testid="stImage"] {
  background: none !important;
  box-shadow: none !important;
  border-radius: 10px !important;
  margin-top: -32px !important;
  padding-top: 0 !important;
  overflow: hidden !important;
}
[data-testid="stImage"] img {
  height: 160px !important;
  width: 100% !important;
  object-fit: cover !important;
  border-radius: 10px;
}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 3) Data loading + Drive thumbnails
# ---------------------------------------------------------------
@st.cache_data
def make_drive_display_url(link):
    if not isinstance(link, str):
        return None
    if "drive.google.com" in link and "/d/" in link:
        file_id = link.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/thumbnail?id={file_id}"
    return link

@st.cache_data
def load_artworks():
    # CSV MUST have columns: title, artist, link, price_range, suburb, (optional tags, style, medium, price)
    url = "https://raw.githubusercontent.com/Shaileshtimalsena/openai_demo/refs/heads/main/Arts_.csv"
    df = pd.read_csv(url)
    df["image"] = df["link"].apply(make_drive_display_url)
    # ensure string fields
    for c in ["title","artist","price_range","suburb","tags","style","medium"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df.to_dict("records")

ARTWORKS = load_artworks()

# ---------------------------------------------------------------
# 4) Utility: simple keyword score (fallback / blend)
# ---------------------------------------------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()

def compute_keyword_score(query: str, art: dict) -> float:
    """
    Lightweight relevance scorer using token overlap against multiple fields.
    Ensures we can reorder even if AI fails or is vague.
    """
    q = set(normalize_text(query).split())
    if not q: return 0.0

    # concatenate searchable fields
    fields = []
    for key in ("title","artist","tags","style","medium","suburb","price_range"):
        if key in art and isinstance(art[key], str):
            fields.append(art[key])
    hay = normalize_text(" ".join(fields))
    hset = set(hay.split())

    # weighted overlap
    overlap = q.intersection(hset)
    base = len(overlap) / max(1, len(q))

    # small boost if any query token appears in TITLE specifically
    title_hit = q.intersection(set(normalize_text(art.get("title","")).split()))
    if title_hit:
        base += 0.25
    return min(base, 1.0)

# ---------------------------------------------------------------
# 5) AI recommendation -> list of titles + explanation
# ---------------------------------------------------------------
def get_ai_ranked_titles(query: str, artworks: list):
    """
    Returns (explanation_text, ranked_titles_list).
    If API fails, returns (None, []).
    """
    if not query or not openai.api_key:
        return None, []

    titles_csv = ", ".join([a["title"] for a in artworks])
    prompt = (
        "You are an expert art curator. "
        f"The buyer is looking for: '{query}'. "
        f"Available artworks: {titles_csv}. "
        "Reply with a short recommendation paragraph (2–4 sentences), then a numbered list of the top 3 titles only."
    )

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()

        # extract titles from numbered list lines "1. Title", "2. Title"
        ranked = re.findall(r"^\s*\d+\.\s*([^\n]+?)\s*$", text, flags=re.MULTILINE)
        ranked = [t.strip(" -*\"") for t in ranked if t.strip()]
        return text, ranked[:3]
    except Exception as e:
        return f"(AI error: {e})", []

# ---------------------------------------------------------------
# 6) Blend AI ranking + keyword fallback => final ordered list
# ---------------------------------------------------------------
def order_artworks(query: str, artworks: list):
    """
    Returns: (explanation_text, ordered_artworks, top3_ids)
    - Always reorders.
    - Uses keyword score and AI ranks (if available).
    """
    explanation, ai_titles = get_ai_ranked_titles(query, artworks)

    # Build AI boost map: 1st:1.00, 2nd:0.66, 3rd:0.33 (matched fuzzily to real titles)
    ai_boost = defaultdict(float)
    if ai_titles:
        real_titles = [a["title"] for a in artworks]
        for idx, t in enumerate(ai_titles):
            best = difflib.get_close_matches(t.lower(), [rt.lower() for rt in real_titles], n=1, cutoff=0.4)
            if best:
                ai_boost[best[0]] = [1.0, 0.66, 0.33][idx]

    # Score each artwork
    scored = []
    for art in artworks:
        kw = compute_keyword_score(query, art) if query else 0.0

        # fuzzy similarity against any AI title to help align phrasing
        fuzz = 0.0
        for t in ai_titles:
            fuzz = max(fuzz, difflib.SequenceMatcher(None, art["title"].lower(), t.lower()).ratio())

        # AI boost by name, if present
        name_boost = ai_boost.get(art["title"].lower(), 0.0)

        # Final score blend:
        # keyword relevance (0.6) + fuzzy match to AI (0.25) + explicit AI rank boost (0.15)
        final = 0.60*kw + 0.25*fuzz + 0.15*name_boost
        scored.append((final, art))

    # Sort desc by score
    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [a for _, a in scored]

    # Identify top 3 (for badge)
    top3_ids = set(id(ordered[i]) for i in range(min(3, len(ordered))))
    return explanation, ordered, top3_ids

# ---------------------------------------------------------------
# 7) UI
# ---------------------------------------------------------------
st.markdown("<h1>Eastern Suburbs Art Group (ESAG)</h1>", unsafe_allow_html=True)
st.subheader("Discover Art That Speaks to You")

home_tab, about_tab, privacy_tab, contact_tab = st.tabs(["Home", "About Us", "Privacy Policy", "Contact Us"])

with home_tab:
    st.sidebar.header("Find Your Art")
    query = st.sidebar.text_input("Describe what you’re looking for", placeholder="e.g. abstract calm ocean scene")

    artists = sorted(set(a["artist"] for a in ARTWORKS))
    prices  = sorted(set(a["price_range"] for a in ARTWORKS))
    a_sel = st.sidebar.selectbox("Filter by Artist", ["All"] + artists)
    p_sel = st.sidebar.selectbox("Filter by Price Range", ["All"] + prices)

    filtered = ARTWORKS
    if a_sel != "All":
        filtered = [a for a in filtered if a["artist"] == a_sel]
    if p_sel != "All":
        filtered = [a for a in filtered if a["price_range"] == p_sel]

    # --- AI + Fallback ordering ---
    st.markdown("### Recommended Artworks")
    if query:
        with st.spinner("Ranking best matches..."):
            rec_text, ordered, top3_ids = order_artworks(query, filtered)
        # ALWAYS show explanation (even if AI failed, you'll see the note)
        if rec_text:
            st.info(rec_text)
    else:
        ordered = filtered
        top3_ids = set()

    # --- Gallery ---
    st.markdown("### Gallery (Best match first)")
    cols = st.columns(3)
    for i, art in enumerate(ordered):
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            # badge for top-3
            if id(art) in top3_ids:
                st.markdown("<div class='badge'>⭐ Recommended</div>", unsafe_allow_html=True)

            img_url = art.get("image")
            link = art.get("link", "")

            if img_url:
                st.image(img_url, use_column_width=True)
                if isinstance(link, str) and link.lower().endswith(".pdf"):
                    st.markdown(f"[📄 View Full PDF]({link})", unsafe_allow_html=True)
            else:
                st.warning("Preview not available.")

            st.markdown(
                f"**{art.get('title','Untitled')}**<br>*by {art.get('artist','Unknown')}*",
                unsafe_allow_html=True
            )
            sub = art.get("suburb","")
            pr  = art.get("price_range","")
            if pr or sub:
                st.caption(f"{pr} • {sub}")
            if art.get("price"):
                st.caption(f"💲{art['price']}")

            st.markdown("</div>", unsafe_allow_html=True)

    # --- Optional: debug panel
    with st.expander("Debug (dev)"):
        st.write("Filters →", {"Artist": a_sel, "Price Range": p_sel})
        st.write("Query →", query or "(none)")

    # --- AI Tagging & Analysis ---
    st.markdown("### AI Tagging & Analysis")
    up = st.file_uploader("Upload artwork (JPG/PNG) for AI analysis", type=["jpg","jpeg","png"], key="ai_uploader")
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded Artwork", use_column_width=True)
        if openai.api_key:
            with st.spinner("Analyzing artwork..."):
                try:
                    b64 = base64.b64encode(up.read()).decode("utf-8")
                    res = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":"Describe theme, colors, and emotion of this artwork."},
                            {"role":"user","content":[
                                {"type":"text","text":"Analyze this."},
                                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}
                        ],
                        temperature=0.2,
                    )
                    st.success("**AI Analysis Result:**")
                    st.write(res.choices[0].message.content)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")

with about_tab:
    st.markdown("## About Us")
    st.markdown("**Eastern Suburbs Art Group (ESAG)** connects local artists with art lovers worldwide.")

with privacy_tab:
    st.markdown("## Privacy Policy")
    st.markdown("No personal data is stored. Uploaded images are temporary for AI use only.")

with contact_tab:
    st.markdown("## Contact Us")
    st.markdown("Visit our blog: [easternsuburbsartgroup.blogspot.com](http://easternsuburbsartgroup.blogspot.com)")
