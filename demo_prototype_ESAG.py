# ===============================================================
#  ESAG Art Hub – Final Streamlit App (AI Sort + Working Price Filter)
# ===============================================================

import streamlit as st
import pandas as pd
import openai, os, base64, re, difflib
from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------------------------
# 1. Secure API Key Loading
# ---------------------------------------------------------------
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------
# 2. Page Setup & Styling
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
          box-shadow:0 4px 12px rgba(0,0,0,0.1);}
    div.stButton>button{background-color:#3b5998;color:white;border:none;
          border-radius:10px;padding:8px 20px;font-weight:600;transition:0.3s;}
    div.stButton>button:hover{background-color:#ffd700;color:#4a2600;transform:scale(1.05);}
    footer{visibility:hidden;}

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
    .tags {
        font-size: 0.85em;
        color: #555;
    }
    .tag {
        background-color: #ffe082;
        color: #4a2600;
        border-radius: 8px;
        padding: 2px 8px;
        margin-right: 4px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 3. Helper – Convert Google Drive links to thumbnail URLs
# ---------------------------------------------------------------
@st.cache_data
def make_drive_display_url(link):
    if not isinstance(link, str):
        return None
    if "drive.google.com" in link and "/d/" in link:
        file_id = link.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/thumbnail?id={file_id}"
    return link


# ---------------------------------------------------------------
# 4. Load Artwork Data from Local CSV
# ---------------------------------------------------------------
@st.cache_data
def load_artworks():
    df = pd.read_csv("Arts.csv")
    df.columns = df.columns.str.strip().str.lower()

    # Convert Drive link to thumbnail
    df["image"] = df["link"].apply(make_drive_display_url)

    # Ensure safe defaults for all expected columns
    for col in ["artist", "title", "price", "suburb", "tag 1", "tag 2"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # Convert price to numeric for filtering
    if "price" in df.columns:
        df["price_num"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce"
        )
    else:
        df["price_num"] = pd.NA

    return df.to_dict("records")


ARTWORKS = load_artworks()

# ---------------------------------------------------------------
# 5. AI Recommendation Helper (Short output + real sort)
# ---------------------------------------------------------------


def recommend_artworks_with_openai(query, artworks):
    """
    Ask OpenAI for up to 5 short recommendations (1 line each),
    then reorder the gallery to follow the AI's order exactly.
    """
    if not query:
        return None, artworks

    # Give the AI more context (titles + tags + suburb) so it picks literal matches
    catalogue_lines = []
    for a in artworks:
        catalogue_lines.append(
            f"- {a.get('title','Untitled')} (tags: {a.get('tag 1','')}, {a.get('tag 2','')}, suburb: {a.get('suburb','')})"
        )

    prompt = (
    f"You are an expert art curator. The buyer request is: '{query}'.\n"
    "Here is the artwork catalogue with tags and suburbs:\n"
    + "\n".join([
        f"- {a.get('title','Untitled')} "
        f"(tags: {a.get('tag 1','')}, {a.get('tag 2','')})"
        for a in artworks
    ]) +
    "\n\nRules:\n"
    "• Give highest priority to artworks whose TITLE or TAGS literally mention the buyer’s query words.\n"
    "• Only if no literal matches exist, then choose conceptually related ones based on the analysis of the photo or the art.\n"
    "• If the query is a PLACE (e.g., 'Sydney', 'Paris') or any scene, nature, animals, prefer artworks depicting that place, its skyline, harbour, things, environment or landmarks.\n"
    "• Return up to 5 short recommendations (1 line each) in this format:\n"
    "Top Recommendations:\n"
    "1. <Artwork Title> – <Reason>\n"
    "2. ...\n"
    "...\n"
    "5. ..."
)


    try:
        import re, difflib
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.4,
        )
        text = response.choices[0].message.content.strip()

        # Extract up to 10 titles, accepting "–", "-" or ":" separators after the title
        lines = re.findall(r"^\s*\d+\.\s*([^\n]+)", text, flags=re.MULTILINE)
        ranked_titles = []
        for ln in lines:
            # split on first separator and keep the title part
            t = re.split(r"[–—:-]", ln, maxsplit=1)[0].strip()
            if t:
                ranked_titles.append(t.lower())

        if not ranked_titles:
            # Nothing parsed from AI -> keep original order but still show the text
            return text, artworks

        # Reorder to follow AI order exactly; track USED by index (ints), not dicts
        used_idx = set()
        ordered = []
        for ai in ranked_titles[:10]:
            best_idx = None
            best_score = 0.0
            for idx, art in enumerate(artworks):
                title_l = str(art.get("title", "")).lower()
                score = difflib.SequenceMatcher(None, ai, title_l).ratio()
                if score > best_score:
                    best_score = score
                    best_idx = idx
            # accept the match if reasonably close
            if best_idx is not None and best_idx not in used_idx and best_score >= 0.35:
                ordered.append(artworks[best_idx])
                used_idx.add(best_idx)

        # Append remaining artworks (not listed by AI) in original order
        for idx, art in enumerate(artworks):
            if idx not in used_idx:
                ordered.append(art)

        return text, ordered

    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None, artworks




# ---------------------------------------------------------------
# 6. Header & Tabs
# ---------------------------------------------------------------
st.markdown("<h1>Eastern Suburbs Art Group (ESAG)</h1>", unsafe_allow_html=True)
st.subheader("Discover Art That Speaks to You")

# Tabs with working Home refresh (safe for all filters)
home_tab, about_tab, privacy_tab, contact_tab = st.tabs(["🏠 Home", "About Us", "Privacy Policy", "Contact Us"])

# --- Safe Home-tab refresh logic (fixed: no auto loop) ---
if "just_clicked_home" not in st.session_state:
    st.session_state["just_clicked_home"] = False

# Detect first visit to Home tab
if st.session_state.get("active_tab") != "home":
    st.session_state["active_tab"] = "home"
    st.session_state["just_clicked_home"] = False
else:
    # Refresh only if user *manually clicked* the Home tab again
    if st.session_state["just_clicked_home"]:
        st.session_state["q"] = ""
        st.session_state["artist_sel"] = "All"
        st.session_state["suburb_sel"] = "All"
        st.session_state["price_sel"] = "All"
        st.session_state["just_clicked_home"] = False
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

# Button to trigger Home refresh manually
if st.sidebar.button("⟳ Refresh"):
    st.session_state["just_clicked_home"] = True
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()



# ===============================================================
#  HOME TAB
# ===============================================================
with home_tab:
    st.sidebar.header("Find Your Art")
    query = st.sidebar.text_input("Describe what you’re looking for", placeholder="e.g. abstract calm ocean scene")

    artists = sorted(set(a["artist"] for a in ARTWORKS if a["artist"]))
    suburbs = sorted(set(a["suburb"] for a in ARTWORKS if a["suburb"]))
    a_sel = st.sidebar.selectbox("Filter by Artist", ["All"] + artists)
    s_sel = st.sidebar.selectbox("Filter by Suburb", ["All"] + suburbs)

    # ---- Price Range Filter (bands) ----
    price_band_labels = [
        "All",
        "100 - 500",
        "500 - 1000",
        "1000 - 2000",
        "2000 - 5000",
        "5000 - 10000",
    ]
    p_sel = st.sidebar.selectbox("Filter by Price Range (AUD)", price_band_labels)

    # Apply filters
    filtered = ARTWORKS
    if a_sel != "All":
        filtered = [a for a in filtered if a.get("artist") == a_sel]
    if s_sel != "All":
        filtered = [a for a in filtered if a.get("suburb") == s_sel]

    # Price range filter
    bands = {
        "100 - 500": (100, 500),
        "500 - 1000": (500, 1000),
        "1000 - 2000": (1000, 2000),
        "2000 - 5000": (2000, 5000),
        "5000 - 10000": (5000, 10000),
    }
    if p_sel != "All":
        lo, hi = bands[p_sel]
        filtered = [
            a for a in filtered
            if a.get("price_num") is not None
            and pd.notna(a.get("price_num"))
            and lo <= float(a.get("price_num")) <= hi
        ]

    # --- AI Recommendations ---
    st.markdown("### Start Discovering your Art from Sidebar ")
    if query and openai.api_key:
        with st.spinner("Finding best matches..."):
            rec_text, ordered = recommend_artworks_with_openai(query, filtered)
        if rec_text:
            st.success("**AI Recommendations:**")
            st.write(rec_text)
        else:
            st.warning("No recommendations found.")
            ordered = filtered
    else:
        ordered = filtered

    # --- Gallery Display ---
    st.markdown("### Gallery")
    cols = st.columns(3)
    for i, art in enumerate(ordered):
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            img_url = art.get("image")
            link = art.get("link", "")

            if img_url:
                st.image(img_url, use_column_width=True)
                if link.lower().endswith(".pdf"):
                    st.markdown(f"[📄 View Full PDF]({link})", unsafe_allow_html=True)
            else:
                st.warning("Preview not available.")

            # Artwork info
            st.markdown(f"**{art.get('title','Untitled')}**<br>*by {art.get('artist','Unknown')}*",
                        unsafe_allow_html=True)

            if art.get("suburb"):
                st.caption(f"📍 {art['suburb']}")
            if art.get("price"):
                st.caption(f"💲 {art['price']}")

            tags = [art.get("tag 1",""), art.get("tag 2","")]
            tags_html = "".join([f"<span class='tag'>🏷️ {t}</span>" for t in tags if t])
            if tags_html:
                st.markdown(f"<div class='tags'>{tags_html}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

 

# ===============================================================
#  OTHER TABS
# ===============================================================
with about_tab:
    st.markdown("## About Us")
    st.markdown("**Eastern Suburbs Art Group (ESAG)** connects local artists with art lovers worldwide.")

with privacy_tab:
    st.markdown("## Privacy Policy")
    st.markdown("No personal data is stored. Uploaded images are temporary for AI use only.")

with contact_tab:
    st.markdown("## Contact Us")
    st.markdown("Visit our blog: [easternsuburbsartgroup.blogspot.com](http://easternsuburbsartgroup.blogspot.com)")
