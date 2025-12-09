"""
ESAG Art Hub – Final Streamlit App
==================================

This file contains the Streamlit application for the Eastern Suburbs Art Group
(ESAG) Art Hub. The app showcases artwork from local artists, allowing users to
browse and filter pieces by artist, suburb and price range. It also features
an AI‑powered recommendation engine to suggest artworks based on a textual
search query. A key improvement over the previous version is that the sidebar
"Refresh" button now resets all filters and the search box, ensuring a clean
state when the user wants to start over.

To run the app, install Streamlit (`pip install streamlit`) and then run
`streamlit run esag_art_hub.py`. Make sure that a file named `Arts.csv` is in
the same directory (or adjust the path in `load_artworks`). The app reads
credentials for the OpenAI API either from Streamlit's secrets or from an
environment variable.
"""

import os
import base64
import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image


# ---------------------------------------------------------------
# 1. OpenAI API Key Loading
#
# You can directly include the key between the big brackets inside
# double quotation marks but for safety you can create a variable to
# avoid exposing your API. This checks Streamlit's secrets first and
# falls back to an environment variable loaded via dotenv.
# ---------------------------------------------------------------
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------
# 2. Page Setup & Styling part (from font, color, size, etc.)
# ---------------------------------------------------------------
st.set_page_config(page_title="ESAG Art Hub", page_icon="🎨", layout="wide")
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 3. Converting the Google Drive links to thumbnail for Gallery section
#
# We convert any Google Drive link into its thumbnail URL. This is cached
# because generating the same thumbnail repeatedly is expensive.
# ---------------------------------------------------------------
@st.cache_data
def make_drive_display_url(link: Any) -> Optional[str]:
    """Convert a Google Drive link to a thumbnail link if possible."""
    if not isinstance(link, str):
        return None
    if "drive.google.com" in link and "/d/" in link:
        file_id = link.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/thumbnail?id={file_id}"
    return link


# ---------------------------------------------------------------
# 4. Loading the Artwork Data from CSV file
#
# The CSV file should have at least the following columns: artist,
# title, price, suburb, tag, and link. Extra columns are
# ignored. We also add a numeric price column for filtering.
# ---------------------------------------------------------------

def load_artworks() -> List[Dict[str, Any]]:
    """
    Load artworks from the CSV file. This version does NOT cache,
    so any updates to Arts.csv will be immediately reflected.
    """
    df = pd.read_csv("Arts.csv")
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert Drive link to thumbnail
    df["image"] = df["link"].apply(make_drive_display_url)

    # Ensure safe defaults for all expected columns
    for col in ["artist", "title", "price", "suburb", "tag"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    # Convert price to numeric for filtering
    if "price" in df.columns:
        df["price_num"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce",
        )
    else:
        df["price_num"] = pd.NA

    return df.to_dict("records")




# ---------------------------------------------------------------
# Load artworks once
ARTWORKS = load_artworks()


# ---------------------------------------------------------------
# 5. AI Recommendation Helper (Short output + real sort)
# ---------------------------------------------------------------
def recommend_artworks_with_openai(query: str, artworks: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Ask OpenAI for up to 5 short recommendations (1 line each), then reorder the
    gallery to follow the AI's order exactly. If no query is provided, or if
    the API call fails, the original order is preserved.
    """
    if not query:
        return None, artworks

    # Give the AI more context (titles + tags + suburb) so it picks literal matches
    catalogue_lines = []
    for a in artworks:
        catalogue_lines.append(
            f"- {a.get('title','Untitled')} (tags: {a.get('tag','')}, suburb: {a.get('suburb','')})"
        )

    prompt = (
        f"You are an expert art curator. The buyer request is: '{query}'.\n"
        "Here is the artwork catalogue with tags and suburbs:\n"
        + "\n".join(
            [
                f"- {a.get('title','Untitled')} "
                f"(tags: {a.get('tag','')}, "
                f"suburb: {a.get('suburb','')}, "
                f"price: {a.get('price_num','')})"
                f"(artist: {a.get('artist','')}, "

                for a in artworks
            ]
        )
        + "\n\nRules:\n"
        " Match the buyer’s search query with highest precision using exact words or close synonyms. \n"

        "• Use this priority order when evaluating artworks:\n"
        "•Title (highest priority)\n"
        "•Tags (second priority)\n"
        "•Image content (objects, scenes, colours, mood, landmarks)\n"
        "•Logical inference (only if no direct matches exist)\n"
        "•If the query is a place or scene (e.g., “Sydney”, “Paris”, “beach”, “mountain”, “harbour”), prefer artworks depicting that place, related landmarks, or similar environments.\n"
        "•If multiple artworks match, choose the top 5 most relevant.\n"
        "•If the query is broad (e.g., “nice art”, “colourful”), recommend the 5 artworks with the strongest title/tag alignment and visual coherence.\n"
        "•Output format (strict):\n"
        "•Up to 5 recommendations\n"
        "•1 short sentence each\n"
        "•Do not add explanations, greetings, or extra text. Only output the final list.\n"
        
        "Top Recommendations:\n"
        "1. <Artwork Title> – <Reason>\n"
        "2. ...\n"
        "...\n"
        "5. ..."
        
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.4,
        )
        text = response.choices[0].message.content.strip()

        # Extract up to 10 titles, accepting en dash, em dash or colon separators after the title
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
        ordered: List[Dict[str, Any]] = []
        for ai_title in ranked_titles[:10]:
            best_idx: Optional[int] = None
            best_score = 0.0
            for idx, art in enumerate(artworks):
                title_l = str(art.get("title", "")).lower()
                score = difflib.SequenceMatcher(None, ai_title, title_l).ratio()
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
# 6. Header & Tabs section
# ---------------------------------------------------------------
# The Home tab contains the gallery and filters. Additional tabs provide
# information about the group, privacy policy and contact details.
# ---------------------------------------------------------------
st.markdown("<h1>Eastern Suburbs Art Group (ESAG)</h1>", unsafe_allow_html=True)
st.subheader("Discover Art That Speaks to You")

home_tab, about_tab, privacy_tab, contact_tab = st.tabs([
    "🏠 Home",
    "About Us",
    "Privacy Policy",
    "Contact Us",
])


# Maintain state about whether the user has clicked the home tab again
if "just_clicked_home" not in st.session_state:
    st.session_state["just_clicked_home"] = False

# Detect first visit to Home tab
if st.session_state.get("active_tab") != "home":
    st.session_state["active_tab"] = "home"
    st.session_state["just_clicked_home"] = False
else:
    # Refresh only if user manually clicked the Home tab again
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


# ======================================================================================
# Button to trigger Home refresh manually
#
# This button now clears the search box and all filters immediately before
# triggering a rerun. Using session state keys ensures the widgets show
# fresh values upon rerun. This resolves the issue where only the gallery
# refreshed while the filters and search box remained unchanged.
# ====================================================================================
if st.sidebar.button("⟳ Refresh"):
    # Clear stored values for the widgets
    st.session_state["q"] = ""
    st.session_state["artist_sel"] = "All"
    st.session_state["suburb_sel"] = "All"
    st.session_state["price_sel"] = "All"
    st.session_state["just_clicked_home"] = False
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# ===============================================================
#  HOME TAB
# ===============================================================
with home_tab:
    st.sidebar.header("Find Your Art")
    # Use keys to bind widget values to session state so we can reset them
    query: str = st.sidebar.text_input(
        "Describe what you’re looking for",
        placeholder="e.g. abstract calm ocean scene",
        key="q",
    )

    artists = sorted(set(a["artist"] for a in ARTWORKS if a["artist"]))
    suburbs = sorted(set(a["suburb"] for a in ARTWORKS if a["suburb"]))
    # The selectboxes are bound to session state via keys so that clearing
    # session values resets them to their default ("All").
    a_sel: str = st.sidebar.selectbox(
        "Filter by Artist",
        ["All"] + artists,
        key="artist_sel",
    )
    s_sel: str = st.sidebar.selectbox(
        "Filter by Suburb",
        ["All"] + suburbs,
        key="suburb_sel",
    )

    # Price Range Filter (bands)
    price_band_labels = [
        "All",
        "100 - 500",
        "500 - 1000",
        "1000 - 2000",
        "2000 - 5000",
        "5000 - 10000",
    ]
    p_sel: str = st.sidebar.selectbox(
        "Filter by Price Range (AUD)",
        price_band_labels,
        key="price_sel",
    )

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
            a
            for a in filtered
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
            st.markdown(
                f"**{art.get('title','Untitled')}**<br>*by {art.get('artist','Unknown')}*",
                unsafe_allow_html=True,
            )

            if art.get("suburb"):
                st.caption(f"📍 {art['suburb']}")
            if art.get("price"):
                st.caption(f"💲 {art['price']}")

            tags = [art.get("tag", "")]
            tags_html = "".join(
                [f"<span class='tag'>🏷️ {t}</span>" for t in tags if t]
            )
            if tags_html:
                st.markdown(
                    f"<div class='tags'>{tags_html}</div>", unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)


# ===============================================================
#  Extra TABS Page
# ===============================================================
with about_tab:
    st.markdown("## About Us")
    st.markdown(
        "**Eastern Suburbs Art Group (ESAG)** connects local artists with art lovers worldwide."
    )

with privacy_tab:
    st.markdown("## Privacy Policy")
    st.markdown(
        "No personal data is stored. Uploaded images are temporary for AI use only."
    )

with contact_tab:
    st.markdown("## Contact Us")
    st.markdown(
        "Visit our blog: [easternsuburbsartgroup.blogspot.com](http://easternsuburbsartgroup.blogspot.com)"
    )
