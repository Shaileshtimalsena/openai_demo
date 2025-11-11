# ===============================================================
#  ESAG Art Hub – Final Streamlit App (Dynamic CSV + Drive Thumbnails + AI)
# ===============================================================

import streamlit as st
import pandas as pd
import openai, os, base64, re, requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

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
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 3. Load Artwork Data from GitHub (using Google Drive Thumbnails)
# ---------------------------------------------------------------
@st.cache_data
def make_drive_display_url(link):
    """
    Converts any Google Drive sharing link into a thumbnail preview link.
    Works for both images and PDFs.
    """
    if not isinstance(link, str):
        return None
    if "drive.google.com" in link and "/d/" in link:
        file_id = link.split("/d/")[1].split("/")[0]
        # Always use Google Drive's thumbnail endpoint for speed and compatibility
        return f"https://drive.google.com/thumbnail?id={file_id}"
    return link


@st.cache_data
def load_artworks():
    # ✅ Your actual CSV file hosted on GitHub
    url = "https://raw.githubusercontent.com/Shaileshtimalsena/openai_demo/refs/heads/main/Arts.csv"

    df = pd.read_csv(url)
    # Add displayable image column
    df["image"] = df["link"].apply(make_drive_display_url)
    return df.to_dict("records")


ARTWORKS = load_artworks()

# ---------------------------------------------------------------
# 4. AI Recommendation Helper
# ---------------------------------------------------------------
def recommend_artworks_with_openai(query, artworks):
    if not query:
        return None, artworks
    prompt = f"You are an AI art curator. The buyer is looking for: '{query}'. " \
             f"Here are the available artworks: {', '.join([a['title'] for a in artworks])}. " \
             "Rank the 3 most relevant and explain briefly."
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
        )
        text = response.choices[0].message.content
        pattern = r"\d+\.\s*([^\n-]+)"
        matched = re.findall(pattern, text)
        ordered = []
        for m in matched:
            for art in artworks:
                if art["title"].lower().startswith(m.strip().lower()):
                    ordered.append(art)
                    break
        for art in artworks:
            if art not in ordered:
                ordered.append(art)
        return text, ordered
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None, artworks

# ---------------------------------------------------------------
# 5. Header & Tabs
# ---------------------------------------------------------------
st.markdown("<h1>Eastern Suburbs Art Group (ESAG)</h1>", unsafe_allow_html=True)
st.subheader("Discover Art That Speaks to You")

home_tab, about_tab, privacy_tab, contact_tab = st.tabs(["Home", "About Us", "Privacy Policy", "Contact Us"])

# ===============================================================
#  HOME TAB
# ===============================================================
with home_tab:
    st.sidebar.header("Find Your Art")
    query = st.sidebar.text_input("Describe what you’re looking for", placeholder="e.g. abstract calm ocean scene")

    artists = sorted(set(a["artist"] for a in ARTWORKS))
    prices = sorted(set(a["price_range"] for a in ARTWORKS))
    a_sel = st.sidebar.selectbox("Filter by Artist", ["All"] + artists)
    p_sel = st.sidebar.selectbox("Filter by Price Range", ["All"] + prices)

    filtered = ARTWORKS
    if a_sel != "All":
        filtered = [a for a in filtered if a["artist"] == a_sel]
    if p_sel != "All":
        filtered = [a for a in filtered if a["price_range"] == p_sel]

    # --- AI Recommendations ---
    st.markdown("### Recommended Artworks")
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

    # --- Gallery ---
    st.markdown("### Gallery")
    cols = st.columns(3)
    for i, art in enumerate(ordered):
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            img_url = art.get("image")
            link = art.get("link", "")

            if img_url:
                st.image(img_url, use_column_width=True)
                # Add PDF view button if applicable
                if link.lower().endswith(".pdf"):
                    st.markdown(f"[📄 View Full PDF]({link})", unsafe_allow_html=True)
            else:
                st.warning("Preview not available.")

            st.markdown(f"**{art.get('title','Untitled')}**<br>*by {art.get('artist','Unknown')}*",
                        unsafe_allow_html=True)
            st.caption(f"{art.get('price_range','')} • {art.get('suburb','')}")
            if art.get("price"):
                st.caption(f"💲{art['price']}")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- AI Tagging & Analysis ---
    st.markdown("### AI Tagging & Analysis")
    up = st.file_uploader("Upload artwork (JPG/PNG) for AI analysis", type=["jpg","jpeg","png"], key="ai_uploader")
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded Artwork", use_column_width=True)
        if openai.api_key:
            with st.spinner("Analyzing artwork..."):
                prompt = "Describe theme, colors, and emotion of this artwork."
                b64 = base64.b64encode(up.read()).decode("utf-8")
                try:
                    res = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":prompt},
                            {"role":"user","content":[
                                {"type":"text","text":"Analyze this."},
                                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}
                        ],
                    )
                    st.success("**AI Analysis Result:**")
                    st.write(res.choices[0].message.content)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")

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
