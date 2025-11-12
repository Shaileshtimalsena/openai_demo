# ===============================================================
#  ESAG Art Hub – Final Streamlit App (Updated for New CSV)
# ===============================================================

import streamlit as st
import pandas as pd
import openai, os, base64, re
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

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower()
    # Add image preview
    df["image"] = df["link"].apply(make_drive_display_url)

    # Ensure missing values are safe
    for col in ["artist", "title", "price", "suburb", "tag 1", "tag 2"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    return df.to_dict("records")


ARTWORKS = load_artworks()

# ---------------------------------------------------------------
# 5. AI Recommendation Helper
# ---------------------------------------------------------------
def recommend_artworks_with_openai(query, artworks):
    if not query:
        return None, artworks

    # ✨ Updated prompt: ask for logical explanations per artwork
    prompt = (
        f"You are an art curator. A buyer says: '{query}'.\n"
        f"Here are the available artworks: {', '.join([a.get('title', 'Untitled') for a in artworks])}.\n\n"
        "Please:\n"
        "1. Recommend up to 3 artworks that best match what the buyer is looking for.\n"
        "2. For each recommendation, give a short and logical reason (e.g., theme, emotion, value, colour, symbolism).\n"
        "3. Format your reply clearly as:\n"
        "1. <Artwork Title> – <Reason>\n"
        "2. ...\n"
        "3. ...\n"
        "Keep explanations concise but meaningful."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.6,
        )

        text = response.choices[0].message.content.strip()

        # 🧠 Extract artwork titles mentioned by AI
        matched_titles = re.findall(r"^\s*\d+\.\s*([^-:\n]+)", text, flags=re.MULTILINE)
        matched_titles = [t.strip() for t in matched_titles if t.strip()]

        # 🧩 Fuzzy reorder the artworks so AI matches come first
        ordered = []
        for t in matched_titles:
            for art in artworks:
                if art["title"].lower().startswith(t.lower()) or t.lower() in art["title"].lower():
                    if art not in ordered:
                        ordered.append(art)
                        break

        # add remaining artworks afterward
        for art in artworks:
            if art not in ordered:
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

home_tab, about_tab, privacy_tab, contact_tab = st.tabs(["Home", "About Us", "Privacy Policy", "Contact Us"])

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

    filtered = ARTWORKS
    if a_sel != "All":
        filtered = [a for a in filtered if a["artist"] == a_sel]
    if s_sel != "All":
        filtered = [a for a in filtered if a["suburb"] == s_sel]

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

    # --- AI Tagging & Analysis ---
    st.markdown("### AI Tagging & Analysis")
    up = st.file_uploader("Upload artwork (JPG/PNG) for AI analysis", type=["jpg","jpeg","png"], key="ai_uploader")
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded Artwork", use_column_width=True)
        if openai.api_key:
            with st.spinner("Analyzing artwork..."):
                prompt = "Describe theme, colours, and emotion of this artwork."
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
