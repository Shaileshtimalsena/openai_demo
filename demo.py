# ===============================================================
#  ESAG Art Hub – AI-Powered Prototype (Final Version with Image Sorting)
# ===============================================================

import streamlit as st
import openai, os, base64, re
from dotenv import load_dotenv
from PIL import Image
import random
from io import BytesIO

# ---------------------------------------------------------------
# 1. Secure API-Key Loading
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# 3. Artwork Data
# ---------------------------------------------------------------
ARTWORKS = [
    {"title": "Beach View", "artist": "Artist 1",
     "image": "https://raw.githubusercontent.com/Shaileshtimalsena/capstone_demo/main/images/Beach.png",
     "price": "AUD $$$"},
    {"title": "Dramatic Clouds", "artist": "Artist 2",
     "image": "https://raw.githubusercontent.com/Shaileshtimalsena/capstone_demo/main/images/Cloud.png",
     "price": "AUD $$$"},
    {"title": "Collision of Colors", "artist": "Artist 5",
     "image": "https://images.pexels.com/photos/1266808/pexels-photo-1266808.jpeg",
     "price": "AUD $$$"},
    {"title": "Monalisa", "artist": "Artist 4",
     "image": "https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg",
     "price": "AUD $$$"},
    {"title": "Mountain with Yellow Flowers", "artist": "Artist 3",
     "image": "https://raw.githubusercontent.com/Shaileshtimalsena/capstone_demo/main/images/flowers.png",
     "price": "AUD $$$"},
]

# ---------------------------------------------------------------
# 4. Helper Functions
# ---------------------------------------------------------------
def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def recommend_artworks_with_openai(query, artworks):
    """Use OpenAI to recommend and rank artworks matching buyer description."""
    if not query:
        return None, artworks

    prompt = f"""
    You are an AI art curator.
    The buyer is looking for: "{query}".
    Here are the artworks available:
    {', '.join([art['title'] for art in artworks])}.
    Rank them from most to least relevant (top 3 only) and briefly explain why.
    Respond strictly like this:
    1. <title> - reason
    2. <title> - reason
    3. <title> - reason
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
        )
        text = response.choices[0].message.content

        # Extract titles using regex
        pattern = r"\d+\.\s*([^\n-]+)"
        matched_titles = re.findall(pattern, text)

        # Reorder artworks according to titles
        reordered = []
        titles_lower = [t.strip().lower() for t in matched_titles]
        for title in titles_lower:
            for art in artworks:
                if art["title"].lower().startswith(title):
                    reordered.append(art)
                    break

        # Add remaining artworks not mentioned
        for art in artworks:
            if art not in reordered:
                reordered.append(art)

        return text, reordered

    except Exception as e:
        st.error(f"OpenAI recommendation error: {e}")
        return None, artworks


# ---------------------------------------------------------------
# 5. Header & Navigation
# ---------------------------------------------------------------
st.markdown("<h1>Eastern Suburbs Art Group (ESAG)</h1>", unsafe_allow_html=True)
st.subheader("Discover Art That Speaks to You")

home_tab, about_tab, privacy_tab, contact_tab = st.tabs(
    ["Home", "About Us", "Privacy Policy", "Contact Us"]
)

# ===============================================================
#  HOME TAB
# ===============================================================
with home_tab:
    st.sidebar.header("Find Your Art")
    query = st.sidebar.text_input(
        "Describe what you’re looking for",
        placeholder="e.g. abstract calm ocean scene",
    )

    # ---------- AI Recommendations ----------
    st.markdown("### Recommended Artworks")
    if query and openai.api_key:
        with st.spinner("Finding best matches using OpenAI..."):
            recommendations, ordered_artworks = recommend_artworks_with_openai(query, ARTWORKS)
        if recommendations:
            st.success("**AI Recommendations:**")
            st.write(recommendations)
        else:
            st.warning("No recommendations available. Showing all artworks.")
            ordered_artworks = ARTWORKS
    else:
        ordered_artworks = ARTWORKS
        st.info("Type something in the sidebar to get AI-based art suggestions.")

    # ---------- Display sorted gallery ----------
    st.markdown("### Gallery")
    cols = st.columns(3)
    for i, art in enumerate(ordered_artworks):
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(art["image"], use_column_width=True)
            st.markdown(f"**{art['title']}** <br>*by {art['artist']}*", unsafe_allow_html=True)
            st.caption(f"{art['price']}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- AI Tagging / Analysis ----------
    st.markdown("### AI Tagging & Analysis")
    uploaded = st.file_uploader(
        "Upload artwork (JPG/PNG) to analyze with AI",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Artwork", use_column_width=True)

        if openai.api_key:
            st.info("Using OpenAI for real analysis…")
            with st.spinner("Analyzing artwork with OpenAI (GPT-4o-mini)…"):
                prompt = (
                    "You are an art expert. Analyze the uploaded artwork and describe "
                    "its theme, color palette, and emotion briefly. Respond like:\n"
                    "Theme: <theme>, Colour palette: <palette>, Emotion: <emotion>."
                )
                img_b64 = image_to_base64(img)
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                       messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Analyze this artwork."},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                            ]}
                        ],
                    )
                    st.success("**AI Analysis Result:**")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")
        else:
            st.warning("No API key detected — running in demo mode.")
            themes = ["Abstract", "Nature", "Portrait", "Modern"]
            colors = ["Warm", "Cool", "Neutral", "Vibrant"]
            emotions = ["Calm", "Joyful", "Mysterious", "Energetic"]
            st.write("**Suggested tags (simulated):**")
            st.write(f"- Theme: **{random.choice(themes)}**")
            st.write(f"- Colour palette: **{random.choice(colors)}**")
            st.write(f"- Emotion: **{random.choice(emotions)}**")

    st.markdown("---")
    st.caption("Eastern Suburbs Art Group – Powered by AI")

# ===============================================================
#  OTHER TABS
# ===============================================================
with about_tab:
    st.markdown("## About Us")
    st.markdown(
        "**Eastern Suburbs Art Group (ESAG)** is a Sydney-based creative start-up "
        "connecting local artists with art lovers everywhere."
    )

with privacy_tab:
    st.markdown("## Privacy Policy")
    st.markdown(
        "This prototype does not store or share personal data. "
        "Uploaded images are used temporarily for AI analysis only."
    )

with contact_tab:
    st.markdown("## Contact Us")
    st.markdown(
        "Reach out via our blog to stay updated on news and events. <br>"
        "**Website:** [easternsuburbsartgroup.blogspot.com]"
        "(http://easternsuburbsartgroup.blogspot.com)",
        unsafe_allow_html=True,
    )
