# ===============================================================
#  ESAG Art Hub – Final Prototype (Dynamic CSV + AI + PDF/JPG Support)
# ===============================================================

import streamlit as st
import pandas as pd
import openai, os, base64, re, requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes

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
# 3. Load Artworks from GitHub CSV
# ---------------------------------------------------------------
@st.cache_data
def load_artworks():
    # TODO: Replace <username> and <repo> with your actual GitHub details
    url = "https://raw.githubusercontent.com/<username>/<repo>/main/Arts.csv"
    df = pd.read_csv(url)
    return df.to_dict("records")

ARTWORKS = load_artworks()

# ---------------------------------------------------------------
# 4. Helper: Convert PDF to Image
# ---------------------------------------------------------------
@st.cache_data
def pdf_to_image_base64(pdf_url):
    try:
        response = requests.get(pdf_url)
        if response.status_code != 200:
            return None
        images = convert_from_bytes(response.content, first_page=1, last_page=1)
        img = images[0]
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        st.warning(f"Could not preview PDF: {e}")
        return None

# ---------------------------------------------------------------
# 5. AI Recommendation (based on text query)
# ---------------------------------------------------------------
def recommend_artworks_with_openai(query, artworks):
    """Use OpenAI to recommend and rank artworks matching buyer description."""
    if not query:
        return None, artworks

    prompt = f"""
    You are an AI art curator. The buyer is looking for: "{query}".
    Here are the artworks available:
    {', '.join([art['title'] for art in artworks])}.
    Rank the 3 most relevant artworks and explain briefly.
    Respond like:
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
        pattern = r"\d+\.\s*([^\n-]+)"
        matched_titles = re.findall(pattern, text)
        reordered = []
        titles_lower = [t.strip().lower() for t in matched_titles]
        for title in titles_lower:
            for art in artworks:
                if art["title"].lower().startswith(title):
                    reordered.append(art)
                    break
        for art in artworks:
            if art not in reordered:
                reordered.append(art)
        return text, reordered
    except Exception as e:
        st.error(f"OpenAI recommendation error: {e}")
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

    # Search & Filters
    query = st.sidebar.text_input("Describe what you’re looking for", placeholder="e.g. abstract calm ocean scene")

    artists = sorted(set(a["artist"] for a in ARTWORKS))
    price_ranges = sorted(set(a["price_range"] for a in ARTWORKS))

    selected_artist = st.sidebar.selectbox("Filter by Artist", ["All"] + artists)
    selected_price = st.sidebar.selectbox("Filter by Price Range", ["All"] + price_ranges)

    filtered = ARTWORKS
    if selected_artist != "All":
        filtered = [a for a in filtered if a["artist"] == selected_artist]
    if selected_price != "All":
        filtered = [a for a in filtered if a["price_range"] == selected_price]

    # AI Recommendations
    st.markdown("### Recommended Artworks")
    if query and openai.api_key:
        with st.spinner("Finding best matches using OpenAI..."):
            recommendations, ordered_artworks = recommend_artworks_with_openai(query, filtered)
        if recommendations:
            st.success("**AI Recommendations:**")
            st.write(recommendations)
        else:
            st.warning("No AI recommendations found. Showing all artworks.")
            ordered_artworks = filtered
    else:
        ordered_artworks = filtered

    # Display Gallery
    st.markdown("### Gallery")
    cols = st.columns(3)
    for i, art in enumerate(ordered_artworks):
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            link = art.get("link", "")
            if link:
                file_id = link.split("/d/")[1].split("/")[0] if "/d/" in link else ""
                if link.endswith(".pdf"):
                    img_b64 = pdf_to_image_base64(link)
                    if img_b64:
                        st.image(f"data:image/png;base64,{img_b64}", use_column_width=True)
                    else:
                        st.warning("PDF preview not available.")
                else:
                    thumb_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                    st.image(thumb_url, use_column_width=True)
            st.markdown(f"**{art['title']}**<br>*by {art['artist']}*", unsafe_allow_html=True)
            st.caption(f"{art['price_range']} • {art['artist']}")
            st.markdown("</div>", unsafe_allow_html=True)

    # Upload for AI Tagging
    st.markdown("### AI Tagging & Analysis")
    uploaded = st.file_uploader("Upload artwork (JPG/PNG) to analyze with AI", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Artwork", use_column_width=True)
        if openai.api_key:
            st.info("Using OpenAI for real analysis…")
            with st.spinner("Analyzing artwork with GPT-4o-mini…"):
                prompt = ("You are an art expert. Analyze the uploaded artwork and describe its "
                          "theme, color palette, and emotion briefly.")
                img_b64 = base64.b64encode(uploaded.read()).decode("utf-8")
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Analyze this artwork."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}
                        ],
                    )
                    st.success("**AI Analysis Result:**")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")
        else:
            st.warning("No API key detected — running in demo mode.")

# ===============================================================
#  ABOUT / PRIVACY / CONTACT
# ===============================================================
with about_tab:
    st.markdown("## About Us")
    st.markdown("**Eastern Suburbs Art Group (ESAG)** is a Sydney-based creative start-up "
                "connecting local artists with art lovers everywhere.")

with privacy_tab:
    st.markdown("## Privacy Policy")
    st.markdown("This prototype does not store or share personal data. "
                "Uploaded images are used temporarily for AI analysis only.")

with contact_tab:
    st.markdown("## Contact Us")
    st.markdown("Visit our blog for updates: "
                "[easternsuburbsartgroup.blogspot.com]"
                "(http://easternsuburbsartgroup.blogspot.com)")
