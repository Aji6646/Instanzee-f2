# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from datetime import datetime, date
from transformers import pipeline,AutoModelForSequenceClassification, AutoTokenizer
from fpdf import FPDF
import plotly.express as px
import os
from statistics import mean
from PIL import Image
import cv2
import tempfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from openai import OpenAI
import instaloader
from instaloader import Post
import re  

AutoTokenizer.from_pretrained("twitter-roberta-base-sentiment", cache_dir="./model")
AutoModelForSequenceClassification.from_pretrained("twitter-roberta-base-sentiment", cache_dir="./model")


# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="cardiffnlp/twitter-roberta-base-sentiment",
#     local_dir="./models/twitter-roberta-base-sentiment",
#     local_dir_use_symlinks=False  # ensures full files are copied
# )

# -------------------------------
# Session State and Access Control
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

VALID_TOKENS = {"demo123", "abcxyz", "insta2025"}

if not st.session_state.logged_in:
    st.set_page_config(page_title="Instagram Predictor", layout="centered")
    st.title("Welcome to Instanzee")
    st.subheader("🔐 Enter Access Token")
    st.markdown("Enter a valid token to continue. Use `demo123` to try it out.")
    token = st.text_input("Token", value="demo123", type="password")
    if st.button("🔓 Login"):
        if token in VALID_TOKENS:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid token. Please try again.")
    st.stop()

st.set_page_config(page_title="Instagram Engagement Predictor", layout="centered")
st.sidebar.button("🔒 Logout", on_click=lambda: st.session_state.update({"logged_in": False}))

# -------------------------------
# Load BERT sentiment pipeline
# -------------------------------
# @st.cache_resource
# def load_sentiment_pipeline():
#     try:
#         return pipeline(
#             "sentiment-analysis",
#             model="./models/twitter-roberta-base-sentiment",
#             tokenizer="./models/twitter-roberta-base-sentiment"
#         )
#     except Exception as e:
#         st.warning(f"⚠️ Couldn't load sentiment model: {e}")
#         return None
# @st.cache_resource
# def load_sentiment_pipeline():
#     try:
#         return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
#     except Exception as e:
#         st.warning(f"Couldn't load sentiment model: {e}")
#         return None

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline(
            "sentiment-analysis",
            model="./models/twitter-roberta-base-sentiment",
            tokenizer="./models/twitter-roberta-base-sentiment"
        )
    except Exception as e:
        st.warning(f"⚠️ Couldn't load sentiment model: {e}")
        return None



sentiment_pipe = load_sentiment_pipeline()
if sentiment_pipe is None:
    st.error("Could not initialize sentiment analysis. Some features may not work.")
    st.stop()


# -------------------------------
# AI-generated suggestions
# -------------------------------
def gpt_suggestions(content, modality="video"):
    prompt = f"""
    Analyze the uploaded {content.get('modality', modality)} based on its brightness, motion, and duration.
    Provide three personalized, professional-quality recommendations to improve engagement for social media posting.
    Metrics:
    - Brightness: {content.get('brightness', 0.5):.2f}
    - Motion: {content.get('motion', 'N/A')}
    - Duration: {content.get('duration', 'N/A')} seconds
    Return the suggestions as a bulleted list with no additional commentary.
    """
    
    try:
        # Initialize the OpenAI client
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")   # Always prefer environment variable
        )
        
        # Make the API call
        completion = client.chat.completions.create(
            model="gpt-4-turbo",  # Using a more capable model
            messages=[
                {"role": "system", "content": "You are a social media expert providing concise, actionable recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        # Process the response
        response_text = completion.choices[0].message.content
        suggestions = [line.strip() for line in response_text.split("\n") if line.strip()]
        
        # Ensure we always return exactly 3 suggestions
        if len(suggestions) > 3:
            suggestions = suggestions[:3]
        elif len(suggestions) < 3:
            suggestions.extend(["Consider adjusting lighting for better visibility",
                             "Try adding engaging captions or text overlays",
                             "Experiment with different posting times"])[:3-len(suggestions)]
        
        return suggestions
        
    except Exception as e:
        # Fallback suggestions if API fails
        return [
            "Optimize brightness for better visibility",
            "Increase motion/variety to maintain viewer interest",
            "Adjust duration to match platform best practices"
        ]
    
# -------------------------------
# Feature Extractors
# -------------------------------
def extract_image_features(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))
        np_image = np.array(image)
        brightness = np.mean(np_image) / 255.0
        return brightness, {"brightness": brightness, "modality": "image"}
    except:
        return 0.5, {"brightness": 0.5, "modality": "image"}

def extract_video_features(uploaded_file):
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        brightness_values = []
        motion_magnitude = []
        prev_frame = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            resized = cv2.resize(frame, (224, 224))
            brightness = np.mean(resized)
            brightness_values.append(brightness)

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                motion = np.mean(diff)
                motion_magnitude.append(motion)
            prev_frame = gray

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        os.unlink(tfile.name)

        avg_brightness = np.mean(brightness_values) / 255.0 if brightness_values else 0.5
        avg_motion = np.mean(motion_magnitude) / 255.0 if motion_magnitude else 0.2
        duration_sec = frame_count / fps if fps > 0 else 5

        return avg_brightness, {
            "brightness": avg_brightness,
            "motion": avg_motion,
            "duration": duration_sec,
            "modality": "video"
        }
    except:
        return 0.5, {
            "brightness": 0.5,
            "motion": 0.2,
            "duration": 5.0,
            "modality": "video"
        }
    
def get_bert_sentiment(text):
    if not sentiment_pipe:
        return 0.0, 0.5, "Neutral"
    try:
        result = sentiment_pipe(text[:512])[0]
        label = result['label'].upper()
        score = result['score']
        if "NEGATIVE" in label:
            return -score, 0.8, "Negative"
        elif "NEUTRAL" in label:
            return 0.0, 0.5, "Neutral"
        else:
            return score, 0.8, "Positive"
    except:
        return 0.0, 0.5, "Neutral"

def get_comment_sentiment_score(comments):
    """Calculate average sentiment score for a list of comments"""
    if not isinstance(comments, list) or not comments:
        return 0.0
    
    scores = []
    for comment in comments[:10]:  # Analyze first 10 comments to save time
        try:
            result = sentiment_pipe(comment[:512])[0]  # Truncate to 512 tokens
            label = result['label']
            score = result['score']
            
            # Convert sentiment labels to numeric scores
            if label == 'LABEL_0':  # Negative
                scores.append(-1 * score)
            elif label == 'LABEL_1':  # Neutral
                scores.append(0)
            else:  # Positive (LABEL_2)
                scores.append(1 * score)
        except Exception as e:
            continue  # Skip if analysis fails
            
    return mean(scores) if scores else 0.0  # Return average or 0 if no valid scores

# -------------------------------
# Load and Train Model
# -------------------------------
@st.cache_resource
def load_assets():
    df = pd.read_csv("Instagram - Posts.csv")
    df.dropna(subset=["description", "likes", "followers", "date_posted", "content_type"], inplace=True)

    df["likes"] = pd.to_numeric(df["likes"], errors="coerce")
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce")
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")
    df.dropna(subset=["likes", "followers", "num_comments"], inplace=True)

    if "hashtags" not in df.columns:
        df["hashtags"] = "[]"
    if "latest_comments" not in df.columns:
        df["latest_comments"] = "[]"
    if "location" not in df.columns:
        df["location"] = "Unknown"

    df["caption_length"] = df["description"].apply(len)
    df["hashtag_count"] = df["hashtags"].apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith("[") else len(re.findall(r"#\w+", str(x))))
    df["bert_polarity"], df["bert_subjectivity"] = zip(*df["description"].apply(lambda x: get_bert_sentiment(x)[:2]))
    df["comment_sentiment"] = df["latest_comments"].apply(lambda x: get_comment_sentiment_score(eval(x)) if isinstance(x, str) and x.startswith("[") else 0)
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df.dropna(subset=["date_posted"], inplace=True)
    df["post_hour"] = df["date_posted"].dt.hour
    df["weekday"] = df["date_posted"].dt.day_name()
    df["month"] = df["date_posted"].dt.month
    df["location"] = df["location"].fillna("Unknown").replace("", "Unknown")
    df["visual_score"] = 0.5

    numeric_features = ["caption_length", "hashtag_count", "bert_polarity", "bert_subjectivity", 
                      "comment_sentiment", "post_hour", "month", "followers", "visual_score"]
    categorical_cols = ["content_type", "weekday", "location"]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    df = df.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    X = pd.concat([df[numeric_features], encoded_df], axis=1)
    y_likes = df["likes"].reset_index(drop=True)
    y_comments = df["num_comments"].reset_index(drop=True)

    model_likes = RandomForestRegressor(n_estimators=100, random_state=42)
    model_likes.fit(X, y_likes)

    model_comments = RandomForestRegressor(n_estimators=100, random_state=42)
    model_comments.fit(X, y_comments)

    best_day = df.groupby("weekday")["likes"].mean().idxmax()

    return {
        "model_likes": model_likes,
        "model_comments": model_comments,
        "encoder": encoder,
        "numeric_features": numeric_features,
        "categorical_cols": categorical_cols,
        "all_features": numeric_features + encoder.get_feature_names_out(categorical_cols).tolist(),
        "best_day": best_day
    }

assets = load_assets()
model_likes = assets["model_likes"]
model_comments = assets["model_comments"]
encoder = assets["encoder"]
numeric_features = assets["numeric_features"]
categorical_cols = assets["categorical_cols"]
encoded_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
best_day = assets["best_day"]
all_features = assets["all_features"]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📸 Instagram Post Performance Predictor")
tab1, tab2 = st.tabs(["📝 Create New Post", "🔗 Analyze IG Link"])

with tab1:
    st.subheader("🧪 Simulate a New Instagram Post")
    caption = st.text_area("Caption")
    hashtags = st.text_input("Hashtags", placeholder="#fun #travel")
    followers = st.number_input("Follower Count", value=1000)
    post_hour = st.slider("Posting Hour", 0, 23, 12)
    post_date = st.date_input("Post Date", value=date.today())
    content_type = st.selectbox("Content Type", ["Reel", "Video", "Carousel"])

    st.markdown("### 📷 Upload a Thumbnail Image or Video")
    image_file = st.file_uploader("Upload Image (optional)", type=["jpg", "jpeg", "png"], key="image")
    video_file = st.file_uploader("Upload Video (optional)", type=["mp4", "mov"], key="video")

    visual_score = 0.5  # Default value
    content_info = None  # For GPT suggestions

    if image_file:
        visual_score, content_info = extract_image_features(image_file)
        st.image(image_file, caption="Uploaded Image Preview", use_column_width=True)
    elif video_file:
        visual_score, content_info = extract_video_features(video_file)
        st.video(video_file)

    if st.button("🔮 Predict Performance"):
        # Prepare numeric features
        input_features = {
            "caption_length": len(caption),
            "hashtag_count": len([tag for tag in hashtags.split() if tag.startswith("#")]),
            "bert_polarity": get_bert_sentiment(caption)[0],
            "bert_subjectivity": get_bert_sentiment(caption)[1],
            "comment_sentiment": 0.0,
            "post_hour": post_hour,
            "month": post_date.month,
            "followers": followers,
            "visual_score": visual_score
        }

        # Prepare categorical features
        cat_features = pd.DataFrame([[content_type, post_date.strftime("%A"), "Unknown"]], 
                                  columns=categorical_cols)
        
        # One-hot encode categorical features
        encoded_cat = encoder.transform(cat_features)
        encoded_df = pd.DataFrame(encoded_cat, columns=encoded_feature_names)

        # Create numeric features DataFrame
        numeric_df = pd.DataFrame([input_features], columns=numeric_features)
        
        # Combine numeric and encoded categorical features
        final_input = pd.concat([numeric_df, encoded_df], axis=1)
        
        # Ensure we only have the expected columns in the right order
        final_input = final_input[all_features]
        
        # Make predictions
        try:
            predicted_likes = int(model_likes.predict(final_input)[0])
            predicted_comments = int(model_comments.predict(final_input)[0])
            engagement_rate = (predicted_likes / followers) * 100 if followers else 0

            st.subheader("📊 Predicted Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("👍 Likes", f"{predicted_likes:,}")
            col2.metric("💬 Comments", f"{predicted_comments:,}")
            col3.metric("📈 Engagement", f"{engagement_rate:.2f}%")

            fig = px.bar(x=["Likes", "Comments"], y=[predicted_likes, predicted_comments],
                         labels={"x": "Metric", "y": "Count"}, title="Predicted Engagement Breakdown")
            st.plotly_chart(fig)

            sentiment_label = get_bert_sentiment(caption)[2]
            st.markdown("### 🧠 Sentiment Analysis")
            st.markdown(f"- Sentiment: `{sentiment_label}`")
            st.markdown(f"- Polarity: `{input_features['bert_polarity']:.2f}`")
            st.markdown(f"- Subjectivity: `{input_features['bert_subjectivity']:.2f}`")

            st.info(f"📅 Best day to post: **{best_day}**")

            result_df = pd.DataFrame([{
                "Caption": caption,
                "Hashtag Count": input_features["hashtag_count"],
                "Followers": followers,
                "Post Type": content_type,
                "Polarity": input_features["bert_polarity"],
                "Subjectivity": input_features["bert_subjectivity"],
                "Visual Score": visual_score,
                "Predicted Likes": predicted_likes,
                "Predicted Comments": predicted_comments,
                "Engagement Rate": engagement_rate
            }])

            if image_file or video_file:
                with st.expander("💡 AI Suggestions to Improve Your Media"):
                    st.markdown("Fetching suggestions from GPT-3.5...")
                    suggestions = gpt_suggestions(content_info)
                    for s in suggestions:
                        st.markdown(f"- {s}")

            # Export options
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="instagram_prediction.csv", mime='text/csv')

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Instagram Post Prediction Report", ln=True, align="C")
            for col in result_df.columns:
                pdf.cell(200, 10, txt=f"{col}: {result_df[col][0]}", ln=True)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download PDF", data=pdf_bytes, file_name="instagram_prediction.pdf", mime='application/pdf')

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()

with tab2:
    st.subheader("🔍 Analyze Public Instagram Post Link")
    post_link = st.text_input("Paste IG Post URL")
    ig_user = st.text_input("Your Instagram Username (for session file)", placeholder="e.g. tommy")

    if st.button("📥 Fetch & Analyze Post") and post_link and ig_user:
        try:
            # Clean the post link
            clean_link = post_link.split('?')[0].strip()
            
            # Validate Instagram URL format
            if not re.match(r'^https?://(www\.)?instagram\.com/p/[^/]+/?$', clean_link):
                st.error("❌ Invalid Instagram post URL format. Please use format: https://www.instagram.com/p/XXXXX/")
                st.stop()
                
            shortcode = clean_link.rstrip('/').split('/')[-1]
            st.info(f"Extracted shortcode: {shortcode}")

            # Initialize Instaloader with proper settings
            L = instaloader.Instaloader(
                quiet=True,
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_geotags=False,
                download_comments=False,
                save_metadata=False
            )
            
            # Try to load session
            session_file = f"{ig_user}.session"
            try:
                L.load_session_from_file(ig_user, filename=session_file)
                st.success(f"✅ Using existing session: {session_file}")
            except FileNotFoundError:
                st.warning("Some data may be limited for private accounts.")
                L.context.username = ig_user  # Still helpful for rate limiting

            try:
                # Get the post
                post = Post.from_shortcode(L.context, shortcode)
                
                # Extract post features
                caption = post.caption or ""
                caption_len = len(caption)
                hashtag_count = len(re.findall(r"#\w+", caption))
                bert_polarity, bert_subjectivity, sentiment_label = get_bert_sentiment(caption)
                post_hour = post.date_utc.hour
                post_date = post.date_utc
                weekday = post_date.strftime("%A")
                month = post.date_utc.month
                
                # Determine content type
                if post.is_video:
                    content_type = "Video"
                elif post.typename == "GraphSidecar":
                    content_type = "Carousel"
                else:
                    content_type = "Reel"
                
                location = post.location.name if post.location else "Unknown"
                followers = post.owner_profile.followers if post.owner_profile else 5000  # Fallback

                # Prepare input features
                input_features = {
                    "caption_length": caption_len,
                    "hashtag_count": hashtag_count,
                    "bert_polarity": bert_polarity,
                    "bert_subjectivity": bert_subjectivity,
                    "comment_sentiment": 0.0,  # Can't get comments without authentication
                    "post_hour": post_hour,
                    "month": month,
                    "followers": followers,
                    "visual_score": 0.5  # Default since we can't analyze media
                }

                # Prepare categorical features
                cat_features = pd.DataFrame([[content_type, weekday, location]], 
                                          columns=categorical_cols)
                
                # One-hot encode categorical features
                encoded_cat = encoder.transform(cat_features)
                encoded_df = pd.DataFrame(encoded_cat, columns=encoded_feature_names)

                # Create numeric features DataFrame
                numeric_df = pd.DataFrame([input_features], columns=numeric_features)
                
                # Combine features
                final_input = pd.concat([numeric_df, encoded_df], axis=1)[all_features]

                # Make predictions
                predicted_likes = int(model_likes.predict(final_input)[0])
                predicted_comments = int(model_comments.predict(final_input)[0])
                engagement_rate = (predicted_likes / followers) * 100

                # Display results
                st.subheader("📊 Analysis Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("👍 Predicted Likes", f"{predicted_likes:,}")
                col2.metric("💬 Predicted Comments", f"{predicted_comments:,}")
                col3.metric("📈 Engagement Rate", f"{engagement_rate:.2f}%")

                st.markdown("### 📝 Post Details")
                st.markdown(f"- **Caption:** {caption[:200] + '...' if len(caption) > 200 else caption}")
                st.markdown(f"- **Posted on:** {post_date.strftime('%Y-%m-%d %H:%M')} ({weekday})")
                st.markdown(f"- **Content Type:** {content_type}")
                if location != "Unknown":
                    st.markdown(f"- **Location:** {location}")
                
                st.markdown("### 🧠 Sentiment Analysis")
                st.markdown(f"- **Sentiment:** `{sentiment_label}`")
                st.markdown(f"- **Polarity:** `{bert_polarity:.2f}` (Range: -1 to 1)")
                st.markdown(f"- **Subjectivity:** `{bert_subjectivity:.2f}` (Range: 0 to 1)")

                st.info(f"💡 Based on historical data, your best posting day is **{best_day}**")

            except instaloader.exceptions.PrivateProfileNotFollowed:
                st.error("🔒 This account is private and you don't follow it")
            except instaloader.exceptions.QueryReturnedBadRequestException:
                st.error("❌ Post not found or unavailable")
            except Exception as e:
                st.error(f"❌ Error analyzing post: {str(e)}")

        except Exception as e:
            st.error(f"❌ Failed to process URL: {str(e)}")