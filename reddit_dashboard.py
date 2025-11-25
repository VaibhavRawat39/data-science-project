import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


EMOTION_PATH = "/Users/vaibhav/Desktop/Reddit Data/reddit_emotion_roberta_full.csv"
VA_PATH      = "/Users/vaibhav/Desktop/Reddit Data/reddit_with_valence_arousal.csv"

st.set_page_config(
    page_title="Reddit Mental Health Dashboard",
    layout="wide",
    page_icon=""
)


sns.set_theme(style="whitegrid")


st.markdown(
    """
    <style>
        /* Main background */
        .stApp {
            background: radial-gradient(circle at top left, #1f2933 0, #020617 45%, #000000 100%);
            color: #e5e7eb;
        }

        /* Tweak default text */
        h1, h2, h3, h4 {
            color: #f9fafb !important;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        p, span, div {
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Metric cards */
        .metric-card {
            padding: 0.75rem 1rem 0.75rem 1rem;
            border-radius: 0.75rem;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.4);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
        }

        /* Section container */
        .section-card {
            padding: 1rem 1.25rem;
            border-radius: 0.9rem;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 12px 35px rgba(0,0,0,0.35);
            margin-bottom: 1rem;
        }

        /* Dataframe wrapper */
        .stDataFrame {
            background-color: #020617 !important;
        }

        /* Slider label color fix */
        .stSlider label {
            color: #e5e7eb !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.98);
            border-right: 1px solid rgba(51, 65, 85, 0.85);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    emo = pd.read_csv(EMOTION_PATH)
    va  = pd.read_csv(VA_PATH)

  
    va = va.drop_duplicates(subset=["post_id"])

    df = pd.merge(
        emo,
        va[["post_id", "valence", "arousal"]],
        on="post_id",
        how="left"
    )

    emotion_cols = ["joy", "optimism", "anger", "sadness"]
    df["dominant_emotion"] = df[emotion_cols].idxmax(axis=1)

    return df

df = load_data()
emotion_cols = ["joy", "optimism", "anger", "sadness"]


st.sidebar.title("🔧 Filters")

all_subs = sorted(df["subreddit"].dropna().unique().tolist())
selected_subs = st.sidebar.multiselect(
    "Select subreddits to view",
    options=all_subs,
    default=all_subs[:5] if len(all_subs) > 5 else all_subs
)

if selected_subs:
    filtered = df[df["subreddit"].isin(selected_subs)].copy()
else:
    filtered = df.copy()

st.sidebar.markdown(f"**Posts in view:** {len(filtered):,}")


st.markdown(
    """
    <h1> Reddit Mental Health Emotion Dashboard</h1>
    <p style="color:#cbd5f5; font-size:0.95rem; margin-bottom:0.5rem;">
        Interactive view of emotional patterns, valence–arousal space, and subreddit-level summaries
        from mental health-related Reddit communities.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total posts (filtered)", f"{len(filtered):,}")
    st.markdown("</div>", unsafe_allow_html=True)

with m2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Unique subreddits", filtered["subreddit"].nunique())
    st.markdown("</div>", unsafe_allow_html=True)

with m3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Avg valence", f"{filtered['valence'].mean():.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

with m4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Avg arousal", f"{filtered['arousal'].mean():.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Average Emotion Scores by Subreddit")

    avg_emotion = (
        filtered.groupby("subreddit")[emotion_cols]
        .mean()
        .sort_values("sadness", ascending=False)
        .head(15)  # top 15 for readability
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    avg_emotion.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average score")
    ax.set_xlabel("Subreddit")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown(
        "<p style='font-size:0.85rem;color:#9ca3af;'>Subreddits are ordered by average sadness score to highlight "
        "the most distressed communities in the current selection.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧾 Dominant Emotion Distribution")

    dom_counts = filtered["dominant_emotion"].value_counts().reindex(emotion_cols)

    fig, ax = plt.subplots(figsize=(5, 4))
    dom_counts.plot(
        kind="bar",
        color=["#22c55e", "#3b82f6", "#f97316", "#ef4444"],
        ax=ax,
    )
    ax.set_ylabel("Number of posts")
    ax.set_xlabel("Emotion")
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown(
        "<p style='font-size:0.85rem;color:#9ca3af;'>Each post is assigned a dominant emotion based on the highest "
        "RoBERTa probability across joy, optimism, anger, and sadness.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


c3, c4 = st.columns(2)

with c3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🌀 Circumplex Mapping (Valence–Arousal)")

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(filtered) > 5000:
        sample = filtered.sample(5000, random_state=42)
    else:
        sample = filtered

    scatter = ax.scatter(
        sample["valence"],
        sample["arousal"],
        c=sample["valence"],
        cmap="coolwarm",
        alpha=0.4,
        s=10
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Valence (Positive ↔ Negative)")
    ax.set_ylabel("Arousal (High ↔ Low)")
    ax.set_title("Posts in Valence–Arousal Space")
    cbar = fig.colorbar(scatter, ax=ax, label="Valence")
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown(
        "<p style='font-size:0.85rem;color:#9ca3af;'>Points in the lower-left represent low-valence, low-arousal "
        "states (e.g. numbness, hopelessness), while upper-right points reflect high-valence, high-arousal affect "
        "(e.g. excitement, relief).</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📈 Emotion Score Distributions")

    fig, ax = plt.subplots(figsize=(6, 5))
    for emo in emotion_cols:
        sns.kdeplot(filtered[emo], label=emo, ax=ax, linewidth=2)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("KDE of Emotion Scores")
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown(
        "<p style='font-size:0.85rem;color:#9ca3af;'>This shows how emotion probabilities are distributed across "
        "the dataset, indicating how often each emotion is strongly expressed.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📚 Subreddit-Level Emotion Summary")

sub_summary = (
    filtered.groupby("subreddit")[emotion_cols + ["valence", "arousal"]]
    .mean()
    .round(3)
    .reset_index()
)

st.dataframe(sub_summary, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📝 Example Posts in Current Selection")

n_examples = st.slider("Number of example posts", 3, 30, 5)

example_cols = [
    "subreddit",
    "post_title",
    "post_text",
    "dominant_emotion",
    "joy",
    "optimism",
    "anger",
    "sadness",
    "valence",
    "arousal",
]

st.dataframe(
    filtered[example_cols].head(n_examples),
    use_container_width=True
)
st.markdown(
    "<p style='font-size:0.85rem;color:#9ca3af;'>Use this section to qualitatively inspect how the model’s "
    "emotion and valence–arousal scores align with real posts.</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)