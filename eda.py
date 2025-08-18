import pandas as pd
import re
from bs4 import BeautifulSoup
import spacy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# === Load spaCy model ===
nlp = spacy.load("en_core_web_sm")

# === File paths ===
FILE_PATH = "/Users/vaibhav/Downloads/subreddit_combined.xlsx"
OUTPUT_CSV = "/Users/vaibhav/Downloads/cleaned_reddit_data.csv"
WORDCLOUD_PNG = "/Users/vaibhav/Downloads/wordcloud.png"
KEYWORD_PNG = "/Users/vaibhav/Downloads/keyword_frequency.png"

# === Load data ===
df = pd.read_excel(FILE_PATH)


# Drop rows with missing or empty post_title/post_text
df = df.dropna(subset=['post_title', 'post_text'])
df = df[~df['post_title'].astype(str).str.strip().eq('')]
df = df[~df['post_text'].astype(str).str.strip().eq('')]

# Fill NaNs in comment_body
df['comment_body'] = df['comment_body'].fillna('')

# === Cleaning functions ===
emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)
url_pattern = re.compile(r'http\S+|www\S+')

def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(url_pattern, '', text)
    text = re.sub(emoji_pattern, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_remove_stopwords_spacy(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# === Apply cleaning and tokenization ===
for col in ['post_title', 'post_text', 'comment_body']:
    df[col] = df[col].astype(str).apply(clean_text)
    df[col] = df[col].apply(tokenize_and_remove_stopwords_spacy)

# === Save cleaned CSV ===
df.to_csv(OUTPUT_CSV, index=False)
print("✅ Cleaned CSV saved to:", OUTPUT_CSV)

# === Combine text ===
df['combined_text'] = df['post_title'] + ' ' + df['post_text'] + ' ' + df['comment_body']
df = df[df['combined_text'].str.strip().astype(bool)]

# === Word Cloud ===
all_text = ' '.join(df['combined_text'])
wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(all_text)

# Save and display word cloud
plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Reddit Mental Health Posts", fontsize=16)
plt.tight_layout()
plt.savefig(WORDCLOUD_PNG)
print("✅ Word cloud saved to:", WORDCLOUD_PNG)
plt.show()

# === Keyword Frequency Bar Chart (% of posts) ===
keywords = ['anxiety', 'depression', 'panic', 'suicidal', 'therapy',
            'stress', 'alone', 'worthless', 'fear', 'hopeless']

total_posts = len(df)
keyword_counts = {
    word: df['combined_text'].str.contains(rf'\b{word}\b', case=False).sum()
    for word in keywords
}
keyword_percentages = {
    word: round((count / total_posts) * 100, 2)
    for word, count in keyword_counts.items()
}
df_keywords = pd.DataFrame({
    'Keyword': list(keyword_percentages.keys()),
    'Percentage': list(keyword_percentages.values())
}).sort_values(by='Percentage', ascending=False)

# Plot and save keyword frequency chart
plt.figure(figsize=(10, 5))
sns.barplot(data=df_keywords, x='Keyword', y='Percentage', palette='Reds_r')
plt.title("🔍 % of Reddit Posts Mentioning Mental Health Keywords", fontsize=14)
plt.ylabel("Percentage of Posts (%)")
plt.xlabel("Keyword")
plt.ylim(0, max(df_keywords['Percentage']) + 2)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(KEYWORD_PNG)
print("✅ Keyword frequency chart saved to:", KEYWORD_PNG)
plt.show()