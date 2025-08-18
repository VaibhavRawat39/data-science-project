import praw
import pandas as pd
from datetime import datetime, timezone
import re
import time

# Reddit API credentials
reddit = praw.Reddit(
    client_id='PFJO2I0qraAwbZtpk-NPAQ',
    client_secret='8_em_eriKKj8cD4u26XuMlncdFmv1w',
    user_agent='kd71998'
)

# List of subreddits
subreddits = [
    'schizophrenia', 'Anxiety', 'OCD', 'PTSD',
    'bipolar', 'depression', 'SuicideWatch', 'ADHD', 'socialanxiety',
    'aspergers', 'AutismInWomen', 'AutisticPride', 'CPTSD', 'EDAnonymous',
    'AnorexiaNervosa', 'EatingDisorders', 'BingeEatingDisorder', 'selfharm',
    'stopdrinking', 'addiction', 'DecidingToBeBetter', 'Healthygamergg',
    'KindVoice', 'NonZeroDay', 'MMFB', 'MentalHealthPH', 'workstress',
    'burnout', 'stress', 'offmychest', 'TrueOffMyChest', 'depersonalization',
    'mentalillness', 'lonely', 'needafriend', 'therapy'
]

# Scraping parameters
posts_per_subreddit = 1000
top_comments_per_post = 10
cutoff_timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()

# Target conditions and regex
conditions = [
    "Anxiety Disorder", "Depressive Disorder", "Substance Use", "PTSD",
    "Bipolar Affective Disorder", "OCD", "Eating Disorders",
    "Schizophrenia", "Personality Disorder"
]
condition_patterns = {c: re.compile(rf'\b{re.escape(c)}\b', re.IGNORECASE) for c in conditions}

# Save function
def save_checkpoint(data_chunk, name):
    df = pd.DataFrame(data_chunk)
    filename = f'reddit_checkpoint_{name}.csv'
    df.to_csv(filename, index=False)
    print(f"💾 Saved {len(data_chunk)} rows to {filename}")

# Scrape each subreddit
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    print(f"\n🔍 Scraping r/{subreddit_name}...")

    data = []  # Reset per subreddit
    post_counter = 0
    matched_count = 0
    others_count = 0

    for post in subreddit.new(limit=posts_per_subreddit):
        post_counter += 1
        try:
            time.sleep(0.25)  # faster scraping
            post.comments.replace_more(limit=0)
        except Exception as e:
            print(f"⚠️ Error loading comments for post {post.id}: {e}")
            continue

        comment_count = 0
        for comment in post.comments:
            if comment.created_utc < cutoff_timestamp:
                continue

            text = comment.body
            parent_type = comment.parent_id[:2]

            try:
                parent_author = str(comment.parent().author)
            except:
                parent_author = "[deleted]"

            matched = False
            for condition, pattern in condition_patterns.items():
                if pattern.search(text):
                    matched = True
                    matched_count += 1
                    data.append({
                        'subreddit': subreddit_name,
                        'post_id': post.id,
                        'post_author': str(post.author),
                        'post_title': post.title,
                        'post_text': post.selftext,
                        'post_score': post.score,
                        'post_url': post.url,
                        'comment_id': comment.id,
                        'comment_body': text,
                        'comment_score': comment.score,
                        'comment_author': str(comment.author),
                        'comment_created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                        'matched_condition': condition,
                        'parent_id': comment.parent_id,
                        'parent_author': parent_author,
                        'reply_to_post': parent_type == 't3'
                    })
                    break  # one condition per comment for speed

            if not matched:
                others_count += 1
                data.append({
                    'subreddit': subreddit_name,
                    'post_id': post.id,
                    'post_author': str(post.author),
                    'post_title': post.title,
                    'post_text': post.selftext,
                    'post_score': post.score,
                    'post_url': post.url,
                    'comment_id': comment.id,
                    'comment_body': text,
                    'comment_score': comment.score,
                    'comment_author': str(comment.author),
                    'comment_created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                    'matched_condition': "Others",
                    'parent_id': comment.parent_id,
                    'parent_author': parent_author,
                    'reply_to_post': parent_type == 't3'
                })

            comment_count += 1
            if comment_count >= top_comments_per_post:
                break

        if post_counter % 250 == 0:
            print(f"🛑 Reached {post_counter}/{posts_per_subreddit} posts in r/{subreddit_name}. Sleeping 30 seconds...")
            time.sleep(30)

    # Save everything once per subreddit
    if len(data) > 0:
        save_checkpoint(data, f"{subreddit_name}_final")
        print(f"📁 {subreddit_name}: {len(data)} total rows — {matched_count} matched, {others_count} others")
    else:
        print(f"⚠️ No comments collected for r/{subreddit_name}")

print("\n✅ All subreddits processed.")