import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/vaibhav/Downloads/cleaned_reddit_data 2.csv")


required_cols = ["subreddit", "post_author", "comment_author"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Column '{c}' not found in dataframe. "
                         f"Available columns: {df.columns.tolist()}")

post_df = df[["subreddit", "post_author"]].dropna()
post_df = post_df.rename(columns={"post_author": "user"})

comment_df = df[["subreddit", "comment_author"]].dropna()
comment_df = comment_df.rename(columns={"comment_author": "user"})

user_sub_df = pd.concat([post_df, comment_df], ignore_index=True)


user_subreddits = (
    user_sub_df.groupby("user")["subreddit"]
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)



G = nx.Graph()


for sub in sorted(df["subreddit"].dropna().unique()):
    G.add_node(sub)


for _, row in user_subreddits.iterrows():
    subs = row["subreddit"]
    if len(subs) < 2:
        continue
    for s1, s2 in itertools.combinations(subs, 2):
        if G.has_edge(s1, s2):
            G[s1][s2]["weight"] += 1
        else:
            G.add_edge(s1, s2, weight=1)

print(f"Total subreddits (nodes): {G.number_of_nodes()}")
print(f"Total co-occurrence edges: {G.number_of_edges()}")


isolated = [n for n, d in G.degree() if d == 0]
G.remove_nodes_from(isolated)


if G.number_of_nodes() == 0:
    raise RuntimeError("Graph is empty after removing isolated nodes.")

components = list(nx.connected_components(G))
largest_cc = max(components, key=len)
core = G.subgraph(largest_cc).copy()

print(f"Core component nodes: {core.number_of_nodes()}")
print(f"Core component edges: {core.number_of_edges()}")


pos = nx.kamada_kawai_layout(core)


deg = dict(core.degree())
max_deg = max(deg.values()) if deg else 1
node_sizes = [300 + 1200 * (deg[n] / max_deg) for n in core.nodes()]  # 300–1500


weights = nx.get_edge_attributes(core, "weight")
max_w = max(weights.values()) if weights else 1
edge_widths = [0.5 + 4.5 * (weights[e] / max_w) for e in core.edges()]  # 0.5–5


top_nodes = sorted(deg, key=deg.get, reverse=True)[:TOP_N]
labels = {n: n for n in top_nodes}  # only label top hubs

plt.figure(figsize=(10, 10))
nx.draw_networkx_edges(
    core,
    pos,
    width=edge_widths,
    edge_color="lightgray",
    alpha=0.7
)

# Nodes
nx.draw_networkx_nodes(
    core,
    pos,
    node_size=node_sizes,
    node_color="skyblue",
    edgecolors="black",
    linewidths=0.7,
    alpha=0.9
)


    core,
    pos,
    labels=labels,
    font_size=9,
    font_weight="bold"
)

plt.title("Subreddit Co-occurrence Network (Core, Sized by Degree)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("subreddit_network_core.png", dpi=300)
plt.show()

print("✅ Saved network plot as 'subreddit_network_core.png'")






em_df = pd.read_csv("/Users/vaibhav/Desktop/Reddit Data/reddit_emotion_roberta_full.csv")


emotion_cols = ['joy', 'optimism', 'anger', 'sadness']
for c in emotion_cols:
    if c not in em_df.columns:
        raise ValueError(f"Emotion column '{c}' not found in reddit_emotion_roberta_full.csv")

em_df['dominant_emotion'] = em_df[emotion_cols].idxmax(axis=1)


em_df_sorted = em_df.sort_values(by=['post_id'])


transitions = []
for post_id, group in em_df_sorted.groupby('post_id'):
    prev = None
    for _, row in group.iterrows():
        current = row['dominant_emotion']
        if prev is not None and prev != current:
            transitions.append((prev, current))
        prev = current


G_em = nx.DiGraph()
for src, dst in transitions:
    if G_em.has_edge(src, dst):
        G_em[src][dst]['weight'] += 1
    else:
        G_em.add_edge(src, dst, weight=1)


plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G_em, seed=42)


weights = [G_em[u][v]['weight'] for u, v in G_em.edges()]
edge_colors = ['green' if w > 25 else 'orange' if w > 10 else 'grey' for w in weights]
edge_widths = [1 + 0.2 * w for w in weights]

nx.draw_networkx_nodes(G_em, pos, node_size=2500, node_color='lightblue', edgecolors='black')
nx.draw_networkx_labels(G_em, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(
    G_em, pos,
    width=edge_widths,
    edge_color=edge_colors,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=20,
    connectionstyle='arc3,rad=0.1'
)
nx.draw_networkx_edge_labels(
    G_em, pos,
    edge_labels={(u, v): G_em[u][v]['weight'] for u, v in G_em.edges()},
    font_color='red', font_size=10
)

plt.title("Thread-Level Emotion Transition Network")
plt.axis('off')
plt.tight_layout()
plt.savefig("emotion_transition_network.png", dpi=300)
plt.show()

print("✅ Saved emotion transition graph as 'emotion_transition_network.png'")