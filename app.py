import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# =============================
# CONFIGURAZIONE DATABASE
# =============================
NEO4J_URI = "bolt://192.168.1.129:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # aggiorna con la tua password

@st.cache_data(ttl=600)
def load_data_from_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # Tutti gli utenti
        user_query = """
        MATCH (u:UserDisplay)
        RETURN u.username AS username,
               u.sentiment_vader_mean AS sentiment_vader_mean,
               u.sentiment_textblob_mean AS sentiment_textblob_mean,
               u.subjectivity_textblob_mean AS subjectivity_textblob_mean,
               u.toxicity_toxigen_mean AS toxicity_toxigen_mean,
               u.post_count AS post_count,
               u.platform_nunique AS platform_nunique,
               u.community_nunique AS community_nunique
        """
        user_result = session.run(user_query)
        df_users = pd.DataFrame([record.data() for record in user_result])

        # Tutti i post
        post_query = """
        MATCH (p:PostDisplay)<-[:WROTE]-(u:UserDisplay)
        OPTIONAL MATCH (p)-[:TALKS_ABOUT]->(t:TopicDisplay)
        RETURN p.post_id AS post_id,
               p.interaction_type AS interaction_type,
               u.username AS username,
               p.publish_date_only AS publish_date,
               p.clean_content AS clean_content,
               p.processed_text AS processed_text,
               p.sentiment_vader AS sentiment_vader,
               p.sentiment_textblob AS sentiment_textblob,
               p.subjectivity_textblob AS subjectivity_textblob,
               p.platform AS platform,
               p.community AS community,
               t.topic_label AS topic_label
        """
        post_result = session.run(post_query)
        df_posts = pd.DataFrame([record.data() for record in post_result])

        # Tutti i topic
        topic_query = """
        MATCH (t:TopicDisplay)
        RETURN t.topic_label AS topic_label,
               t.topic_id AS topic_id
        """
        topic_result = session.run(topic_query)
        df_topics = pd.DataFrame([record.data() for record in topic_result])

    driver.close()
    return df_users, df_posts, df_topics

# =============================
# CARICAMENTO DATI
# =============================
df_users, df_posts, df_topics = load_data_from_neo4j()

# =============================
# CONFIGURAZIONE STREAMLIT
# =============================
st.set_page_config(page_title="Advanced Social Research Dashboard", layout="wide")
st.title("📊 Advanced Social Research Dashboard")

# =============================
# SEZIONE 1: Overview globale
# =============================
st.header("1️⃣ Overview globale")
col1, col2, col3 = st.columns(3)
col1.metric("Numero di post", df_posts['post_id'].nunique())
col2.metric("Numero di utenti", df_users['username'].nunique())
col3.metric("Numero di topic", df_topics['topic_label'].nunique())

# =============================
# SEZIONE 2: Analisi utenti
# =============================
st.header("2️⃣ Analisi utenti")
st.subheader("📊 Post per utente")
user_post_counts = df_posts.groupby('username')['post_id'].count().reset_index()
user_post_counts.columns = ['username', 'num_posts']
fig_users = px.bar(user_post_counts, x='username', y='num_posts', title="Numero di post per utente")
st.plotly_chart(fig_users, use_container_width=True)

st.subheader("📊 Sentiment medio utenti")
fig_sentiment_users = px.bar(df_users, x='username', 
                             y=['sentiment_vader_mean','sentiment_textblob_mean'],
                             title="Sentiment medio per utente",
                             barmode='group')
st.plotly_chart(fig_sentiment_users, use_container_width=True)

# =============================
# SEZIONE 3: Analisi topic e preferenze
# =============================
st.header("3️⃣ Analisi topic")
topic_counts = df_posts['topic_label'].value_counts().reset_index()
topic_counts.columns = ['topic', 'num_posts']
fig_topics = px.pie(topic_counts, names='topic', values='num_posts', title="Distribuzione post per topic")
st.plotly_chart(fig_topics, use_container_width=True)

st.subheader("🧩 Preferenze utenti per topic")
user_topic = df_posts.groupby(['username','topic_label']).size().reset_index(name='num_posts')
fig_user_topic = px.sunburst(user_topic, path=['username','topic_label'], values='num_posts', title="Distribuzione topic per utente")
st.plotly_chart(fig_user_topic, use_container_width=True)

# =============================
# SEZIONE 4: Network e influenza
# =============================
st.header("4️⃣ Network utenti e influenza")
post_to_user = dict(zip(df_posts['post_id'], df_posts['username']))
edges = df_posts[['username','post_id']].copy()
edges['parent_post_id'] = df_posts['post_id']  # assumiamo le relazioni REPLIED_TO siano presenti in df_posts
edges_mapped = [(post_to_user.get(parent), username) for username, parent in zip(edges['username'], edges['parent_post_id']) if post_to_user.get(parent)]

G = nx.DiGraph()
G.add_edges_from(edges_mapped)
influence = dict(G.in_degree())
node_sizes = [influence.get(node,1)*200 for node in G.nodes()]

plt.figure(figsize=(8,8))
nx.draw_networkx(G,
                 with_labels=True,
                 node_size=node_sizes,
                 node_color="skyblue",
                 alpha=0.7,
                 font_size=10)
st.pyplot(plt)

# =============================
# SEZIONE 5: Clustering utenti
# =============================
st.header("5️⃣ Clustering utenti per comportamento")
features = ['sentiment_vader_mean','sentiment_textblob_mean','subjectivity_textblob_mean',
            'toxicity_toxigen_mean','post_count','platform_nunique','community_nunique']
df_users_clust = df_users.fillna(0)[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_users_clust)

k = st.slider("Numero di cluster", 2, 10, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_users['cluster'] = labels

fig_cluster = px.scatter(df_users, x='post_count', y='toxicity_toxigen_mean', color='cluster',
                         hover_data=['username','sentiment_vader_mean','sentiment_textblob_mean'],
                         title="Clustering utenti")
st.plotly_chart(fig_cluster, use_container_width=True)

# =============================
# SEZIONE 6: Tabella completa
# =============================
st.header("6️⃣ Tabella completa utenti")
st.dataframe(df_users)

st.header("6️⃣ Tabella completa post")
st.dataframe(df_posts)
