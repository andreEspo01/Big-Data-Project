import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase

# =============================
# CONFIGURAZIONE DATABASE
# =============================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # cambia con la tua password

@st.cache_data(ttl=600)
def load_data_from_neo4j(topic_label='politics'):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Connessione al database predefinito "neo4j"
    with driver.session() as session:
        query = f"""
        MATCH (td:TopicDisplay {{topic_label: '{topic_label}'}})
        MATCH (pd:PostDisplay)-[:TALKS_ABOUT]->(td)
        MATCH (ud:UserDisplay)-[:WROTE]->(pd)
        OPTIONAL MATCH (pd)-[:REPLIED_TO]->(parent:PostDisplay)
        RETURN ud.username AS username,
               pd.post_id AS post_id,
               parent.post_id AS parent_post_id,
               td.topic_label AS topic
        LIMIT 100
        """
        result = session.run(query)
        df = pd.DataFrame([record.data() for record in result])
    driver.close()
    return df

# =============================
# SELEZIONE TOPIC
# =============================
topic_filter = st.sidebar.text_input("Topic da visualizzare", value="politics")

# =============================
# CARICAMENTO DATI
# =============================
try:
    df = load_data_from_neo4j(topic_label=topic_filter)
    if df.empty:
        st.warning("Il database Neo4j non ha dati per questo topic, verranno usati dati di test.")
        raise Exception("Empty DB")
except:
    st.info("Uso dati di test")
    import numpy as np
    np.random.seed(42)
    n_posts = 20
    n_users = 5
    df = pd.DataFrame({
        "post_id": range(1, n_posts + 1),
        "username": [f"user{i%5}" for i in range(n_posts)],
        "parent_post_id": np.random.choice([None]+list(range(1, n_posts)), size=n_posts),
        "topic": ["politics"]*n_posts
    })

# =============================
# CONFIGURAZIONE STREAMLIT
# =============================
st.set_page_config(page_title="Social Dashboard", layout="wide")
st.title(f"📊 Social Dashboard - Topic: {topic_filter}")

# =============================
# SEZIONE 1: Overview
# =============================
st.header("1️⃣ Overview")
col1, col2 = st.columns(2)
col1.metric("Numero di post", df['post_id'].nunique())
col2.metric("Numero di utenti", df['username'].nunique())

# =============================
# SEZIONE 2: Rete utenti (Post -> User)
# =============================
st.header("2️⃣ Rete di interazioni")

# Creazione mappatura post_id -> username
post_to_user = dict(zip(df['post_id'], df['username']))

# Generazione edge list (username -> parent username)
edges = df[['username', 'parent_post_id']].dropna().values.tolist()
edges_mapped = [(post_to_user.get(parent), username) for username, parent in edges if post_to_user.get(parent)]

# Costruzione grafo
G = nx.DiGraph()
G.add_edges_from(edges_mapped)

plt.figure(figsize=(7, 7))
nx.draw_networkx(
    G, 
    with_labels=True, 
    node_size=500, 
    node_color="skyblue", 
    alpha=0.7, 
    font_size=10
)
st.pyplot(plt)
