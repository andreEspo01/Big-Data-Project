import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

# =============================
# CARICAMENTO DATASET
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

st.set_page_config(page_title="Social Dashboard", layout="wide")
st.title("📊 Social Dashboard")

# =============================
# SEZIONE 1: Overview
# =============================
st.header("1️⃣ Overview")
st.write("Statistiche generali sul dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Numero di post", len(df))
col2.metric("Numero di utenti", df['user_id'].nunique())
col3.metric("Comunità", df['community'].nunique())

# =============================
# SEZIONE 2: Sentiment nel tempo
# =============================
st.header("2️⃣ Sentiment nel tempo")
if "publish_date_only" in df.columns:
    sentiment_over_time = df.groupby(["publish_date_only", "sentiment_dominant_roberta"]).size().reset_index(name="count")
    fig = px.line(sentiment_over_time, x="publish_date_only", y="count", color="sentiment_dominant_roberta", title="Sentiment Over Time")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# SEZIONE 3: Emozioni
# =============================
st.header("3️⃣ Distribuzione delle emozioni")
if "emotion_dominant" in df.columns:
    emotion_counts = df["emotion_dominant"].value_counts().reset_index()
    fig = px.bar(emotion_counts, x="index", y="emotion_dominant", title="Distribuzione Emozioni")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# SEZIONE 4: Rete utenti
# =============================
st.header("4️⃣ Rete di interazioni (esempio semplice)")

if "user_id" in df.columns and "parent_user_id" in df.columns:
    edges = df[["user_id", "parent_user_id"]].dropna().values.tolist()
    G = nx.DiGraph()
    G.add_edges_from(edges)

    plt.figure(figsize=(6, 6))
    nx.draw_networkx(G, with_labels=False, node_size=20, alpha=0.6)
    st.pyplot(plt)
