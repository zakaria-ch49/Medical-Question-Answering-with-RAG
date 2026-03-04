from langchain_core.documents import Document

# ─── Modèle d'embedding ───────────────────────────────────────────────────────
# BAAI/bge-base-en : meilleur modèle pour la recherche sémantique en anglais.
# Requiert le préfixe "Represent this sentence for searching relevant passages: "
# uniquement pour les QUESTIONS (pas pour les documents).
MODEL_NAME = "BAAI/bge-base-en"

# Préfixe BGE requis pour les requêtes (améliore la pertinence de la recherche)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Lazy singleton — not instantiated at import time so that unit tests
# and CI environments that don't have torch/sentence-transformers installed
# can still import this module without error.
_embeddings = None


def get_embeddings():
    """Return the shared HuggingFaceEmbeddings singleton (lazy-loaded)."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
        print(f"Chargement du modèle d'embedding : {MODEL_NAME} ...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # Requis pour BGE (cosine similarity)
        )
    return _embeddings

def load_documents_from_articles(articles):
    """
    Charge les articles téléchargés sous forme de documents utilisables par FAISS.
    """
    return [
        Document(page_content=article["abstract"], metadata={"source": article["pmid"], "title": article["title"]})
        for article in articles
    ]

def create_vector_store(documents):
    """
    Crée un index vectoriel FAISS à partir des documents.
    """
    from langchain_community.vectorstores import FAISS  # noqa: PLC0415
    print("Création de l'index vectoriel FAISS ...")
    return FAISS.from_documents(documents, get_embeddings())

def search_similar_documents(vector_store, query, k=2):
    """
    Effectue une recherche sémantique dans l'index vectoriel.
    """
    bge_query = BGE_QUERY_PREFIX + query
    print(f"\nRecherche pour : '{query}'")
    results = vector_store.similarity_search(bge_query, k=k)

    print("\nRésultats similaires :")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [PMID: {doc.metadata['source']}] {doc.metadata['title']}\n     {doc.page_content[:200]}...")
    return results

def search_with_score(vector_store, query, k=10, score_threshold=0.30, min_results=3):
    """
    Recherche sémantique avec score de similarité cosinus (BGE + normalize=True).
    Distance L2 normalisée : proche de 0 = très pertinent.
    Garantit toujours au moins min_results documents en complétant si besoin.
    """
    bge_query = BGE_QUERY_PREFIX + query
    # Récupérer plus de candidats que nécessaire
    results_with_scores = vector_store.similarity_search_with_score(bge_query, k=max(k, min_results * 2))
    # Filtrer par seuil
    filtered = [(doc, score) for doc, score in results_with_scores if score <= score_threshold]
    # Si on n'a pas assez de résultats, compléter avec les meilleurs restants
    if len(filtered) < min_results:
        seen_ids = {id(doc) for doc, _ in filtered}
        extras = [(doc, score) for doc, score in results_with_scores if id(doc) not in seen_ids]
        filtered += extras[:min_results - len(filtered)]
    return filtered

if __name__ == "__main__":
    from download_pubmed import download_articles  # noqa: PLC0415

    # Exemple de pipeline complet
    query = "prostatitis"

    # Étape 1 : Télécharger les articles depuis PubMed
    print("Téléchargement des articles depuis PubMed...")
    articles = download_articles(query, retmax=100)

    if not articles:
        print("Aucun article téléchargé. Fin du programme.")
    else:
        # Étape 2 : Charger les articles en tant que documents
        documents = load_documents_from_articles(articles)

        # Étape 3 : Créer l'index vectoriel FAISS
        vector_store = create_vector_store(documents)

        # Étape 4 : Effectuer une recherche sémantique
        user_query = "Quels sont les traitements pour la prostatite ?"
        search_similar_documents(vector_store, user_query, k=3)