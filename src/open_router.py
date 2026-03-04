import requests
import json
import os
import sys
from dotenv import load_dotenv

# Ajouter le répertoire app au chemin Python pour les imports relatifs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
    Statistiques d'utilisation :
    Tokens utilisés : 903 (626 pour le prompt, 277 pour la complétion).
    Coût : 0 (aucun coût signalé).
"""

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Charger la clé API depuis le fichier .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "<OPENROUTER_API_KEY>")

# Vérifier si la clé API est définie
if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "<OPENROUTER_API_KEY>":
    print("Erreur : La clé API OPENROUTER_API_KEY n'est pas définie. Veuillez la définir dans le fichier .env.")
    exit(1)

def generate_messages_from_documents(documents, user_query):
    """
    Génère dynamiquement les messages à envoyer à l'API OpenRouter en fonction des documents pertinents.
    """
    # Construire le contexte avec tous les documents pertinents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(
            f"[Document {i}]\n"
            f"PMID: {doc.metadata.get('source', 'N/A')}\n"
            f"Title: {doc.metadata.get('title', 'N/A')}\n"
            f"Abstract: {doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical assistant specialized in evidence-based medicine. "
                "Answer the user's question using ONLY the provided PubMed documents. "
                "Structure your answer clearly with sections. "
                "Always cite the document number (e.g. [Document 1]) when using information from it. "
                "For each treatment mentioned, clearly indicate whether it is: "
                "(1) well-established with strong evidence, "
                "(2) moderately supported but needing more studies, or "
                "(3) experimental/traditional with limited evidence. "
                "Always remind the user to consult a healthcare professional before starting any treatment, "
                "to verify dosages, and that some approaches mentioned may be experimental. "
                "Respond in the same language as the user's question."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {user_query}\n\n"
                f"Here are the relevant PubMed documents to answer this question:\n\n"
                f"{context}"
            )
        }
    ]
    return messages

def stream_openrouter(documents, user_query):
    """
    Version streaming de query_openrouter.
    Génère les tokens de la réponse au fur et à mesure (Server-Sent Events).
    Utilise avec st.write_stream() dans Streamlit pour afficher la réponse en temps réel.
    """
    messages = generate_messages_from_documents(documents, user_query)
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "qwen/qwen3-vl-235b-a22b-thinking",
                "messages": messages,
                "stream": True,
            }),
            stream=True,
            timeout=180,
        )

        if response.status_code != 200:
            yield f"\n❌ Erreur API {response.status_code} : {response.text}"
            return

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                data    = json.loads(data_str)
                delta   = data["choices"][0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    except Exception as e:
        yield f"\n❌ Erreur de connexion : {e}"


def query_openrouter(documents, user_query):
    """
    Envoie une requête à l'API OpenRouter avec les documents pertinents et la question utilisateur.
    """
    messages = generate_messages_from_documents(documents, user_query)

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "qwen/qwen3-vl-235b-a22b-thinking",
                "messages": messages
            })
        )

        # Vérifier le statut de la réponse
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌  Erreur {response.status_code} : {response.text}")
            return None

    except Exception as e:
        print("Une erreur s'est produite :")
        print(e)
        return None

if __name__ == "__main__":
    # Exemple d'utilisation
    from bio_clinical_embeddings import search_similar_documents, create_vector_store, load_documents_from_articles
    from download_pubmed import download_articles

    # Étape 1 : Télécharger les articles depuis PubMed
    query = "prostatitis"
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
        results = search_similar_documents(vector_store, user_query, k=3)

        # Étape 5 : Envoyer les résultats à l'API OpenRouter
        query_openrouter(results, user_query)