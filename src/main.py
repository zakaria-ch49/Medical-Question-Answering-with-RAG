import os
import sys

# Ajouter le répertoire app au chemin Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from download_pubmed import download_articles
from bio_clinical_embeddings import load_documents_from_articles, create_vector_store, search_with_score
from open_router import query_openrouter, OPENROUTER_API_KEY
import requests, json

# ─── Constantes ───────────────────────────────────────────────────────────────
SEPARATOR    = "═" * 60
SUB_SEP      = "─" * 60
MAX_ARTICLES = 30    # Nombre max d'articles à télécharger (PubMed + FDA)
TOP_K        = 5     # Nombre de documents similaires à récupérer

# ─── Traduction automatique vers l'anglais ────────────────────────────────────
def translate_to_english(question, pubmed_hint):
    """
    Génère une requête PubMed médicale précise en anglais à partir
    de la question de l'utilisateur et du mot-clé PubMed donné.
    Retourne le mot-clé original si la traduction échoue.
    """
    try:
        prompt = (
            f"The user asked this medical question: '{question}'\n"
            f"The user suggested this PubMed keyword: '{pubmed_hint}'\n"
            "Generate a concise and precise PubMed search query in English using MeSH terms when possible "
            "(2-5 words, use standard medical English terminology like 'common cold', 'rhinovirus', 'upper respiratory tract infection', etc.). "
            "Do NOT include generic words like 'treatment' unless essential. "
            "Reply with ONLY the query, nothing else."
        )
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "qwen/qwen3-vl-235b-a22b-thinking",
                "messages": [
                    {"role": "system", "content": "You are a biomedical search expert."},
                    {"role": "user",   "content": prompt}
                ]
            })
        )
        if response.status_code == 200:
            data = response.json()
            # Gérer les modèles de raisonnement (content peut être vide)
            content = data["choices"][0]["message"].get("content", "").strip()
            if not content:
                details = data["choices"][0]["message"].get("reasoning_details", [])
                for d in details:
                    if d.get("type") == "reasoning.text":
                        content = d.get("text", "").strip().splitlines()[0]
                        break
            if content:
                return content
    except Exception:
        pass
    return pubmed_hint  # fallback : retourner le mot-clé original

# ─── Affichage lisible de la réponse ──────────────────────────────────────────
def display_response(response):
    """
    Affiche la réponse de l'API OpenRouter de manière lisible.
    """
    if not response:
        print("\n⚠️  Aucune réponse reçue de l'API.")
        return

    choices = response.get("choices", [])
    if not choices:
        print("\n⚠️  La réponse ne contient pas de contenu.")
        return

    message = choices[0].get("message", {})
    content = message.get("content", "").strip()

    # Fallback : certains modèles de raisonnement mettent le texte dans reasoning_details
    if not content:
        for detail in message.get("reasoning_details", []):
            if detail.get("type") == "reasoning.text":
                content = detail.get("text", "").strip()
                break

    if not content:
        print("\n⚠️  La réponse ne contient pas de contenu.")
        return

    usage = response.get("usage", {})

    print(f"\n{SEPARATOR}")
    print("🤖  RÉPONSE DU MODÈLE")
    print(SEPARATOR)
    print(content)
    print(f"\n{SUB_SEP}")
    print(f"📊  Statistiques  —  "
          f"Prompt : {usage.get('prompt_tokens', '?')} tokens  |  "
          f"Réponse : {usage.get('completion_tokens', '?')} tokens  |  "
          f"Total : {usage.get('total_tokens', '?')} tokens")
    print(SUB_SEP)
    print()
    print("⚠️  AVERTISSEMENT MÉDICAL")
    print(SUB_SEP)
    print("  • Cette réponse est générée automatiquement à partir d'articles PubMed")
    print("    et de fiches médicaments openFDA — elle ne remplace PAS l'avis d'un professionnel de santé.")
    print("  • Vérifiez chaque référence dans sa source originale avant toute utilisation.")
    print("  • Ne dépassez jamais les doses recommandées sans avis médical.")
    print("  • Certaines approches mentionnées peuvent être expérimentales ou")
    print("    non validées dans votre contexte clinique spécifique.")
    print("  • En cas d'urgence médicale, contactez immédiatement un médecin.")
    print(SUB_SEP)

# ─── Affichage lisible des documents trouvés ──────────────────────────────────
def display_documents(results_with_scores):
    """
    Affiche les documents pertinents trouvés avec leur score de pertinence.
    Avec BGE + normalize_embeddings : distance L2 normalisée, score bas = plus pertinent.
    Affiche un badge [FDA] ou un lien PubMed selon la source.
    """
    print(f"\n{SEPARATOR}")
    print(f"📄  DOCUMENTS PERTINENTS TROUVÉS ({len(results_with_scores)})")
    print(SEPARATOR)
    for i, (doc, score) in enumerate(results_with_scores, 1):
        pertinence = "🟢 Très pertinent" if score < 0.2 else "🟡 Pertinent" if score < 0.35 else "🔴 Peu pertinent"
        pmid = doc.metadata['source']
        is_fda = str(pmid).startswith("fda-")
        source_tag = "🟠 [FDA Drug Label]" if is_fda else f"🔵 [PubMed] PMID:{pmid}"
        print(f"\n  [{i}] {source_tag}  |  Score : {score:.3f}  {pertinence}")
        print(f"      Titre : {doc.metadata['title']}")
        print(f"      Résumé : {doc.page_content[:300]}...")
        print(f"  {SUB_SEP}")

# ─── Pipeline principal ────────────────────────────────────────────────────────
def run_rag_pipeline(pubmed_query, user_question, retmax=MAX_ARTICLES, top_k=TOP_K):
    """
    Pipeline RAG complet :
      1. Télécharge les articles PubMed pour 'pubmed_query'
      2. Crée un index vectoriel FAISS
      3. Recherche les documents les plus proches de 'user_question'
      4. Génère une réponse via l'API OpenRouter
    """
    print(f"\n{SEPARATOR}")
    print(f"🔍  QUESTION : {user_question}")
    print(f"🧬  REQUÊTE PUBMED : {pubmed_query}")
    print(SEPARATOR)

    # Étape 0 : Génération d'une requête PubMed précise + traduction de la question
    print("\n🌐  Étape 0/4 — Traduction et optimisation des requêtes en anglais...")
    pubmed_query_en   = translate_to_english(user_question, pubmed_query)
    question_en       = translate_to_english(user_question, user_question)
    if pubmed_query_en.lower() != pubmed_query.lower():
        print(f"✅  Requête PubMed  : « {pubmed_query} »  →  « {pubmed_query_en} »")
    else:
        print(f"✅  Requête PubMed  : « {pubmed_query_en} »")
    if question_en.lower() != user_question.lower():
        print(f"✅  Question FAISS  : « {user_question} »  →  « {question_en} »")

    # Étape 1 : Téléchargement des articles (PubMed + openFDA en parallèle)
    print("\n📥  Étape 1/4 — Téléchargement depuis PubMed + openFDA...")
    articles = download_articles(pubmed_query_en, retmax=retmax)

    if not articles:
        print("⚠️  Aucun document trouvé (ni PubMed ni openFDA). Fin du pipeline.")
        return

    n_fda    = sum(1 for a in articles if a["pmid"].startswith("fda-"))
    n_pubmed = len(articles) - n_fda
    print(f"✅  {len(articles)} documents récupérés (PubMed : {n_pubmed} | FDA : {n_fda}).")

    # Étape 2 : Création de l'index vectoriel
    print("\n🔢  Étape 2/4 — Création de l'index vectoriel...")
    documents    = load_documents_from_articles(articles)
    vector_store = create_vector_store(documents)
    print(f"✅  Index créé avec {len(documents)} documents.")

    # Étape 3 : Recherche sémantique (en anglais pour correspondre aux docs PubMed)
    print(f"\n🔎  Étape 3/4 — Recherche des {top_k} documents les plus pertinents...")
    results_with_scores = search_with_score(vector_store, question_en, k=top_k * 2, score_threshold=0.30)
    results = [doc for doc, _ in results_with_scores[:top_k]]
    display_documents(results_with_scores[:top_k])

    # Étape 4 : Génération de la réponse (question originale + documents en anglais)
    print(f"\n💬  Étape 4/4 — Génération de la réponse par le modèle...")
    response = query_openrouter(results, user_question)
    display_response(response)

# ─── Interface interactive ─────────────────────────────────────────────────────
def main():
    print(f"\n{SEPARATOR}")
    print("🏥  ASSISTANT MÉDICAL RAG — PubMed + openFDA + OpenRouter")
    print(SEPARATOR)
    print("Tapez 'quitter' pour arrêter.\n")

    while True:
        # Question médicale de l'utilisateur
        user_question = input("❓  Votre question médicale : ").strip()
        if user_question.lower() in ("quitter", "quit", "exit", "q"):
            print("\n👋  Au revoir !")
            break
        if not user_question:
            print("⚠️  Veuillez entrer une question valide.\n")
            continue

        # Requête PubMed (mot-clé pour la recherche)
        pubmed_query = input("🧬  Mot-clé PubMed (laissez vide pour utiliser la question) : ").strip()
        if not pubmed_query:
            pubmed_query = user_question

        # Nombre d'articles à télécharger
        retmax_input = input(f"📥  Nombre d'articles à télécharger (défaut : {MAX_ARTICLES}) : ").strip()
        retmax = int(retmax_input) if retmax_input.isdigit() else MAX_ARTICLES

        # Lancer le pipeline
        run_rag_pipeline(pubmed_query, user_question, retmax=retmax, top_k=TOP_K)

        print("\n" + SEPARATOR + "\n")

if __name__ == "__main__":
    main()
