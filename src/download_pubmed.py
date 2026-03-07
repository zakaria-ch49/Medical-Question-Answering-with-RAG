import re
import requests
import xml.etree.ElementTree as ET
import os
import time
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─── Constantes réseau ────────────────────────────────────────────────────────
DATA_DIR        = "./data"
CONNECT_TIMEOUT = 10   # secondes pour établir la connexion
READ_TIMEOUT    = 60   # secondes pour lire la réponse (PubMed search)
READ_TIMEOUT_XL = 180  # pour les gros batches efetch
BATCH_SIZE      = 20   # articles par requête efetch (réduit pour fiabilité)


# ─── Session HTTP avec retry / backoff automatique ────────────────────────────
def _make_session(retries: int = 4, backoff: float = 0.5) -> requests.Session:
    """
    Crée une session requests avec retry exponentiel sur les erreurs réseau
    et les codes HTTP 429, 500, 502, 503, 504.
    Délais backoff : 1.5s → 3s → 6s → 12s (pour retries=4, backoff=1.5).
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,        # réessaie sur ReadTimeout
        connect=retries,     # réessaie sur ConnectTimeout
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


# ─── Utilitaires ─────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Nettoie un texte : espaces, caractères spéciaux (sauf . et ,)."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,]", "", text)
    return text


def get_data_file(query, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    sanitized = query.replace(" ", "_").lower()[:80]
    return os.path.join(data_dir, f"{sanitized}_data.txt")


# ─── PubMed : recherche ────────────────────────────────────────────────────────
def search_pubmed(query: str, retmax: int = 500) -> list:
    """
    Retourne la liste des PMIDs correspondant à la requête (jusqu'à retmax).
    Utilise une session avec retry + timeout augmenté.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"}
    session = _make_session()
    try:
        response = session.get(url, params=params,
                               timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        response.raise_for_status()
        return response.json()["esearchresult"]["idlist"]
    except requests.exceptions.RequestException as e:
        print(f"[PubMed] Erreur recherche : {e}")
        return []


# ─── PubMed : téléchargement par batch avec split récursif ───────────────────
def fetch_articles_batch(pmids: list, attempt: int = 0) -> list:
    """
    Télécharge le titre + abstract de plusieurs articles PubMed en une requête.
    En cas d'échec réseau, découpe le batch en deux et réessaie (max 3 niveaux).
    """
    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    session = _make_session(retries=2, backoff=0.5)

    # Timeout proportionnel à la taille du batch : ~2s par article, min 30s
    read_timeout = max(30, len(pmids) * 2)

    try:
        response = session.get(url, params=params,
                               timeout=(CONNECT_TIMEOUT, read_timeout))
        response.raise_for_status()
        root = ET.fromstring(response.content)

        articles = []
        for art_el in root.findall(".//PubmedArticle"):
            pmid_el  = art_el.find(".//PMID")
            pmid     = pmid_el.text if pmid_el is not None else "N/A"
            title_el = art_el.find(".//ArticleTitle")
            title    = "".join(title_el.itertext()).strip() if title_el is not None else ""

            abstract_parts = []
            for abs_text in art_el.iter("AbstractText"):
                label = abs_text.get("label", "")
                text  = "".join(abs_text.itertext()).strip()
                if text:
                    abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(abstract_parts)

            if abstract.strip():
                articles.append({
                    "pmid":     pmid,
                    "title":    preprocess_text(title),
                    "abstract": preprocess_text(abstract),
                })
        return articles

    except (requests.exceptions.RequestException, ET.ParseError) as e:
        if attempt < 2 and len(pmids) > 1:
            print(f"[PubMed] Erreur batch, découpage en deux et réessai… (niveau {attempt+1}/2)")
            mid   = len(pmids) // 2
            half1 = fetch_articles_batch(pmids[:mid], attempt + 1)
            time.sleep(0.3)
            half2 = fetch_articles_batch(pmids[mid:], attempt + 1)
            return half1 + half2
        else:
            print(f"[PubMed] Batch ignoré après {attempt} découpages : {e}")
            return []
    except Exception as e:
        print(f"[PubMed] Erreur inattendue batch fetch : {e}")
        return []


# ─── openFDA : nettoyage de requête ──────────────────────────────────────────
def _clean_query_for_fda(query: str) -> list:
    """
    Convertit une requête PubMed (MeSH, booléens, guillemets…) en une liste
    de requêtes simplifiées pour openFDA, du plus précis au plus large.

    Exemple :
      "rheumatoid arthritis"[MeSH] AND treatment
      → ["rheumatoid arthritis treatment", "rheumatoid arthritis", "arthritis"]
    """
    # 1. Supprimer les tags MeSH et de champ : [MeSH], [tiab], [tw], [majr]…
    cleaned = re.sub(r"\[[\w\s/]+\]", " ", query)
    # 2. Supprimer les opérateurs booléens PubMed
    cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", cleaned, flags=re.IGNORECASE)
    # 3. Supprimer les guillemets et parenthèses
    cleaned = re.sub(r'["\(\)]', " ", cleaned)
    # 4. Normaliser les espaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = cleaned.split()
    candidates = []
    # Requête complète nettoyée
    if cleaned:
        candidates.append(cleaned)
    # Premier mot-clé seul (terme principal, souvent le nom de la maladie)
    if len(words) >= 2:
        candidates.append(words[0])
    # Deux premiers mots (ex : "rheumatoid arthritis")
    if len(words) >= 3:
        candidates.append(" ".join(words[:2]))

    # Dédupliquer en gardant l'ordre
    seen, result = set(), []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            result.append(c)
    return result


# ─── openFDA : fiches médicaments ────────────────────────────────────────────
def download_openfda_articles(query: str, limit: int = 20) -> list:
    """
    Interroge l'API openFDA (drug/label) avec la requête nettoyée.
    Si la requête principale retourne 0 résultats, essaie automatiquement
    des variantes plus courtes (fallback progressif).

    Retourne des documents au même format que PubMed : {"pmid", "title", "abstract"}.
    """
    url     = "https://api.fda.gov/drug/label.json"
    session = _make_session(retries=2, backoff=0.5)

    # Générer les variantes de requête du plus précis au plus large
    query_candidates = _clean_query_for_fda(query)
    print(f"[openFDA] Requêtes candidates : {query_candidates}")

    raw_results = []
    used_query  = query

    for candidate in query_candidates:
        params = {
            "search": f"indications_and_usage:{candidate}",
            "limit":  limit,
        }
        try:
            resp = session.get(url, params=params,
                               timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if resp.status_code == 404:
                print(f"[openFDA] 0 résultat pour '{candidate}', essai suivant…")
                continue
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("results", [])
            if raw_results:
                used_query = candidate
                print(f"[openFDA] {len(raw_results)} résultats avec '{candidate}'.")
                break
            else:
                print(f"[openFDA] 0 résultat pour '{candidate}', essai suivant…")
        except requests.exceptions.RequestException as e:
            print(f"[openFDA] Erreur réseau pour '{candidate}' : {e}")
            continue
        except Exception as e:
            print(f"[openFDA] Erreur inattendue pour '{candidate}' : {e}")
            continue

    if not raw_results:
        print(f"[openFDA] Aucun résultat après tous les essais.")
        return []

    articles = []
    for entry in raw_results:
        openfda = entry.get("openfda", {})

        # Nom du médicament
        brand   = openfda.get("brand_name",   [""])[0]
        generic = openfda.get("generic_name", [""])[0]
        title   = brand if brand else (generic if generic else "Unknown Drug")

        # Identifiant unique FDA
        app_nums = openfda.get("application_number", [])
        fda_id   = "fda-" + (app_nums[0] if app_nums else entry.get("set_id", "unknown"))

        # Abstract : concaténation des sections les plus informatives
        sections = []
        for field in [
            "indications_and_usage",
            "mechanism_of_action",
            "description",
            "dosage_and_administration",
            "warnings",
            "adverse_reactions",
        ]:
            value = entry.get(field)
            if value and isinstance(value, list) and value[0].strip():
                label = field.replace("_", " ").title()
                text  = value[0].strip()[:500]   # limité à 500 chars/section
                sections.append(f"[{label}] {text}")

        abstract = " ".join(sections).strip()
        if not abstract:
            continue  # fiche sans contenu textuel → ignorée

        name_label = f"{title} ({generic})" if generic and generic != title else title
        articles.append({
            "pmid":     fda_id,
            "title":    preprocess_text(f"[FDA Drug Label] {name_label}"),
            "abstract": preprocess_text(abstract),
        })

    print(f"[openFDA] {len(articles)} fiches médicaments récupérées (requête : '{used_query}').")
    return articles


# ─── Fonction principale : PubMed + openFDA en parallèle ─────────────────────
def download_articles(query: str, data_dir=None, retmax: int = 100) -> list:
    """
    Télécharge les documents depuis PubMed ET openFDA pour une requête donnée.
    - Les deux sources sont interrogées en parallèle (2 threads).
    - Si PubMed échoue (timeout / 0 résultats), openFDA seul est utilisé.
    - Les résultats sont fusionnés et dédupliqués par pmid.

    Retourne une liste de dicts {"pmid", "title", "abstract"}.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n[Sources] Recherche parallèle PubMed + openFDA pour : '{query}'…")

    # ── Worker PubMed ────────────────────────────────────────────────────────
    def _pubmed_worker():
        ids = search_pubmed(query, retmax=retmax)
        if not ids:
            print("[PubMed] Aucun article trouvé.")
            return []
        print(f"[PubMed] {len(ids)} IDs trouvés — téléchargement par batch de {BATCH_SIZE}…")
        arts = []
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i : i + BATCH_SIZE]
            print(f"  [PubMed] Batch {i//BATCH_SIZE + 1} : {len(batch)} articles…")
            arts.extend(fetch_articles_batch(batch))
            if i + BATCH_SIZE < len(ids):
                time.sleep(0.2)
        skipped = max(0, len(ids) - len(arts))
        print(f"[PubMed] {len(arts)} articles récupérés, {skipped} sans abstract ignorés.")
        return arts

    # ── Worker openFDA ───────────────────────────────────────────────────────
    def _openfda_worker():
        # Limiter à max 20 fiches pour ne pas saturer le contexte LLM
        limit = min(retmax // 3, 20)
        return download_openfda_articles(query, limit=limit)

    # ── Lancement en parallèle ───────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_pubmed  = executor.submit(_pubmed_worker)
        f_openfda = executor.submit(_openfda_worker)
        pubmed_articles  = f_pubmed.result()
        openfda_articles = f_openfda.result()

    # ── Fusion + déduplication par pmid ──────────────────────────────────────
    seen, merged = set(), []
    for article in pubmed_articles + openfda_articles:
        pid = article["pmid"]
        if pid not in seen:
            seen.add(pid)
            merged.append(article)

    n_pub = len(pubmed_articles)
    n_fda = len(openfda_articles)
    print(f"\n[Sources] PubMed : {n_pub} | openFDA : {n_fda} | Total : {len(merged)}")
    if not merged:
        print("[Sources] ⚠️  Aucun document trouvé depuis aucune source.")
    return merged


# ─── Mode autonome ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    query    = "prostatitis"
    articles = download_articles(query, DATA_DIR, retmax=100)
    if articles:
        print(f"\n{len(articles)} documents au total.")
        for art in articles[:3]:
            print(f"PMID/ID : {art['pmid']}")
            print(f"Title   : {art['title']}")
            print(f"Abstract: {art['abstract'][:200]}…")
            print("-" * 60)
