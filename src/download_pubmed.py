import requests
import xml.etree.ElementTree as ET
import os
import time

# Répertoire de données (exporté pour import externe)
DATA_DIR = "./data"

def search_pubmed(query, retmax=500):
    url= "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data["esearchresult"]["idlist"]
    
    except requests.exceptions.RequestException as e:
        print(f"Error lors de la recherche dans pubmed: {e}")
        return []
    
def fetch_article(pmid):
    #recuperer le titre et abstract d'un article pubmed

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # extraire le titre
        #title contient le titre final
        title = ""

        #title_el contient le titre brut XML expl : <ArticleTitle>Effect of CBD on Prostatitis</ArticleTitle>
        title_el = root.find(".//ArticleTitle")
        if title_el is not None:
            """
            .itertext() permet de recuperer le texte ce qui est dans XML expl : ["Effect of ", "CBD", " on Prostatitis"]
            .join perment de concatener les element pour devient comme ca : "Effect of CBD on Prostatitis"
            .strip() permet de supprimer les espaces au debut et a la fin du titre
            """
            title = "".join(title_el.itertext()).strip()

        # extraire l'abstract
        abstract_parts = []
        for abstract_text in root.iter("AbstractText"):
            #recuperer les labels si existe
            label = abstract_text.get("label", "")
            #recuperer le texte de label
            text = "".join(abstract_text.itertext()).strip()
            if text:
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

        abstract = " ".join(abstract_parts)
        return title, abstract

    except requests.exceptions.RequestException as e:
        print(f"Erreur reseau pour pmid {pmid}: {e}")
        return "", ""
    except ET.ParseError as e:
        print(f"Errur XML pour pmid {pmid}: {e}")
        return "", ""
    except Exception as e:
        print(f"Erreur inattendue pour PMID {pmid}: {e}")
        return "", ""

#configuration
data_dir = "./data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def fetch_articles_batch(pmids):
    """
    Récupère le titre et l'abstract de PLUSIEURS articles PubMed en une seule requête.
    Beaucoup plus rapide que fetch_article() appelé en boucle (pas de sleep entre chaque).
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db":     "pubmed",
        "id":     ",".join(pmids),
        "retmode": "xml",
    }
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        articles = []
        for article_el in root.findall(".//PubmedArticle"):
            pmid_el  = article_el.find(".//PMID")
            pmid     = pmid_el.text if pmid_el is not None else "N/A"

            title_el = article_el.find(".//ArticleTitle")
            title    = "".join(title_el.itertext()).strip() if title_el is not None else ""

            abstract_parts = []
            for abstract_text in article_el.iter("AbstractText"):
                label = abstract_text.get("label", "")
                text  = "".join(abstract_text.itertext()).strip()
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

    except requests.exceptions.RequestException as e:
        print(f"Erreur réseau batch fetch: {e}")
        return []
    except ET.ParseError as e:
        print(f"Erreur XML batch fetch: {e}")
        return []
    except Exception as e:
        print(f"Erreur inattendue batch fetch: {e}")
        return []


def get_data_file(query, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    santized_query = query.replace(" ", "_").lower()[:80]  # limite longueur
    return os.path.join(data_dir, f"{santized_query}_data.txt")

def preprocess_text(text):
    """
    Nettoie et prétraite un texte donné.
    - Supprime les espaces inutiles.
    - Normalise les caractères spéciaux.
    """
    import re
    text = text.strip()  # Supprime les espaces en début et fin
    text = re.sub(r"\s+", " ", text)  # Remplace les espaces multiples par un seul
    text = re.sub(r"[^\w\s.,]", "", text)  # Supprime les caractères spéciaux (sauf .,)
    return text

def download_articles(query, data_dir=None, retmax=100):
    """
    Télécharge les articles PubMed pour une requête donnée.
    Retourne une liste de dictionnaires contenant les articles.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n[PubMed] Recherche pour : '{query}'...")
    ids = search_pubmed(query, retmax=retmax)

    if not ids:
        print("[PubMed] Aucun article trouvé pour cette requête.")
        return []

    print(f"[PubMed] {len(ids)} articles trouvés. Téléchargement en batch (plus rapide)...")

    # Téléchargement par lots de 100 (limite PubMed par requête)
    articles = []
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        print(f"  Batch {i//batch_size + 1} : {len(batch)} articles...")
        batch_articles = fetch_articles_batch(batch)
        articles.extend(batch_articles)
        # Respecter la limite de l'API PubMed seulement ENTRE les batches
        if i + batch_size < len(ids):
            time.sleep(0.34)

    skipped = retmax - len(articles) if retmax <= len(ids) else len(ids) - len(articles)
    print(f"[PubMed] Terminé : {len(articles)} articles récupérés, {max(0, skipped)} ignorés (sans abstract).")
    return articles

if __name__ == "__main__":
    # Mode autonome : téléchargement direct
    os.makedirs(DATA_DIR, exist_ok=True)
    query = "prostatitis"
    articles = download_articles(query, DATA_DIR, retmax=1000)

    if articles:
        print(f"{len(articles)} articles téléchargés.")
        # Exemple d'utilisation : afficher les 3 premiers articles
        for article in articles[:3]:
            print(f"PMID: {article['pmid']}")
            print(f"Title: {article['title']}")
            print(f"Abstract: {article['abstract'][:200]}...")
            print("-" * 50)
