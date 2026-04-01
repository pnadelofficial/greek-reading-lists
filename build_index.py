import os
import re
import sqlite3
import yaml
import pandas as pd
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer

def fix_malformed_xml(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
 
    content = re.sub(
        r'form_original="([^"]*?)<([^"]*?)"',
        lambda m: f'form_original="{m.group(1)}&lt;{m.group(2)}"',
        content,
    )
    content = re.sub(
        r'form_original="([^"]*?)>([^"]*?)"',
        lambda m: f'form_original="{m.group(1)}&gt;{m.group(2)}"',
        content,
    )
    content = re.sub(r'(form|lemma)="""', r'\1="&quot;"', content)
    content = re.sub(
        r"&(?!lt;|gt;|amp;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)",
        r"&amp;",
        content,
    )
 
    return etree.fromstring(content.encode("utf-8"))

def load_reading_list(yaml_path="reading_list.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
 
    works = []
    for work in config.get("works", []):
        urn = work["urn"]
        data_dir = os.path.join("data", urn)
        has_data = (
            os.path.isfile(os.path.join(data_dir, f"{urn}.xml"))
            and os.path.isfile(os.path.join(data_dir, "glosses.csv"))
        )
        if has_data:
            works.append({
                "urn": urn,
                "author": work["author"],
                "title": work["title"],
            })
        else:
            print(f"  [skip] {urn} — data directory incomplete or missing")
 
    return works

def extract_tokens(urn, author, title):
    """
    Returns a list of dicts, one per non-empty word token:
      urn, author, title, sentence_id, word_id, form, lemma, postag, gloss
    """
    xml_path = os.path.join("data", urn, f"{urn}.xml")
    glosses_path = os.path.join("data", urn, "glosses.csv")
 
    tree = fix_malformed_xml(xml_path)
    glosses_df = pd.read_csv(glosses_path)
 
    # Build a fast gloss lookup: greek_id (int) -> gloss string
    # glosses.csv may have multiple glosses per word id; take the first
    gloss_lookup = (
        glosses_df.dropna(subset=["greek_id", "gloss"])
        .drop_duplicates(subset=["greek_id"])
        .set_index("greek_id")["gloss"]
        .to_dict()
    )
 
    tokens = []
    for sentence in tree.xpath("//sentence"):
        sent_id = sentence.get("id", "")
        for word in sentence.xpath("word"):
            form = word.get("form", "") or ""
            # Skip empty tokens and the special "E" ellipsis marker
            if not form.strip() or form == "E":
                continue
 
            word_id_str = word.get("id", "")
            try:
                word_id_int = int(word_id_str)
            except (ValueError, TypeError):
                word_id_int = None
 
            lemma = word.get("lemma", "") or ""
            postag = word.get("postag", "") or ""
            gloss = gloss_lookup.get(word_id_int, "") if word_id_int is not None else ""
 
            tokens.append({
                "urn": urn,
                "author": author,
                "title": title,
                "sentence_id": sent_id,
                "word_id": word_id_str,
                "form": form,
                "lemma": lemma,
                "postag": postag,
                "gloss": f" {str(gloss)} " if gloss else "",
            })
 
    return tokens

def compute_tfidf(all_tokens):
    """
    Returns a DataFrame with columns: lemma, urn, tfidf_score
    TF-IDF is computed at the work level (each work = one document).
    Uses sublinear TF scaling as discussed.
    """
    # Build a pseudo-document per URN: space-joined sequence of lemmas
    # (preserving repetition so TF is meaningful)
    work_docs = {}
    for t in all_tokens:
        urn = t["urn"]
        lemma = t["lemma"].strip()
        if not lemma:
            continue
        work_docs.setdefault(urn, []).append(lemma)
 
    urns = list(work_docs.keys())
    corpus = [" ".join(work_docs[u]) for u in urns]
 
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",   # treat any non-whitespace run as a token
        sublinear_tf=True,           # 1 + log(tf) instead of raw tf
        use_idf=True,
        smooth_idf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)  # shape: (n_works, n_lemmas)
    feature_names = vectorizer.get_feature_names_out()
 
    rows = []
    for doc_idx, urn in enumerate(urns):
        doc_vector = tfidf_matrix[doc_idx]
        # Only store non-zero scores to keep the table lean
        cx = doc_vector.tocoo()
        for lemma_idx, score in zip(cx.col, cx.data):
            rows.append({
                "lemma": feature_names[lemma_idx],
                "urn": urn,
                "tfidf_score": float(score),
            })
 
    return pd.DataFrame(rows, columns=["lemma", "urn", "tfidf_score"])

def compute_lemma_stats(all_tokens):
    """
    Returns a DataFrame with columns:
      lemma, corpus_count, doc_frequency, corpus_rank
    """
    df = pd.DataFrame(all_tokens)
    df = df[df["lemma"].str.strip() != ""]
 
    # Total occurrences of each lemma across the whole corpus
    corpus_counts = df.groupby("lemma").size().rename("corpus_count")
 
    # Number of distinct works the lemma appears in
    doc_freq = df.groupby("lemma")["urn"].nunique().rename("doc_frequency")
 
    stats = pd.concat([corpus_counts, doc_freq], axis=1).reset_index()
 
    # Rank by frequency (rank 1 = most common)
    stats["corpus_rank"] = stats["corpus_count"].rank(
        method="min", ascending=False
    ).astype(int)
 
    return stats

 
def build_db(db_path="search.db", yaml_path="reading_list.yaml"):
    print(f"Building {db_path} ...")
 
    works = load_reading_list(yaml_path)
    if not works:
        print("No works with data found. Exiting.")
        return
 
    # --- Step 1: extract tokens from every work ---
    all_tokens = []
    for work in works:
        urn = work["urn"]
        print(f"  Parsing {urn} ({work['author']}, {work['title']}) ...")
        tokens = extract_tokens(urn, work["author"], work["title"])
        print(f"    → {len(tokens):,} tokens")
        all_tokens.extend(tokens)
 
    print(f"\n  Total tokens across corpus: {len(all_tokens):,}")
 
    # --- Step 2: compute stats and TF-IDF ---
    print("\n  Computing lemma stats ...")
    lemma_stats_df = compute_lemma_stats(all_tokens)
    print(f"    → {len(lemma_stats_df):,} unique lemmas")
 
    print("  Computing TF-IDF scores ...")
    tfidf_df = compute_tfidf(all_tokens)
    print(f"    → {len(tfidf_df):,} (lemma, work) score pairs")
 
    # --- Step 3: write to SQLite ---
    print(f"\n  Writing to {db_path} ...")
 
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
 
    # Drop and recreate tables so reruns are idempotent
    cur.executescript("""
        DROP TABLE IF EXISTS occurrences;
        DROP TABLE IF EXISTS lemma_stats;
        DROP TABLE IF EXISTS tfidf_scores;
 
        CREATE TABLE occurrences (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            urn         TEXT NOT NULL,
            author      TEXT NOT NULL,
            title       TEXT NOT NULL,
            sentence_id TEXT NOT NULL,
            word_id     TEXT NOT NULL,
            form        TEXT NOT NULL,
            lemma       TEXT NOT NULL,
            postag      TEXT,
            gloss       TEXT
        );
 
        CREATE TABLE lemma_stats (
            lemma           TEXT PRIMARY KEY,
            corpus_count    INTEGER NOT NULL,
            doc_frequency   INTEGER NOT NULL,
            corpus_rank     INTEGER NOT NULL
        );
 
        CREATE TABLE tfidf_scores (
            lemma       TEXT NOT NULL,
            urn         TEXT NOT NULL,
            tfidf_score REAL NOT NULL,
            PRIMARY KEY (lemma, urn)
        );
    """)
 
    # occurrences
    occurrences_df = pd.DataFrame(all_tokens)
    occurrences_df.to_sql("occurrences", conn, if_exists="append", index=False)
 
    # lemma_stats
    lemma_stats_df.to_sql("lemma_stats", conn, if_exists="append", index=False)
 
    # tfidf_scores
    tfidf_df.to_sql("tfidf_scores", conn, if_exists="append", index=False)
 
    # --- Step 4: add indexes for fast query-time lookups ---
    print("  Creating indexes ...")
    cur.executescript("""
        CREATE INDEX IF NOT EXISTS idx_occ_form   ON occurrences(form);
        CREATE INDEX IF NOT EXISTS idx_occ_lemma  ON occurrences(lemma);
        CREATE INDEX IF NOT EXISTS idx_occ_gloss  ON occurrences(gloss);
        CREATE INDEX IF NOT EXISTS idx_occ_urn    ON occurrences(urn);
        CREATE INDEX IF NOT EXISTS idx_tfidf_lemma ON tfidf_scores(lemma);
    """)
 
    conn.commit()
    conn.close()
 
    print(f"\nDone. Database written to {db_path}")
    _print_summary(db_path)
 
 
def _print_summary(db_path):
    conn = sqlite3.connect(db_path)
    for table in ("occurrences", "lemma_stats", "tfidf_scores"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")
    conn.close()

def main():
    build_db()

if __name__ == "__main__":
    main()