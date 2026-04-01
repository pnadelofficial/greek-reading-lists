import sqlite3
from corpus import CorpusRegistry

class SearchIndex:
    def __init__(self, db_path: str, registry: CorpusRegistry):
        self._db_path = db_path
        self._registry = registry
        self._verify()
 
    def _verify(self):
        try: 
            conn = self._connect()
            conn.execute("SELECT 1 FROM occurrences LIMIT 1")
            conn.close()
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Search index at '{self._db_path}' is missing or incomplete. Run build_index.py first.") from e
        
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    
    def search(self, q: str, mode: str, offset: int, page_size: int) -> dict:
        """
        Run a search query and return a result dict ready to pass to the
        search template. Handles form, lemma, and gloss modes.
        """
        conn = self._connect()
 
        with conn:
            rows, total = self._query_occurrences(conn, q, mode, offset, page_size)
            stats, tfidf_by_work = self._query_stats(conn, q, mode)
 
        conn.close()
 
        results = self._attach_read_urls(rows)
        works_found = (
            len({r["urn"] for r in results})
            if results
            else len(tfidf_by_work)
        )
 
        return {
            "results": results,
            "total_occurrences": total,
            "works_found": works_found,
            "stats": stats,
            "tfidf_by_work": tfidf_by_work,
        }
 
    def _query_occurrences(
        self,
        conn: sqlite3.Connection,
        q: str,
        mode: str,
        offset: int,
        page_size: int,
    ) -> tuple[list[dict], int]:
        """Return (rows, total_count) for the given query."""
 
        if mode == "form":
            where = "form = ?"
            params = (q,)
        elif mode == "lemma":
            where = "lemma = ?"
            params = (q,)
        else:  # gloss substring
            where = "gloss LIKE ?"
            params = (f"%{q}%",)
 
        total = conn.execute(
            f"SELECT COUNT(*) FROM occurrences WHERE {where}", params
        ).fetchone()[0]
 
        rows = conn.execute(
            f"""SELECT urn, author, title, sentence_id, word_id,
                       form, lemma, postag, gloss
                FROM occurrences
                WHERE {where}
                ORDER BY author, title, CAST(sentence_id AS INTEGER)
                LIMIT ? OFFSET ?""",
            (*params, page_size, offset),
        ).fetchall()
 
        return [dict(r) for r in rows], total

    def _query_stats(
        self,
        conn: sqlite3.Connection,
        q: str,
        mode: str,
    ) -> tuple[dict | None, list[dict]]:
        """
        Return (lemma_stats, tfidf_by_work) for form and lemma searches.
        Returns (None, []) for gloss searches since there is no single
        lemma to look up stats for.
        """
        if mode == "gloss":
            return None, []
 
        # For form searches, find the most frequent associated lemma
        if mode == "form":
            row = conn.execute(
                """SELECT lemma FROM occurrences
                   WHERE form = ?
                   GROUP BY lemma
                   ORDER BY COUNT(*) DESC
                   LIMIT 1""",
                (q,),
            ).fetchone()
            lookup_lemma = row["lemma"] if row else q
        else:
            lookup_lemma = q
 
        stats_row = conn.execute(
            "SELECT * FROM lemma_stats WHERE lemma = ?", (lookup_lemma,)
        ).fetchone()
        stats = dict(stats_row) if stats_row else None
 
        tfidf_rows = conn.execute(
            """SELECT o.author, o.title, o.urn,
                      COUNT(*) AS count,
                      t.tfidf_score
               FROM occurrences o
               LEFT JOIN tfidf_scores t
                      ON t.lemma = ? AND t.urn = o.urn
               WHERE o.lemma = ?
               GROUP BY o.urn
               ORDER BY t.tfidf_score DESC NULLS LAST""",
            (lookup_lemma, lookup_lemma),
        ).fetchall()
 
        return stats, [dict(r) for r in tfidf_rows]
 
    def _attach_read_urls(self, rows: list[dict]) -> list[dict]:
        """Add a read_url key to each result row."""
        for row in rows:
            row["read_url"] = self._registry.sentence_id_to_read_url(
                row["urn"], row["sentence_id"]
            )
        return rows