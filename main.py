import nltk
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
 
from corpus import CorpusRegistry, PAGE_SIZE
from reader import WorkReader
from renderer import Renderer
from search import SearchIndex

nltk.download("punkt_tab")
 
app = FastAPI()
templates = Jinja2Templates(directory="templates")
 
SEARCH_PAGE_SIZE = 50
SEARCH_DB = "search.db"

registry = CorpusRegistry("reading_list.yaml")
reader = WorkReader(registry)
renderer = Renderer()
search_index = SearchIndex(SEARCH_DB, registry)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "works": registry.works, "title": "Greek Reading List"},
    )

@app.get("/read/{urn}/{slug}", response_class=HTMLResponse)
async def get_section_page(
    request: Request, urn: str, slug: str, offset: int = 0
):
    work_obj = registry.get_work(urn)
    if not work_obj:
        raise HTTPException(status_code=404, detail="Work not found")
 
    section = registry.get_section(work_obj, slug)
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")
 
    # load() returns the cached data dict; concurrent requests for different
    # URNs proceed in parallel, same-URN requests queue behind a per-URN lock
    data = await reader.load(urn)
 
    start, end = section["glaux_sentences"]
    section_length = end - start + 1
 
    offset = max(0, min(offset, section_length - 1))
    page_start = start + offset
    page_end = min(start + offset + PAGE_SIZE - 1, end)
    is_last_page = page_end >= end
 
    sentences = [
        reader.extract_passage(data, str(i))
        for i in range(page_start, page_end + 1)
    ]
    passage_html = renderer.render_passage(sentences)
 
    # Build prev URL
    if offset == 0:
        prev_section = registry.get_adjacent_section(work_obj, slug, -1)
        if prev_section:
            prev_start, prev_end = prev_section["glaux_sentences"]
            prev_length = prev_end - prev_start + 1
            prev_last_offset = ((prev_length - 1) // PAGE_SIZE) * PAGE_SIZE
            prev_url = f"/read/{urn}/{prev_section['slug']}?offset={prev_last_offset}"
        else:
            prev_url = None
    else:
        prev_url = f"/read/{urn}/{slug}?offset={offset - PAGE_SIZE}"
 
    # Build next URL
    if is_last_page:
        next_section = registry.get_adjacent_section(work_obj, slug, 1)
        next_url = (
            f"/read/{urn}/{next_section['slug']}?offset=0" if next_section else None
        )
    else:
        next_url = f"/read/{urn}/{slug}?offset={offset + PAGE_SIZE}"
 
    return templates.TemplateResponse(
        "section.html",
        {
            "request": request,
            "work": work_obj,
            "section": section,
            "passage_html": passage_html,
            "prev_url": prev_url,
            "next_url": next_url,
            "offset": offset,
            "page_start": page_start,
            "page_end": page_end,
            "section_start": start,
            "section_end": end,
        },
    )

@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str | None = None,
    mode: str = "lemma",
    offset: int = 0,
):
    if mode not in ("form", "lemma", "gloss"):
        mode = "lemma"
 
    if not q or not q.strip():
        return templates.TemplateResponse(
            "search.html",
            {
                "request": request,
                "query": None,
                "mode": mode,
                "results": [],
                "stats": None,
                "tfidf_by_work": [],
                "total_occurrences": 0,
                "works_found": 0,
                "offset": 0,
                "page_size": SEARCH_PAGE_SIZE,
            },
        )
 
    q = q.strip()
    result = search_index.search(q, mode, offset, SEARCH_PAGE_SIZE)
 
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": q,
            "mode": mode,
            "offset": offset,
            "page_size": SEARCH_PAGE_SIZE,
            **result,
        },
    )

# from lxml import etree
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import yaml
# import pandas as pd
# import nltk
# import re
# import os
# import sqlite3

# nltk.download('punkt_tab')

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# PAGE_SIZE = 10
# SEARCH_DB = "search.db"
# SEARCH_PAGE_SIZE = 50

# glaux_tree = None
# glosses_lookup = None
# # alignment_lookup = None
# translation_lookup = None
# current_urn = None

# def load_reading_list(yaml_path = "reading_list.yaml"):
#     with open(yaml_path, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)

#     works = []
#     for work in config.get("works", []):
#         urn = work["urn"]
#         data_dir = os.path.join("data", urn)
#         has_data = (
#             os.path.isfile(os.path.join(data_dir, f"{urn}.xml"))
#             and os.path.isfile(os.path.join(data_dir, "glosses.csv"))
#             and os.path.isfile(os.path.join(data_dir, "alignments.csv"))
#             and os.path.isfile(os.path.join(data_dir, "translations.csv"))
#         )
#         sections = []
#         for section in work.get("sections", []):
#             textpart = section["textpart"]
#             slug = textpart.lower().replace(" ", "-")
#             sections.append({
#                 "textpart": textpart,
#                 "slug": slug,
#                 "glaux_sentences": section["glaux_sentences"],
#                 "url": f"/read/{urn}/{slug}" if has_data else None,
#             })

#         works.append({
#             "urn": urn,
#             "author": work["author"],
#             "title": work["title"],
#             "has_data": has_data,
#             "sections": sections,
#         })

#     return works
# WORKS = load_reading_list()

# def fix_malformed_xml(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     # Fix angle brackets in form_original attributes (any position)
#     content = re.sub(
#         r'form_original="([^"]*?)<([^"]*?)"',
#         lambda m: f'form_original="{m.group(1)}&lt;{m.group(2)}"',
#         content
#     )
#     content = re.sub(
#         r'form_original="([^"]*?)>([^"]*?)"',
#         lambda m: f'form_original="{m.group(1)}&gt;{m.group(2)}"',
#         content
#     )
    
#     # Fix unescaped quotes in form and lemma attributes
#     content = re.sub(
#         r'(form|lemma)="""',
#         r'\1="&quot;"',
#         content
#     )
    
#     # Fix malformed entities by escaping the ampersand
#     content = re.sub(
#         r'&(?!lt;|gt;|amp;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)',
#         r'&amp;',
#         content
#     )
    
#     return etree.fromstring(content.encode('utf-8'))


# async def load_data(urn: str):
#     global glaux_tree, glosses_lookup, alignment_lookup, translation_lookup, current_urn
#     if current_urn == urn:
#         return  # already loaded, skip
#     filepath = f"data/{urn}/{urn}.xml"
#     glaux_tree = fix_malformed_xml(filepath)
#     glosses_lookup = pd.read_csv(os.path.join("data", urn, "glosses.csv")).set_index('greek_id')
#     # alignment_lookup = pd.read_csv(os.path.join("data", urn, "alignments.csv"))[['greek_id', 'english_word', 'sent_id', 'greek_word']]
#     # alignment_lookup["alignments"] = ""
#     # alignment_lookup = alignment_lookup.dropna(subset=['greek_id', 'sent_id'])
#     # alignment_lookup['greek_id'] = alignment_lookup['greek_id'].astype(int)
#     translation_lookup = pd.read_csv(os.path.join("data", urn, "translations.csv"))
#     current_urn = urn

# def get_work(urn: str):
#     for w in WORKS:
#         if w["urn"] == urn:
#             return w
#     return None

# def get_section(work: dict, slug: str):
#     for s in work["sections"]:
#         if s["slug"] == slug:
#             return s
#     return None

# def get_adjacent_section(work: dict, slug: str, delta: int):
#     slugs = [s["slug"] for s in work["sections"]]
#     try:
#         idx = slugs.index(slug) + delta
#         return work["sections"][idx] if 0 <= idx < len(work["sections"]) else None
#     except ValueError:
#         return None
    
# def extract_passage(sentence_id=None):
#     if sentence_id:
#         xpath = f"//sentence[@id='{sentence_id}']/word"
#         glaux_elements = glaux_tree.xpath(xpath)
#         translation_row = translation_lookup[translation_lookup['sent_id'] == int(sentence_id)-1]
#         speaker = glaux_elements[0].get("speaker", None) if glaux_elements else None
        
#         translation_text = translation_row['translation'].values[0] if not translation_row.empty else ""
#         if isinstance(translation_text, float) and pd.isna(translation_text):
#             translation_text = "SENTENCE MISSING TRANSLATION"
#         translation_tokens = nltk.word_tokenize(translation_text) if translation_text else []

#         word_to_ids = {}
#         for i, word in enumerate(translation_tokens):
#             if word not in word_to_ids:
#                 word_to_ids[word] = []
#             word_to_ids[word].append(i)

#         word_level_html = glaux_to_html(glaux_elements, word_to_ids, int(sentence_id)-1)        
        
#         return word_level_html, translation_row, translation_tokens, speaker
#     else:
#         raise ValueError("Unsupported citation type")

# def glaux_to_html(glaux_elements, eng_word_to_ids, sentence_id):
#     html_parts = []
#     id_to_word = {element.get("id", ""): element.get("form", "") for element in glaux_elements if element.get("form", "") != "E"}
#     for elem in glaux_elements:
#         word_text = elem.get("form", "") or ""
#         if word_text == "E":
#             continue
#         if word_text.strip():
#             span_html = render_span(elem, eng_word_to_ids, sentence_id, id_to_word=id_to_word)
#             html_parts.append(span_html)
#     return " ".join(html_parts)

# def render_span(elem, eng_word_to_ids, sent_id, id_to_word=None):
#     html_template = '<span class="glossable-token" data-id="{word_id}" data-form="{form}" data-lemma="{lemma}" data-postag="{postag}" data-head="{head}" data-relation="{relation}" data-gloss="{gloss}" data-alignment="{alignment}">{text}</span>'
#     word_id = elem.get("id", "")
#     form = elem.get("form", "")
#     lemma = elem.get("lemma", "")
#     postag = elem.get("postag", "")
#     head_id = elem.get("head", "")
#     relation = elem.get("relation", "")

#     # glosses = glosses_lookup[glosses_lookup['greek_id'] == int(word_id)]
#     gloss = glosses_lookup.loc[int(word_id)] if int(word_id) in glosses_lookup.index else ""
#     head = id_to_word.get(head_id, "Elliptical") if head_id != "0" else "Root"

#     # print(f"Sentence ID: {sent_id}")
#     # alignments = alignment_lookup[
#     #     (alignment_lookup['greek_id'] == word_id) & 
#     #     (alignment_lookup['sent_id'] == sent_id+1)
#     # ]
#     # print(f"Found {len(alignments)} alignments for Greek word ID {word_id} in sentence {sent_id}")

#     # alignment_ids = []
#     # for _, row in alignments.iterrows():
#     #     eng_word = row['english_word'] 
#     #     print(f"Processing alignment for Greek word ID {word_id} (sentence {sent_id}): English word '{eng_word}'")
#     #     if isinstance(eng_word, float) and pd.isna(eng_word):
#     #         continue
#     #     if eng_word in eng_word_to_ids:
#     #         alignment_ids.extend([f"{sent_id}-{id}" for id in eng_word_to_ids[eng_word]])
#     #     else:
#     #         phrase_tokens = eng_word.split()
#     #         tokenized_list = list(eng_word_to_ids.keys())

#     #         found = False
#     #         for i in range(len(tokenized_list) - len(phrase_tokens) + 1):
#     #             window = tokenized_list[i:i+len(phrase_tokens)]
#     #             if ' '.join(window) == eng_word:
#     #                 for token in window:
#     #                     alignment_ids.extend([f"{sent_id}-{id}" for id in eng_word_to_ids[token]])
#     #                 found = True
#     #                 break   
#     #         if not found:
#     #             print(f"Warning: Could not find alignment for '{eng_word}' in sentence {sent_id}")

#     # alignment = ",".join(str(x) for x in alignment_ids)

#     text = form if form.strip() else ""
#     return html_template.format(
#         word_id=word_id, 
#         form=form, 
#         lemma=lemma,
#         postag=postag, 
#         head=head, 
#         relation=relation, 
#         gloss=gloss, 
#         alignment="", #alignment, turning this off for now
#         text=text
#     )

# def format_sentence(word_level_html, translation_row, translation_tokens, speaker=None):
#     sent_id = translation_row['sent_id'].values[0] if not translation_row.empty else 0
#     translation_html = " ".join([
#         f'<span class="translation-word" data-eng-id="{sent_id}-{i}">{word}</span>' 
#         for i, word in enumerate(translation_tokens)
#     ])

#     if speaker: 
#         html_template = """<div class="sentence">
#         <div class="speaker"><b>Speaker: </b>{speaker}</div>
#         <div class="word-level"><b>Original: </b>{word_level_html}</div>
#         <div class="translation"><b>Translation: </b>{translation_html}</div>
#         <details class="note"><summary>Note</summary>{note}</details>
#         <br/>
#         </div>"""
#         note = translation_row['notes'].values[0] if not translation_row.empty else ""
#         return html_template.format(
#             word_level_html=word_level_html, 
#             translation_html=translation_html, 
#             note=note,
#             speaker=speaker
#         )
#     else:
#         html_template = """<div class="sentence">
#         <div class="word-level"><b>Original: </b>{word_level_html}</div>
#         <div class="translation"><b>Translation: </b>{translation_html}</div>
#         <details class="note"><summary>Note</summary>{note}</details>
#         <br/>
#         </div>"""
#         note = translation_row['notes'].values[0] if not translation_row.empty else ""
#         return html_template.format(
#             word_level_html=word_level_html, 
#             translation_html=translation_html, 
#             note=note
#         )

# def get_db():
#     conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
#     conn.row_factory = sqlite3.Row
#     return conn

# def sentence_id_to_read_url(urn: str, sentence_id: str, works: list) -> str | None:
#     work = next((w for w in works if w["urn"] == urn), None)
#     if not work or not work.get("has_data"):
#         return None
 
#     try:
#         sid = int(sentence_id)
#     except (TypeError, ValueError):
#         return None
 
#     for section in work.get("sections", []):
#         start, end = section["glaux_sentences"]
#         if start <= sid <= end:
#             # offset is the start of the PAGE_SIZE-block containing this sentence
#             offset = ((sid - start) // PAGE_SIZE) * PAGE_SIZE
#             return f"/read/{urn}/{section['slug']}?offset={offset}"
#     return None
 

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     works = load_reading_list()
#     return templates.TemplateResponse(
#         "index.html", {"request": request, "works": works, "title": "Greek Reading List"}
#     )

# @app.get("/read/{urn}/{slug}", response_class=HTMLResponse)
# async def get_section_page(request: Request, urn: str, slug: str, offset: int = 0):
#     work_obj = get_work(urn)
#     if not work_obj:
#         raise HTTPException(status_code=404, detail="Work not found")

#     section = get_section(work_obj, slug)
#     if not section:
#         raise HTTPException(status_code=404, detail="Section not found")

#     await load_data(urn)

#     start, end = section["glaux_sentences"]
#     section_length = end - start + 1

#     # Clamp offset to valid range
#     offset = max(0, min(offset, section_length - 1))
#     page_start = start + offset
#     page_end = min(start + offset + PAGE_SIZE - 1, end)
#     is_last_page = page_end >= end

#     aligned_passages = []
#     for i in range(page_start, page_end + 1):
        
#         word_level_html, translation_row, translation_tokens, speaker = extract_passage(str(i))

#         aligned_passages.append((word_level_html, translation_row, translation_tokens, speaker))

#     passage_html = "<div class='sentences'>{}</div>".format(
#         "".join(format_sentence(*p) for p in aligned_passages)
#     )

#     # Build prev URL
#     if offset == 0:
#         prev_section = get_adjacent_section(work_obj, slug, -1)
#         if prev_section:
#             prev_start, prev_end = prev_section["glaux_sentences"]
#             prev_length = prev_end - prev_start + 1
#             prev_last_offset = ((prev_length - 1) // PAGE_SIZE) * PAGE_SIZE
#             prev_url = f"/read/{urn}/{prev_section['slug']}?offset={prev_last_offset}"
#         else:
#             prev_url = None
#     else:
#         prev_url = f"/read/{urn}/{slug}?offset={offset - PAGE_SIZE}"

#     # Build next URL
#     if is_last_page:
#         next_section = get_adjacent_section(work_obj, slug, 1)
#         next_url = f"/read/{urn}/{next_section['slug']}?offset=0" if next_section else None
#     else:
#         next_url = f"/read/{urn}/{slug}?offset={offset + PAGE_SIZE}"

#     return templates.TemplateResponse("section.html", {
#         "request": request,
#         "work": work_obj,
#         "section": section,
#         "passage_html": passage_html,
#         "prev_url": prev_url,
#         "next_url": next_url,
#         "offset": offset,
#         "page_start": page_start,
#         "page_end": page_end,
#         "section_start": start,
#         "section_end": end,
#     })

# @app.get("/search", response_class=HTMLResponse)
# async def search(request: Request, q: str | None = None, mode: str = "lemma", offset: int = 0,):

#     if mode not in ("form", "lemma", "gloss"):
#         mode = "lemma"
 
#     # No query yet – render the empty search page
#     if not q or not q.strip():
#         return templates.TemplateResponse("search.html", {
#             "request": request,
#             "query": None,
#             "mode": mode,
#             "results": [],
#             "stats": None,
#             "tfidf_by_work": [],
#             "total_occurrences": 0,
#             "works_found": 0,
#             "offset": 0,
#             "page_size": SEARCH_PAGE_SIZE,
#         })
 
#     q = q.strip()
 
#     try:
#         conn = get_db()
#     except sqlite3.OperationalError:
#         # DB hasn't been built yet
#         raise HTTPException(
#             status_code=503,
#             detail="Search index not available. Run build_index.py first."
#         )
    
#     with conn:
#         if mode == "form":
#             count_row = conn.execute(
#                 "SELECT COUNT(*) FROM occurrences WHERE form = ?", (q,)
#             ).fetchone()
#             total = count_row[0]
 
#             rows = conn.execute(
#                 """SELECT urn, author, title, sentence_id, word_id,
#                           form, lemma, postag, gloss
#                    FROM occurrences
#                    WHERE form = ?
#                    ORDER BY author, title, CAST(sentence_id AS INTEGER)
#                    LIMIT ? OFFSET ?""",
#                 (q, SEARCH_PAGE_SIZE, offset),
#             ).fetchall()
 
#         elif mode == "lemma":
#             count_row = conn.execute(
#                 "SELECT COUNT(*) FROM occurrences WHERE lemma = ?", (q,)
#             ).fetchone()
#             total = count_row[0]
 
#             rows = conn.execute(
#                 """SELECT urn, author, title, sentence_id, word_id,
#                           form, lemma, postag, gloss
#                    FROM occurrences
#                    WHERE lemma = ?
#                    ORDER BY author, title, CAST(sentence_id AS INTEGER)
#                    LIMIT ? OFFSET ?""",
#                 (q, SEARCH_PAGE_SIZE, offset),
#             ).fetchall()
 
#         else:  # gloss substring
#             pattern = f"%{q}%"
#             count_row = conn.execute(
#                 "SELECT COUNT(*) FROM occurrences WHERE gloss LIKE ?", (pattern,)
#             ).fetchone()
#             total = count_row[0]
 
#             rows = conn.execute(
#                 """SELECT urn, author, title, sentence_id, word_id,
#                           form, lemma, postag, gloss
#                    FROM occurrences
#                    WHERE gloss LIKE ?
#                    ORDER BY author, title, CAST(sentence_id AS INTEGER)
#                    LIMIT ? OFFSET ?""",
#                 (pattern, SEARCH_PAGE_SIZE, offset),
#             ).fetchall()
        
#         stats = None
#         tfidf_by_work = []
 
#         if mode != "gloss":
#             # For form searches we look up stats by lemma of the matched form.
#             # We grab the most common lemma associated with this form if there
#             # are multiple (edge-case for homographs).
#             if mode == "form":
#                 lemma_row = conn.execute(
#                     """SELECT lemma FROM occurrences
#                        WHERE form = ?
#                        GROUP BY lemma
#                        ORDER BY COUNT(*) DESC
#                        LIMIT 1""",
#                     (q,),
#                 ).fetchone()
#                 lookup_lemma = lemma_row["lemma"] if lemma_row else q
#             else:
#                 lookup_lemma = q
 
#             stats_row = conn.execute(
#                 "SELECT * FROM lemma_stats WHERE lemma = ?", (lookup_lemma,)
#             ).fetchone()
#             if stats_row:
#                 stats = dict(stats_row)
 
#             # Per-work occurrence counts joined with TF-IDF scores
#             tfidf_rows = conn.execute(
#                 """SELECT o.author, o.title, o.urn,
#                           COUNT(*) AS count,
#                           t.tfidf_score
#                    FROM occurrences o
#                    LEFT JOIN tfidf_scores t
#                           ON t.lemma = ? AND t.urn = o.urn
#                    WHERE o.lemma = ?
#                    GROUP BY o.urn
#                    ORDER BY t.tfidf_score DESC NULLS LAST""",
#                 (lookup_lemma, lookup_lemma),
#             ).fetchall()
#             tfidf_by_work = [dict(r) for r in tfidf_rows]
 
#     conn.close()
    
#     results = []
#     works = WORKS  # module-level list already loaded at startup
#     for row in rows:
#         r = dict(row)
#         r["read_url"] = sentence_id_to_read_url(r["urn"], r["sentence_id"], works)
#         results.append(r)
 
#     works_found = len({r["urn"] for r in results}) if results else (
#         len(tfidf_by_work) if tfidf_by_work else 0
#     )
 
#     return templates.TemplateResponse("search.html", {
#         "request": request,
#         "query": q,
#         "mode": mode,
#         "results": results,
#         "stats": stats,
#         "tfidf_by_work": tfidf_by_work,
#         "total_occurrences": total,
#         "works_found": works_found,
#         "offset": offset,
#         "page_size": SEARCH_PAGE_SIZE,
#     })
