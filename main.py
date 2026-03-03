from lxml import etree
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import nltk
import re
import os

nltk.download('punkt_tab')

app = FastAPI()
templates = Jinja2Templates(directory="templates")
glaux_tree = None
glosses_lookup = None
alignment_lookup = None
translation_lookup = None

AUTHOR_WORK_TO_PATH = [
    # missing bacchae
    {("Aeschylus", "Eumenides"): "data/0085-007/0085-007.xml"},
    {("Christian Scripture", "John 1-3"): "data/0031-004/0031-004.xml"},
    {("Herodotus", "Histories 1.1-13, 1.29-33, 1.46-56, 1.79-89, 1.107-140, 1.141, 1.152-3, 1.204-216"): "data/0016-001/0016-001.xml"},
    {("Hesiod", "Theogony"): "data/0020-001/0020-001.xml"},
    {("Hesiod", "Works and Days"): "data/0020-002/0020-002.xml"},
    {("Homer", "Iliad 22"): "data/0012-001/0012-001.xml"},
    {("Homer", "Odyssey 9"): "data/0012-002/0012-002.xml"},
    {("Lucian", "True History, Book 1"): "data/0062-0012/0062-0012.xml"},
    {("Lysias", "On the Murder of Eratosthenes"): "data/0540-001/0540-001.xml"},
    {("Plato", "Apology"): "data/0059-002/0059-002.xml"},
    {("Sappho", "1, 16, 31"): "data/0009-002/0009-002.xml"},
    {("Sophocles", "Ajax"): "data/0011-003/0011-003.xml"},
    {("Thucydides", "History of the Peloponnesian War"): "data/0003-001/0003-001.xml"}
]

def fix_malformed_xml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix angle brackets in form_original attributes (any position)
    content = re.sub(
        r'form_original="([^"]*?)<([^"]*?)"',
        lambda m: f'form_original="{m.group(1)}&lt;{m.group(2)}"',
        content
    )
    content = re.sub(
        r'form_original="([^"]*?)>([^"]*?)"',
        lambda m: f'form_original="{m.group(1)}&gt;{m.group(2)}"',
        content
    )
    
    # Fix unescaped quotes in form and lemma attributes
    content = re.sub(
        r'(form|lemma)="""',
        r'\1="&quot;"',
        content
    )
    
    # Fix malformed entities by escaping the ampersand
    content = re.sub(
        r'&(?!lt;|gt;|amp;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)',
        r'&amp;',
        content
    )
    
    return etree.fromstring(content.encode('utf-8'))

async def load_data(glaux_dir_path):
    global glaux_tree, glosses_lookup, alignment_lookup, translation_lookup
    _, glaux_id, _ = glaux_dir_path.split("/")
    glaux_tree = fix_malformed_xml(glaux_dir_path)
    glosses_lookup = pd.read_csv(os.path.join("data", glaux_id, "glosses.csv"))
    alignment_lookup = pd.read_csv(os.path.join("data", glaux_id, "alignments.csv"))
    translation_lookup = pd.read_csv(os.path.join("data", glaux_id, "translations.csv"))

def extract_passage(sentence_id=None):
    if sentence_id:
        xpath = f"//sentence[@id='{sentence_id}']/word"
        glaux_elements = glaux_tree.xpath(xpath)
        translation_row = translation_lookup[translation_lookup['sent_id'] == int(sentence_id)-1]
        speaker = glaux_elements[0].get("speaker", None) if glaux_elements else None
        
        translation_text = translation_row['translation'].values[0] if not translation_row.empty else ""
        translation_tokens = nltk.word_tokenize(translation_text) if translation_text else []

        word_to_ids = {}
        for i, word in enumerate(translation_tokens):
            if word not in word_to_ids:
                word_to_ids[word] = []
            word_to_ids[word].append(i)

        word_level_html = glaux_to_html(glaux_elements, word_to_ids, int(sentence_id)-1)        
        
        return word_level_html, translation_row, translation_tokens, speaker
    else:
        raise ValueError("Unsupported citation type")

def glaux_to_html(glaux_elements, eng_word_to_ids, sentence_id):
    html_parts = []
    id_to_word = {element.get("id", ""): element.get("form", "") for element in glaux_elements if element.get("form", "") != "E"}
    for elem in glaux_elements:
        word_text = elem.get("form", "") or ""
        if word_text == "E":
            continue
        if word_text.strip():
            span_html = render_span(elem, eng_word_to_ids, sentence_id, id_to_word=id_to_word)
            html_parts.append(span_html)
    return " ".join(html_parts)

def render_span(elem, eng_word_to_ids, sent_id, id_to_word=None):
    html_template = '<span class="glossable-token" data-id="{word_id}" data-form="{form}" data-lemma="{lemma}" data-postag="{postag}" data-head="{head}" data-relation="{relation}" data-gloss="{gloss}" data-alignment="{alignment}">{text}</span>'
    word_id = elem.get("id", "")
    form = elem.get("form", "")
    lemma = elem.get("lemma", "")
    postag = elem.get("postag", "")
    head_id = elem.get("head", "")
    relation = elem.get("relation", "")

    glosses = glosses_lookup[glosses_lookup['greek_id'] == int(word_id)]
    gloss = glosses['gloss'].values[0] if not glosses.empty else ""
    head = id_to_word.get(head_id, "Elliptical") if head_id != "0" else "Root"

    alignments = alignment_lookup[
        (alignment_lookup['greek_id'] == int(word_id)) & 
        (alignment_lookup['sent_id'] == sent_id)
    ]

    alignment_ids = []
    for _, row in alignments.iterrows():
        eng_word = row['english_word'] 
        if eng_word in eng_word_to_ids:
            alignment_ids.extend([f"{sent_id}-{id}" for id in eng_word_to_ids[eng_word]])
        else:
            phrase_tokens = eng_word.split()
            tokenized_list = list(eng_word_to_ids.keys())

            found = False
            for i in range(len(tokenized_list) - len(phrase_tokens) + 1):
                window = tokenized_list[i:i+len(phrase_tokens)]
                if ' '.join(window) == eng_word:
                    for token in window:
                        alignment_ids.extend([f"{sent_id}-{id}" for id in eng_word_to_ids[token]])
                    found = True
                    break   
            if not found:
                print(f"Warning: Could not find alignment for '{eng_word}' in sentence {sent_id}")

    alignment = ",".join(str(x) for x in alignment_ids)

    text = form if form.strip() else ""
    return html_template.format(word_id=word_id, form=form, lemma=lemma, postag=postag, head=head, relation=relation, gloss=gloss, alignment=alignment, text=text)

def format_sentence(word_level_html, translation_row, translation_tokens, speaker=None):
    sent_id = translation_row['sent_id'].values[0] if not translation_row.empty else 0
    translation_html = " ".join([
        f'<span class="translation-word" data-eng-id="{sent_id}-{i}">{word}</span>' 
        for i, word in enumerate(translation_tokens)
    ])

    if speaker: 
        html_template = """<div class="sentence">
        <div class="speaker"><b>Speaker: </b>{speaker}</div>
        <div class="word-level"><b>Original: </b>{word_level_html}</div>
        <div class="translation"><b>Translation: </b>{translation_html}</div>
        <div class="note"><b>Note: </b>{note}</div>
        <br/>
        </div>"""
        note = translation_row['notes'].values[0] if not translation_row.empty else ""
        return html_template.format(
            word_level_html=word_level_html, 
            translation_html=translation_html, 
            note=note,
            speaker=speaker
        )
    else:
        html_template = """<div class="sentence">
        <div class="word-level"><b>Original: </b>{word_level_html}</div>
        <div class="translation"><b>Translation: </b>{translation_html}</div>
        <div class="note"><b>Note: </b>{note}</div>
        <br/>
        </div>"""
        note = translation_row['notes'].values[0] if not translation_row.empty else ""
        return html_template.format(
            word_level_html=word_level_html, 
            translation_html=translation_html, 
            note=note
        )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Greek Reading List"})

@app.get("/browse", response_class=HTMLResponse)
async def browse(request: Request):
    authors = set()
    for mapping in AUTHOR_WORK_TO_PATH:
        for (a, w), path in mapping.items():
            authors.add(a)
    return templates.TemplateResponse("browse.html", {"request": request, "authors": sorted(authors)})

@app.get("/browse/{author}", response_class=HTMLResponse)
async def get_author_page(request: Request, author: str):
    print(author)
    works = []
    for mapping in AUTHOR_WORK_TO_PATH:
        for (a, w), path in mapping.items():
            if a.lower() == author.lower():
                works.append((w, path))
    if not works:
        print(f"No works found for author: {author}")
        raise HTTPException(status_code=404, detail="Author not found")
    return templates.TemplateResponse("author.html", {"request": request, "author": author, "works": works})

@app.get("/browse/{author}/{work}", response_class=HTMLResponse)
async def get_work_page(request: Request, author: str, work: str):
    path = None
    work = work.replace("-", " ")
    for mapping in AUTHOR_WORK_TO_PATH:
        for (a, w), p in mapping.items():
            if a.lower() == author.lower() and w.lower() == work.lower():
                path = p
                break
    print(path)
    await load_data(path)

    # check number of sentences in the work
    num_sentences = len(glaux_tree.xpath("//sentence"))
    aligned_passages = []
    for i in range(1, num_sentences+1):
        sentence_id = str(i)
        passage_tup = extract_passage(sentence_id)
        word_level_html, translation_row, translation_tokens, speaker = passage_tup
        aligned_passages.append((word_level_html, translation_row, translation_tokens, speaker))

    html_template = "<div class='sentences'>{aligned_passages}</div>"
    aligned_passages_html = html_template.format(aligned_passages="".join([format_sentence(word_level_html, translation_row, translation_tokens, speaker) for word_level_html, translation_row, translation_tokens, speaker in aligned_passages]))

    if not path:
        raise HTTPException(status_code=404, detail="Work not found")
    return templates.TemplateResponse("work.html", {"request": request, "author": author.capitalize(), "work": work.capitalize(), "path": path, "passage_html": aligned_passages_html})

@app.get("/browse/{author}/{work}/{sentence_id}", response_class=HTMLResponse)
async def get_sentence_page(request: Request, author: str, work: str, sentence_id: str):
    path = None
    work = work.replace("-", " ")
    for mapping in AUTHOR_WORK_TO_PATH:
        for (a, w), p in mapping.items():
            print(f"Checking author: {a} against {author}, work: {w} against {work}")
            if a.lower() == author.lower() and w.lower() == work.lower():
                path = p
                break

    print(path)
    await load_data(path)
    
    if not path:
        raise HTTPException(status_code=404, detail="Sentence not found")
    
    if "-" in sentence_id:
        start, end = sentence_id.split("-")
        aligned_passages = []
        for i in range(int(start), int(end)+1):
            passage_tup = extract_passage(str(i))
            word_level_html, translation_row, translation_tokens, speaker = passage_tup
            aligned_passages.append((word_level_html, translation_row, translation_tokens, speaker))
    else:
        passage_tup = extract_passage(sentence_id)
        word_level_html, translation_row, translation_tokens, speaker = passage_tup
        aligned_passages = [(word_level_html, translation_row, translation_tokens, speaker)]

    html_template = "<div class='sentences'>{aligned_passages}</div>"
    aligned_passages_html = html_template.format(aligned_passages="".join([format_sentence(word_level_html, translation_row, translation_tokens, speaker) for word_level_html, translation_row, translation_tokens, speaker in aligned_passages]))
    return templates.TemplateResponse("sentence.html", {"request": request, "author": author.capitalize(), "work": work.title(), "sentence_id": sentence_id, "passage_html": aligned_passages_html})