import asyncio
import os
import re
from collections import OrderedDict
from io import BytesIO
 
import nltk
import pandas as pd
from lxml import etree

from corpus import CorpusRegistry

MAX_CACHED_WORKS = 2 # max number of works to keep in memory at once

class WorkReader:
    def __init__(self, registry: CorpusRegistry):
        self._registry = registry
        self._cache: OrderedDict[str, dict] = OrderedDict() # used as an LRU Cache for works with keys as URNS and values a loaded data dicts for registry
        self._urn_locks: dict[str, asyncio.Lock] = {} # locks for each URN to prevent concurrent loading of the same work
        self._global_lock = asyncio.Lock() # global lock to protect access to the cache and locks dicts
    
    async def load(self, urn:str) -> dict:
        if urn in self._cache:
            self._cache.move_to_end(urn) # easy case: we already have the data loaded, just mark it as recently used and return it
            return self._cache[urn]
    
        async with self._global_lock:
            if urn not in self._urn_locks:
                self._urn_locks[urn] = asyncio.Lock() # create a lock for this URN if it doesn't exist yet

        async with self._urn_locks[urn]:
            # Double-check inside the lock: another coroutine may have finished
            # loading this URN while we were waiting
            if urn in self._cache:
                self._cache.move_to_end(urn)
                return self._cache[urn]
        
            # Load the work data from disk
            data = await self._load_from_disk(urn)
            self._evict_if_needed() # evict least recently used work if we're at capacity
            self._cache[urn] = data # add the newly loaded work to the cache
        
        return self._cache[urn]
    
    def extract_passage(self, data: dict, sentence_id: str) -> tuple:
        sentence_index = data["sentence_index"]
        glosses = data["glosses"]
        translations = data["translations"]

        glaux_elements = sentence_index.get(sentence_id, [])
        trans_sent_id = int(sentence_id) - 1  # off-by-one offset; this is a issue coming from data generation, will need to be fixed eventually for the Latin 
        trans_row = translations.get(trans_sent_id)

        speaker = glaux_elements[0].get("speaker", None) if glaux_elements else None

        if trans_row:
            translation_text = trans_row.get("translation", "")
            if not isinstance(translation_text, str) or not translation_text.strip():
                translation_text = "SENTENCE MISSING TRANSLATION"
            note = trans_row.get("note", "")
        else:
            translation_text = "SENTENCE MISSING TRANSLATION"
            note = ""
        
        translation_tokens = (nltk.word_tokenize(translation_text) if translation_text else [])

        words_to_ids: dict[str, list[int]] = {}
        for i, word in enumerate(translation_tokens):
            words_to_ids.setdefault(word, []).append(i)
        
        word_level_html = self._glaux_to_html(glaux_elements, words_to_ids, trans_sent_id, glosses)

        trans_dict = {
            "sent_id": trans_sent_id,
            "translation": translation_text,
            "note": note,
        }

        return word_level_html, trans_dict, translation_tokens, speaker
    
    async def _load_from_disk(self, urn: str) -> dict:
        work = self._registry.get_work(urn)
        if not work:
            raise ValueError(f"URN {urn} not found in reading list")
 
        needed_ids = self._get_needed_sentence_ids(work)
        data_dir = os.path.join("data", urn)
 
        # XML: stream only needed sentences
        xml_path = os.path.join(data_dir, f"{urn}.xml")
        sentence_index = self._iterparse_sentences(xml_path, needed_ids)
 
        # Glosses: plain dict for O(1) lookup by greek_id
        glosses_df = pd.read_csv(os.path.join(data_dir, "glosses.csv"))
        glosses = (
            glosses_df.dropna(subset=["greek_id", "gloss"])
            .drop_duplicates(subset=["greek_id"])
            .set_index("greek_id")["gloss"]
            .to_dict()
        )
        del glosses_df
 
        # Translations: plain dict keyed by sent_id for O(1) lookup
        trans_df = pd.read_csv(os.path.join(data_dir, "translations.csv"))
        translations = {
            int(row["sent_id"]): row.to_dict()
            for _, row in trans_df.iterrows()
            if pd.notna(row.get("sent_id"))
        }
        del trans_df
 
        return {
            "urn": urn,
            "sentence_index": sentence_index,
            "glosses": glosses,
            "translations": translations,
        }
 
    @staticmethod
    def _get_needed_sentence_ids(work: dict) -> set[str]:
        """
        Return the set of all sentence id strings needed for this work,
        derived from every section's glaux_sentences range. Non-contiguous
        ranges (e.g. Herodotus) are handled correctly since we union all
        ranges into a single set.
        """
        needed = set()
        for section in work.get("sections", []):
            start, end = section["glaux_sentences"]
            for sid in range(start, end + 1):
                needed.add(str(sid))
        return needed

    @staticmethod
    def _iterparse_sentences(filepath: str, needed_ids: set[str]) -> dict:
        """
        Stream through the XML file and collect only the <sentence> elements
        whose id is in needed_ids. Returns a dict mapping:
            sentence_id (str) -> list of word lxml Elements
 
        Memory profile: at any moment only one <sentence> subtree is live.
        The sentence_elem.clear() + getprevious() loop is the critical lxml
        pattern that actually frees memory during iteration — without both,
        iterparse still accumulates the full tree.
        """
        # Apply XML repairs to the raw string before streaming.
        # This is unavoidable given the malformed source files, but we do it
        # once and then stream rather than building a full tree.
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
 
        index = {}
        context = etree.iterparse(
            BytesIO(content.encode("utf-8")),
            events=("end",),
            tag="sentence",
        )
 
        for _, sentence_elem in context:
            sid = sentence_elem.get("id", "")
            if sid in needed_ids:
                index[sid] = list(sentence_elem)
 
            # Free this element and all preceding siblings from memory
            sentence_elem.clear()
            while sentence_elem.getprevious() is not None:
                del sentence_elem.getparent()[0]
 
        del context
        return index

    def _evict_if_needed(self) -> None:
        while len(self._cache) >= MAX_CACHED_WORKS:
            evicted_urn, _ = self._cache.popitem(last=False)
            # Clean up the lock for the evicted URN to avoid unbounded growth
            self._urn_locks.pop(evicted_urn, None)
    
    def _glaux_to_html(self, glaux_elements: list, eng_word_to_ids: dict, sentence_id: int, glosses: dict) -> str:
        html_parts = []
        id_to_word = {
            elem.get("id", ""): elem.get("form", "")
            for elem in glaux_elements
            if elem.get("form", "") != "E"
        }
        for elem in glaux_elements:
            word_text = elem.get("form", "") or ""
            if word_text == "E" or not word_text.strip():
                continue
            html_parts.append(
                self._render_span(elem, sentence_id, id_to_word, glosses)
            )
        return " ".join(html_parts)

    @staticmethod
    def _render_span(elem, sent_id: int, id_to_word: dict, glosses: dict) -> str:
        html_template = (
            '<span class="glossable-token" '
            'data-id="{word_id}" data-form="{form}" data-lemma="{lemma}" '
            'data-postag="{postag}" data-head="{head}" data-relation="{relation}" '
            'data-gloss="{gloss}" data-alignment="">{text}</span>'
        )
 
        word_id_str = elem.get("id", "")
        form = elem.get("form", "")
        lemma = elem.get("lemma", "")
        postag = elem.get("postag", "")
        head_id = elem.get("head", "")
        relation = elem.get("relation", "")
 
        try:
            word_id_int = int(word_id_str)
        except (ValueError, TypeError):
            word_id_int = None
 
        gloss = glosses.get(word_id_int, "") if word_id_int is not None else ""
        head = id_to_word.get(head_id, "Elliptical") if head_id != "0" else "Root"
 
        return html_template.format(
            word_id=word_id_str,
            form=form,
            lemma=lemma,
            postag=postag,
            head=head,
            relation=relation,
            gloss=gloss,
            text=form.strip(),
        )
