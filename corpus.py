import os
import yaml

PAGE_SIZE = 10

class CorpusRegistry:
    def __init__(self, yaml_path: str = "reading_list.yaml"):
        self.yaml_path = yaml_path
        self._works = self._load()
    
    def _load(self) -> list[dict]:
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        works = []
        for work in config.get('works', []):
            urn = work.get('urn')
            data_dir = os.path.join("data", urn)
            has_data = (
                os.path.isfile(os.path.join(data_dir, f"{urn}.xml"))
                and os.path.isfile(os.path.join(data_dir, "glosses.csv"))
                and os.path.isfile(os.path.join(data_dir, "alignments.csv"))
                and os.path.isfile(os.path.join(data_dir, "translations.csv"))
            )

            sections = []
            for section in work.get('sections', []):
                textpart = section.get('textpart')
                slug = textpart.lower().replace(" ", "-")
                sections.append({
                    "textpart": textpart,
                    "slug": slug,
                    "glaux_sentences": section.get('glaux_sentences', []),
                    "url": f"/read/{urn}/{slug}" if has_data else None
                })

            works.append({
                "urn": urn,
                "author": work.get('author'),
                "title": work.get('title'),
                "has_data": has_data,
                "sections": sections
            })

        return works
    
    @property
    def works(self) -> list[dict]:
        return self._works

    def get_work(self, urn: str) -> dict | None:
        return next((w for w in self._works if w['urn'] == urn), None)
    
    def get_section(self, work: dict, slug: str) -> dict | None:
        return next((s for s in work['sections'] if s['slug'] == slug), None)
    
    def get_adjacent_section(self, work: dict, slug: str, delta: int) -> dict | None:
        slugs = [s["slug"] for s in work['sections']]
        try:
            idx = slugs.index(slug) + delta
            return work['sections'][idx] if 0 <= idx < len(work['sections']) else None
        except ValueError:
            return None
    
    def sentence_id_to_read_url(self, urn:str, sentence_id: str) -> str | None:
        work = self.get_work(urn)
        if not work or not work['has_data']:
            return None
        
        try:
            sid = int(sentence_id)
        except (TypeError, ValueError):
            return None
        
        for section in work['sections']:
            start, end = section["glaux_sentences"]
            if start <= sid <= end:
                offset = ((sid-start) // PAGE_SIZE) * PAGE_SIZE
                return f"/read/{urn}/{section['slug']}?offset={offset}"
        
        return None