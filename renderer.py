class Renderer:

    @staticmethod
    def render_sentence(word_level_html: str, trans_dict: dict, translation_tokens: list[str], speaker: str|None = None) -> str:
        sent_id = trans_dict["sent_id"]
        note = trans_dict.get("note", "") or ""
        
        translation_html = " ".join(
            f'<span class="translation-word" data-eng-id="{sent_id}-{i}">{word}</span>'
            for i, word in enumerate(translation_tokens)
        )
 
        if speaker:
            return (
                '<div class="sentence">'
                f'<div class="speaker"><b>Speaker: </b>{speaker}</div>'
                f'<div class="word-level"><b>Original: </b>{word_level_html}</div>'
                f'<div class="translation"><b>Translation: </b>{translation_html}</div>'
                f'<details class="note"><summary>Note</summary>{note}</details>'
                "<br/></div>"
            )
        else:
            return (
                '<div class="sentence">'
                f'<div class="word-level"><b>Original: </b>{word_level_html}</div>'
                f'<div class="translation"><b>Translation: </b>{translation_html}</div>'
                f'<details class="note"><summary>Note</summary>{note}</details>'
                "<br/></div>"
            )

    @staticmethod
    def render_passage(sentences: list[tuple]) -> str:
        """
        Render a list of (word_level_html, trans_dict, tokens, speaker) tuples
        into a single passage div containing all sentence divs.
        """
        return "<div class='sentences'>{}</div>".format(
            "".join(
                Renderer.render_sentence(*sentence) for sentence in sentences
            )
        )