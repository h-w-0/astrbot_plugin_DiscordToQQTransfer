from __future__ import annotations

from functools import lru_cache
from pathlib import Path


class BundledSensitiveLexicon:
    def __init__(self, words: list[str]):
        self._root = {}
        self.word_count = 0
        for word in words:
            normalized_word = word.casefold()
            if not normalized_word:
                continue
            current_node = self._root
            for character in normalized_word:
                current_node = current_node.setdefault(character, {})
            if "" not in current_node:
                current_node[""] = word
                self.word_count += 1

    def find_match(self, message_text: str) -> str | None:
        normalized_text = str(message_text or "").casefold()
        for start_index in range(len(normalized_text)):
            current_node = self._root
            matched_word = None
            for end_index in range(start_index, len(normalized_text)):
                current_node = current_node.get(normalized_text[end_index])
                if current_node is None:
                    break
                terminal_word = current_node.get("")
                if terminal_word:
                    matched_word = terminal_word
            if matched_word:
                return matched_word
        return None


def _read_vocabulary_words(vocabulary_file: Path) -> list[str]:
    try:
        content = vocabulary_file.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = vocabulary_file.read_text(encoding="gb18030")
    return [line.strip() for line in content.splitlines() if line.strip()]


@lru_cache(maxsize=1)
def load_bundled_sensitive_lexicon() -> BundledSensitiveLexicon:
    vocabulary_directory = Path(__file__).resolve().parent / "resources" / "sensitive_lexicon" / "Vocabulary"
    if not vocabulary_directory.is_dir():
        raise FileNotFoundError(f"本地词汇库目录不存在: {vocabulary_directory}")

    seen_words = set()
    words = []
    for vocabulary_file in sorted(vocabulary_directory.glob("*.txt")):
        for word in _read_vocabulary_words(vocabulary_file):
            normalized_word = word.casefold()
            if normalized_word in seen_words:
                continue
            seen_words.add(normalized_word)
            words.append(word)
    return BundledSensitiveLexicon(words)
