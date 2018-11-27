#!/usr/bin/env python3
"""A set of methods for manipulation with plaintext."""

import os
import json
import re
from sklearn.feature_extraction.text import CountVectorizer


class PlainText:
    word_split_pattern_string = r"(\b([^\W_]|[\u2019'-])+)\b"
    nl_word_pattern_string = r"((?=[^\d_])[\w\u2019'-])+"
    space_pattern_string = r'[\s\u00b7\u2027\u3000\u30fb\uff65]'

    @staticmethod
    def get_natural_language_words(text):
        """Extract natural language words from an input plaintext (note: excludes numbers).

        May be useful for reducing noises in language detection etc.

        :param str text: a plaintext fragment (a sentence, a paragraph or a whole document text)
        :return: a list of natural language words
        :rtype: list[str]
        """
        # If the entire text does not contain any spaces but contain forward slashes,
        # this may represent a URL and is not counted as a natural language word
        if not re.search(PlainText.space_pattern_string, text) and '/' in text:
            return list()

        nl_words = [nl_word[0] for nl_word in re.findall(PlainText.word_split_pattern_string,
                                                         text)]

        for i in reversed(range(len(nl_words))):
            if len(nl_words[i]) < 2 or not re.fullmatch(PlainText.nl_word_pattern_string,
                                                        nl_words[i],
                                                        flags=re.IGNORECASE):
                nl_words.remove(nl_words[i])

        return nl_words

    @staticmethod
    def tokenize_word(word, ngram_size):
        """Split a word in character sequences (n-grams) of user-defined length.

        :param str word: a word to split in character sequences
        :param int ngram_size: a length of character sequences
        :return: a list of character sequences (n-grams)
        :rtype: list[str]
        """
        tokens = list()
        word = ' ' + word + ' '

        for i in range(0, len(word) - ngram_size + 1):
            if word[i:i+ngram_size].strip():
                tokens.append(word[i:i+ngram_size])

        return tokens

    @staticmethod
    def extract_char_ngrams(text, language):
        """Create a dictionary with character n-gram lengths as keys
        and character sequences of a particular length as their values.

        :param str text: a plaintext
        :param str language: an ISO 639-1 language code
        :return: a dictionary with n-grams and their frequency grouped by character sequence length (from 1 to 4)
        :rtype: dict[int, dict[str, int]]
        """
        ngram_dict = dict()

        for i in range(1, 5):
            ngram_dict[i] = dict()

        for line in text.split('\r?\n'):
            nl_words = PlainText.get_natural_language_words(line)

            for i in range(1, 5):
                for nlw in nl_words:
                    if not PlainText.is_valid_language_word(nlw, language):
                        continue

                    tokens = PlainText.tokenize_word(nlw, i)

                    for token in tokens:
                        try:
                            ngram_dict[i][token] += 1
                        except KeyError:
                            ngram_dict[i][token] = 1

        return ngram_dict

    @staticmethod
    def is_valid_language_word(word, language):
        """Classify whether a word is valid for the particular language.

        May be useful for building a clean language model.

        :param str word: any word
        :param str language: an ISO 639-1 language code
        :return: True if the input word only contains characters valid for the specified language, otherwise False
        :rtype: bool
        """
        valid_chars = PlainText.get_language_charset(language)
        invalid_chars = PlainText.get_language_exclude_charset(language)

        for char in word.strip():
            if (valid_chars and not re.search(valid_chars, char)) or (invalid_chars and re.search(invalid_chars, char)):
                return False

        return True

    @staticmethod
    def get_language_charset(language):
        """Return a set of characters (a regex string) valid for the particular language.

        For unsupported languages, an empty string is returned.

        :param str language: an ISO 639-1 language code
        :return: a regex string of valid characters
        :rtype: str
        """
        charset = dict()
        # TODO: define native character sets for more languages
        charset['lv'] = '[A-Za-zĀāČčĒēĢģĪīĶķĻļŅņŠšŪūŽž]'

        try:
            return charset[language]
        except KeyError:
            return ''

    @staticmethod
    def get_language_exclude_charset(language):
        """Return a set of characters (a regex string) valid for the script used by the particular language
        but not used in native words in this language.

        For unsupported languages, an empty string is returned.

        :param str language: an ISO 639-1 language code
        :return: a regex string of invalid characters
        :rtype: str
        """
        charset = dict()
        # TODO: define excludable character sets for more languages
        charset['lv'] = '[QqWwXxYy]'

        try:
            return charset[language]
        except KeyError:
            return ''

    @staticmethod
    def build_language_model_ch(input_filepath, language, output_dirpath, ngram_range=(1, 4)):
        """Build a language model consisting of character n-grams in the JSON format.

        Use cases: language models for language detection (e.g. https://github.com/shuyo/language-detection)

        :param str input_filepath: a plaintext filepath
        :param str language: an ISO 639-1 language code
        :param str output_dirpath: a path of the directory where to save the language model file
        :param tuple ngram_range: the minimal and the maximal size of character ngrams to generate
        """
        model_filepath = os.path.join(output_dirpath, language)

        with open(input_filepath, encoding='utf-8') as source:
            text = [line.strip() for line in source.readlines()]

        vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(text)

        ngram_labels = vectorizer.get_feature_names()
        ngram_frequency = matrix.toarray().astype(int).sum(axis=0)
        ngram_freqs = {label: int(freq) for label, freq in zip(ngram_labels, ngram_frequency) if label.strip()}

        model_data = {'freq': ngram_freqs}

        ngram_cnt_by_len = list()

        for i in range(ngram_range[0], ngram_range[1] + 1):
            counter = 0
            ngram_cnt_by_len.append(counter)

        for ngram, freq in ngram_freqs.items():
            ngram_cnt_by_len[len(ngram) - 1] += int(freq)

        model_data['n_words'] = ngram_cnt_by_len
        model_data['name'] = language

        with open(model_filepath, 'w', encoding='utf-8') as model_file:
            json.dump(model_data, model_file, ensure_ascii=False)


# def main():
#     in_path = r'lv.txt'
#     lang = 'lv'
#     dest_dir = r'C:\Users\janis_000\Documents'
#     ngram_range = (1, 3)
#
#     PlainText.build_language_model_ch(in_path, lang, dest_dir, ngram_range)
#
#
# if __name__ == '__main__':
#     import sys
#     sys.exit(main())
