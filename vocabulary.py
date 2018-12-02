#!/usr/bin/env python3

from plaintext import PlainText


def build_unique(source_filepath, output_filepath):
    """Generate a file containing unique natural languages words in lowercase excluding numbers"""
    seen = set()

    with open(source_filepath, encoding='utf-8') as source, \
            open(output_filepath, 'w', encoding='utf-8', newline='\n') as output:
        for line in source:
            line = line.strip().lower()

            words = PlainText.get_natural_language_words(line)

            if words:
                for word in words:
                    word = word.lower()

                    if word not in seen:
                        output.write(word + '\n')
                        seen.add(word)
