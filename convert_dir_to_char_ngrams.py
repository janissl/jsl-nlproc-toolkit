#!/usr/bin/env python3

import os
import sys

from plaintext import PlainText


def main(src_dir, dest_dir, ngram_length):
    for entry in os.scandir(src_dir):
        if entry.is_file():
            with open(entry.path, encoding='utf-8') as in_file,\
                    open(os.path.join(dest_dir, entry.name), 'w', encoding='utf-8', newline='\n') as out_file:
                for line in in_file:
                    converted_string = PlainText.string_to_char_ngram_string(line.strip(), ngram_length)
                    out_file.write(converted_string + '\n')


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
