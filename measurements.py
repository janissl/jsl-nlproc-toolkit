#!/usr/bin/env python3
"""Measure a quality of sentence alignment for parallel corpora"""

import sys


def load_sentence_pairs(src_filepath, trg_filepath):
    """Read a pair of parallel files and return a dict where keys are sentences in a source language
    and values are their translations in a target language

    :param str src_filepath: a path of the source language plaintext file
    :param str trg_filepath: a path of the target language plaintext file
    :return: sentence pairs
    :rtype: dict
    """
    sent_pairs = dict()

    with open(src_filepath, encoding='utf-8') as src_file, \
            open(trg_filepath, encoding='utf-8') as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            sent_pairs[src_line.strip()] = trg_line.strip()

    return sent_pairs


def calculate_precision(benchmark_pairs, aligned_pairs):
    """Return the fraction of relevant instances among the retrieved instances

    :param dict benchmark_pairs: sentence pairs from a benchmark set
    :param dict aligned_pairs: aligned sentence pairs
    :return: a precision value
    :rtype: float
    """
    aligned_from_benchmark = 0
    aligned_correctly = 0

    for src, trg in benchmark_pairs.items():
        if src in aligned_pairs.keys():
            aligned_from_benchmark += 1

            if trg == aligned_pairs[src]:
                aligned_correctly += 1
        elif trg in aligned_pairs.values():
            aligned_from_benchmark += 1

    return aligned_correctly / aligned_from_benchmark


def calculate_recall(benchmark_pairs, aligned_pairs):
    """Return the fraction of relevant instances that have been retrieved over the total amount of relevant instances

    :param dict benchmark_pairs: sentence pairs from a benchmark set
    :param dict aligned_pairs: aligned sentence pairs
    :return: a recall value
    :rtype: float
    """
    aligned_correctly = 0

    for src, trg in benchmark_pairs.items():
        if src in aligned_pairs.keys() and trg == aligned_pairs[src]:
            aligned_correctly += 1

    return aligned_correctly / len(benchmark_pairs)


def calculate_f1(precision, recall):
    """Return the harmonic average of the precision and recall

    :param float precision: a precision value
    :param float recall: a recall value
    :return: a F1-score value
    :rtype: float
    """
    if precision == 0 or recall == 0:
        return 0

    return (2 * precision * recall) / (precision + recall)


def do_all(src_lang_benchmark_file, trg_lang_benchmark_file, src_lang_file, trg_lang_file):
    """Calculate and print precision, recall and F1-score values for aligned corpora

    :param src_lang_benchmark_file: a path of the source language benchmark file
    :param trg_lang_benchmark_file: a path of the target language benchmark file
    :param src_lang_file: a path of the source language plaintext file
    :param trg_lang_file: a path of the target language plaintext file
    """
    benchmark_pairs = load_sentence_pairs(src_lang_benchmark_file, trg_lang_benchmark_file)
    aligned_pairs = load_sentence_pairs(src_lang_file, trg_lang_file)

    precision = calculate_precision(benchmark_pairs, aligned_pairs)
    print('Precision: {:.2f}'.format(precision))

    recall = calculate_recall(benchmark_pairs, aligned_pairs)
    print('Recall: {:.2f}'.format(recall))

    f1 = calculate_f1(precision, recall)
    print('F1: {:.2f}'.format(f1))


if __name__ == '__main__':
    sys.exit(do_all(*sys.argv[1:]))
