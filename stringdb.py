#!/usr/bin/env python3

# Support for STRING DB / JensenLab tagger input and output formats

from collections import namedtuple
from itertools import zip_longest

# Modified from https://github.com/spyysalo/string-db-tools
StringDocument = namedtuple(
    'StringDocument',
    'doc_id, other_ids, authors, forum, year, text'
)

# Modified from https://github.com/spyysalo/string-db-tools
StringDoc = namedtuple(
    'StringDoc',
    'doc_id, other_ids, authors, forum, year, text, tids, sids, data'
)

# Modified from https://github.com/spyysalo/string-db-tools
StringSpan = namedtuple(
    'StringSpan',
    'doc_id, par_num, sent_num, start, end, text, type_id, serial'
)

# Modified from https://github.com/spyysalo/string-db-tools
def stringdb_unescape_text(text):
    """Unescape text field in database_documents.tsv format."""
    unescaped = []
    pair_iter = zip_longest(text, text[1:])
    for char, next_ in pair_iter:
        if char == '\\' and next_ == '\\':
            # Double backslash -> single backslash
            unescaped.append('\\')
            next(pair_iter)
        elif char == '\\' and next_ == 't':
            # Backslash + t -> tab character
            unescaped.append('\t')
            next(pair_iter)
        else:
            unescaped.append(char)
    return ''.join(unescaped)

# Modified from https://github.com/spyysalo/string-db-tools
def parse_stringdb_input_line(line):
    """Parse line in database_documents.tsv format, return StringDocument."""
    line = line.rstrip('\n')
    fields = line.split('\t')
    doc_id, other_ids, authors, forum, year, text = fields
    text = stringdb_unescape_text(text)
    return StringDocument(doc_id, other_ids, authors, forum, year, text)

# Modified from https://github.com/spyysalo/string-db-tools
def parse_stringdb_span_line(line):
    """Parse line in all_matches.tsv format, return StringSpan."""
    line = line.rstrip('\n')
    fields = line.split('\t')
    doc_id, par_num, sent_num, start, end, text, type_id, serial = fields
    start, end = int(start), int(end)
    return StringSpan(
        doc_id, par_num, sent_num, start, end, text, type_id, serial)


# Modified from https://github.com/spyysalo/string-db-tools
def stream_documents(fn):
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            try:
                document = parse_stringdb_input_line(l)
            except Exception as e:
                raise ValueError('failed to parse {} line {}'.format(fn, ln))
            yield document


if __name__ == '__main__':
    import sys

    # Test I/O
    for fn in sys.argv[1:]:
        for doc in stream_documents(fn):
            print(doc.doc_id, len(doc.text.split()), 'tokens')