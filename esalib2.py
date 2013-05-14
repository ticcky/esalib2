#!/usr/bin/env python
# coding: utf-8
"""
    esalib2.esalib2
    ~~~~~~~~~~~~~~~

    A description which can be long and explain the complete
    functionality of this module even with indented code examples.
    Class/Function however should not be documented here.

    :copyright: Lukas Zilka (2013)
    :license: Apache License Version 2.0, January 2004 http://www.apache.org/licenses
"""

import os
import bz2
import time
from collections import Counter
import sqlite3
import struct
import math
import datetime
import re

import xml.etree.cElementTree as ET
import wiki_extractor
import porter

# run pdb when error occurs
import pdberr
pdberr.init()


def binarize(lst):
    """Convert a list of (int, float) tuples into a binary string."""
    res = []
    for item in lst:
        res.append(struct.pack('if', *item))
    return "".join(res)


def sliding_window_filter(doc_list, window_size=100, window_thresh=0.05):
    """Given a list of documents and their scores, build a list that does not
    contain in the tail values that have changed less during last @window_size
    items, than @window_thresh percent of the maximal value."""
    res = []
    max_score = None
    for i, (doc_id, doc_val, ) in enumerate(doc_list):
        if max_score is None:
            max_score = doc_val

        if len(res) >= window_size:
            window_change = doc_list[max(0, i - window_size)][1] - doc_list[max(0, i - 1)][1]
            if max_score * window_thresh > window_change:
                break

        res.append((doc_id, doc_val, ))

    return res


def normalize_vector(vector, vector_sq_sum=None):
    if not vector_sq_sum:
        vector_sq_sum = sum(i**2 for i in vector)

    for i, (doc_id, val) in enumerate(vector):
        vector[i] = (doc_id, val / vector_sq_sum)


class FilterStem:
    def __init__(self):
        self.stemmer = porter.PorterStemmer()

    def __call__(self, tokens):
        for token in tokens:
            yield self.stemmer.stem(token)


class FilterStopwords:
    def __init__(self, sw_set):
        self.sw_set = sw_set

    @classmethod
    def from_set(cls, sw_set):
        return FilterStopwords(sw_set)

    def __call__(self, tokens):
        for token in tokens:
            if not token in self.sw_set:
                yield token

def filter_lowercase(tokens):
    for token in tokens:
        yield token.lower()

def filter_gibberish(tokens):
    for token in tokens:
        yield token


def filter_chain(iterable, chain):
    if len(chain) > 0:
        my_fltr_fn = chain[0]
        return filter_chain(my_fltr_fn(iterable), chain[1:])
    else:
        return iterable


class ProgressMeasure(object):
    def __init__(self, label=None, target=None):
        self.cntr = 0
        self.last_time = 0
        self.label = label
        self.target = target

    def tick(self):
        self.cntr += 1
        if time.time() - self.last_time > 1.0:
            print datetime.datetime.now(),
            if self.label:
                print self.label,
            if self.target is not None:
                print "(%.4f%%)" % (float(self.cntr) / self.target * 100),
            print self.cntr

            self.last_time = time.time()


class Document(object):
    """Document record of a document from a document collection."""
    doc_id = None     # unique integer id of the document
    title = None      # document title
    content = None    # document text
    meta = None       # additional metadata

    def __init__(self, doc_id, title, content):
        self.doc_id = doc_id
        self.title = title
        self.content = content


class ESATerm(object):
    """Represents a term from the ESA background."""
    term_id = None    # term integer id
    doc_list = None   # document list
    word_map = None   # maps words to

    def __init__(self, term_id, doc_list):
        self.term_id = term_id
        self.doc_list = doc_list


class DocumentIterator(object):
    """Base class for iterating over a document collection."""

    def __iter__(self):
        """Goes over the underlying document collection and yields
        Document classes."""
        raise NotImplementedException("Document iterator needs to implement this method.")


class WikidumpStreamDI(DocumentIterator):
    """Document iterator over the Wikipedia dump."""
    def __init__(self, bzname, limit=None):
        super(WikidumpStreamDI, self).__init__()
        self.bzname = bzname  # filename of the bzipped dump
        self.limit = limit

    def _clean_doc(self, content):
        return wiki_extractor.clean(content)

    def __iter__(self):
        """Load the dump, decompress it in a streaming fashion, parse it and
        yield the Wikipedia articles."""
        bzfile = bz2.BZ2File(self.bzname, "r")

        # start XML parsing
        it = ET.iterparse(bzfile, events=("start", "end", ))
        _, root = next(it)

        # note the time and counts (for performance measurements)
        pm = ProgressMeasure(label="loading wikipedia articles; done: ", target=self.limit)
        for i, (ev, el, ) in enumerate(it):
            if self.limit is not None and i >= self.limit:
                break

            if ev == "end" and el.tag.endswith('page'):
                try:
                    doc_id = next(el.iterfind('{http://www.mediawiki.org/xml/export-0.8/}id')).text
                    title = next(el.iterfind('{http://www.mediawiki.org/xml/export-0.8/}title')).text
                    content = next(el.iterfind('{http://www.mediawiki.org/xml/export-0.8/}revision/{http://www.mediawiki.org/xml/export-0.8/}text')).text
                    content = self._clean_doc(content)
                    yield Document(doc_id=doc_id, title=title, content=content)
                except Exception, e:
                    print "Error processing document", str(e)

                root.clear()  # throw away the data from the parsed tree, as they are processed

                pm.tick()


class WordMap(dict):
    def save_idf(self, conn):
        curr, save_curr = conn.cursor(), conn.cursor()
        res = {}
        doc_cnt = float(curr.execute("SELECT count(*) from (select doc_id, count(*) from doc_term_freq group by doc_id)").fetchone()[0])
        for term_id, term_cnt in curr.execute("SELECT term_id,count(doc_id) FROM doc_term_freq GROUP BY term_id;"):
            idf = math.log(doc_cnt / term_cnt)
            save_curr.execute("INSERT INTO term_idf VALUES (?, ?)", (term_id, idf,) )
            res[term_id] = idf

        return res

    def load_idf(self, conn):
        curr = conn.cursor()
        term_idf = {}
        for term_id, idf in curr.execute("SELECT term_id, term_idf FROM term_idf;"):
            term_idf[term_id] = idf

        return term_idf

    def save(self, conn):
        save_curr = conn.cursor()
        for word, word_id in self.iteritems():
            save_curr.execute("INSERT INTO term_wordmap VALUES (?, ?)",
                (word, word_id))


class WhitespaceTokenizer(object):
    """For tokenizing the text on whitespaces."""
    def tokenize(self, content):
        """Splits the input text into tokens."""
        for token in re.findall('[A-Za-z-]*', content):
            if len(token) > 0:
                yield token


class BackgroundBuilder(object):
    """Build the ESA background."""

    # queries for initializing the database
    drop_table_stmts = [
        "DROP TABLE IF EXISTS doc_term_freq",
        "DROP TABLE IF EXISTS term",
        "DROP TABLE IF EXISTS term_idf",
        "DROP TABLE IF EXISTS term_wordmap",
    ]
    create_table_stmts = [
        "CREATE TABLE doc_term_freq (term_id int, doc_id int, freq int)",
        "CREATE TABLE term (term_id int, term_vector binary)",
        "CREATE TABLE term_idf (term_id int, idf float)",
        "CREATE TABLE term_wordmap (term text, term_id int)",
    ]

    def __init__(self, db_file, token_filter_chain):
        self.wordmap = WordMap()
        self.tokenizer = WhitespaceTokenizer()
        self.token_filter_chain = token_filter_chain

        self.db_file = db_file
        self.conn = None
        self._connect_to_db()

    def compute_termfrequency(self, text):
        """Given the text compute freqency of each word."""
        return Counter(self.tokenize(text)).most_common()

    def tokenize(self, text):
        """Split text into tokens."""
        for token in filter_chain(self.tokenizer.tokenize(text), self.token_filter_chain):
            if not token in self.wordmap:  # add token into wordmap if needed
                self.wordmap[token] = len(self.wordmap)

            yield self.wordmap[token]

    def prepare_database(self):
        """Create/drop tables in database to prepare it for building the ESA background."""
        curr = self.conn.cursor()
        for tbl_stmt in self.drop_table_stmts:
            curr.execute(tbl_stmt)

        for tbl_stmt in self.create_table_stmts:
            curr.execute(tbl_stmt)

    def create_index(self, table_name, curr):
        """Instruct the database to create index on the given table."""
        curr.execute("CREATE INDEX ndx_dtf ON %s (term_id, freq)" % table_name)

    def save_doc_freq(self, curr, doc_id, frq_vector):
        """Save records about term freqency of terms in the given document."""
        for term_id, freq in frq_vector:
            tf = 1.0 + math.log(freq)
            curr.execute("INSERT INTO doc_term_freq VALUES ({term_id}, {doc_id}, {freq})".format(term_id=term_id, freq=tf, doc_id=doc_id))

    def save_term(self, curr, term):
        """Insert the term along with its document vector to the database."""
        doc_list = sliding_window_filter(term.doc_list)  # be
        curr.execute("INSERT INTO term VALUES(?, ?)",
            (term.term_id, buffer(binarize(doc_list))))

    def save_terms(self, term_idf, min_freq=15):
        curr = self.conn.cursor()
        save_curr = self.conn.cursor()
        for term in self.iter_terms(curr, term_idf, min_freq=min_freq):
            self.save_term(save_curr, term)

    def iter_terms(self, curr, term_idf, min_freq=15):
        res = curr.execute("SELECT term_id, doc_id, freq FROM doc_term_freq WHERE freq > ? ORDER BY term_id, freq DESC ", (min_freq, ))

        term_rec = None
        curr_term = None
        curr_doc_list = []
        curr_doc_list_sum = 0.0

        while True:
            term_rec = res.fetchone()
            if term_rec is None:
                break

            if curr_term != term_rec[0] and curr_term is not None:
                normalize_vector(curr_doc_list, curr_doc_list_sum)
                yield ESATerm(term_id=curr_term, doc_list=curr_doc_list)

                curr_doc_list = []
                curr_doc_list_sum = 0.0

            curr_term = term_rec[0]
            tfidf = term_idf[curr_term] * term_rec[2]
            curr_doc_list.append((term_rec[1], tfidf, ))
            curr_doc_list_sum += tfidf


    def _connect_to_db(self):
        self.conn = sqlite3.connect(self.db_file)

    def load_documents(self, doc_iter):
        curr = self.conn.cursor()
        for doc in doc_iter:
            content = doc.content
            term_freq = self.compute_termfrequency(content)
            self.save_doc_freq(curr, doc.doc_id, term_freq)
        self.create_index("doc_term_freq", curr)
        curr.close()

    def build(self, doc_iter, skip_load=False, min_freq=15):
        if not skip_load:
            self.prepare_database()
            self.load_documents(doc_iter)
            self.conn.commit()

            self.wordmap.save(self.conn)
            self.conn.commit()

            # build idf map
            term_idf = self.wordmap.save_idf(self.conn)
            self.conn.commit()
        else:
            term_idf = self.wordmap.load_idf(self.conn)

        self.save_terms(term_idf, min_freq=min_freq)
        self.conn.commit()


class ESA(object):
    def __init__(self, bg_file, token_filter_chain):
        self.bg_file = bg_file
        self.token_filter_chain = token_filter_chain
        self.tokenizer = WhitespaceTokenizer()

        self.conn = sqlite3.connect(bg_file)
        self.curr = self.conn.cursor()
        self.esa_index = None
        self.stemmer = None

        self._load()

    def _load(self):
        vectors = self.curr.execute("SELECT term, term_vector FROM term fv LEFT JOIN term_wordmap wm ON wm.term_id = fv.term_id")
        self.esa_index = {}
        for term, tv_str in vectors:
            tv = {}
            for i in range(0, len(tv_str), 8):
                doc_id, doc_val = struct.unpack('if', tv_str[i:i+8])
                tv[doc_id] = doc_val
            self.esa_index[term] = tv

    def tokenize(self, text):
        for word in filter_chain(self.tokenizer.tokenize(text), self.token_filter_chain):
            yield word

    def get_vector(self, text):
        used_dims = set()
        used_tvs = []
        for token in self.tokenize(text):
            new_tv = self.esa_index[token]
            used_dims.update(new_tv.keys())
            used_tvs.append(new_tv)

        res_vec = {}
        for dim in used_dims:
            res_vec[dim] = sum(tv.get(dim, 0.0) for tv in used_tvs)

        return res_vec

    def similarity(self, v1, v2):
        dims = set(v1.keys() + v2.keys())
        res = 0.0
        res_norm_v1 = 0.0
        res_norm_v2 = 0.0
        for dim in dims:
            v1_val = v1.get(dim, 0.0)
            v2_val = v2.get(dim, 0.0)

            res += v1_val * v2_val
            res_norm_v1 += v1_val ** 2
            res_norm_v2 += v2_val ** 2

        return res / math.sqrt(res_norm_v1 * res_norm_v2)


def get_token_filter_chain():
    return [
            filter_lowercase,
            FilterStem(),
            FilterStopwords.from_set(set(['a', 'the'])),
            filter_gibberish,
    ]


def test_esa():
    esa = ESA("esa_bg.db", token_filter_chain=get_token_filter_chain())
    test_lst = [
        ('four', 'western'),
        ('princip', 'protest'),
        ('sleep', 'princip'),
        ('money', 'sleep'),
        ('money', 'bank'),
        ('beautiful day', 'good day'),
        ('beautiful day', 'bad day'),
    ]
    for w1, w2 in test_lst:
        v1 = esa.get_vector(w1)
        v2 = esa.get_vector(w2)
        print w1, w2, esa.similarity(v1, v2)


def test_build_background():
    wsdi = WikidumpStreamDI('/xdisk/devel/esalib/enwiki-20130403-pages-articles.xml.bz2', limit=10000)

    bb = BackgroundBuilder("esa_bg.db", token_filter_chain=get_token_filter_chain())
    bb.build(wsdi,
        skip_load=False,
        min_freq=0
    )


if __name__ == '__main__':
    test_build_background()
    test_esa()