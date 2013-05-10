import os
import bz2
import time
from collections import Counter
import sqlite3
import struct
import math
import functools
import unicodedata

import xml.etree.cElementTree as ET

#from mwlib.uparser import simpleparse

#import mwlib
import wiki_extractor
import porter

import pdberr
pdberr.init()

def sanitize_nodes(nodes_):
    """Given an iterable of nodes, returns plain text
    without links, images, etc."""
    l = []
    nodes = mwlib.parser.nodes

    for node in mwlib.refine.core.walknode(nodes_):
        if isinstance(node, nodes.Text):
            l.append(node.caption)
        elif isinstance(node, nodes.Link):
            if not node.children:
                l.append(node.caption or node.target)
        elif isinstance(node, nodes.NamedURL):
            if not node.children:
                l.append(node.caption or node.target)

    return u_normalize(u"".join(l))


def sanitize_text(text, wikidb=None, inline=False):
    """Expands templates and removes markup"""
    if not text: return text

    if inline:
        text = "<span>%s</span>" % text

    if not wikidb:
        wikidb = mwlib.expander.DictDB()

    te = mwlib.expander.Expander(text, wikidb=wikidb)
    text = te.expandTemplates()

    xopts = mwlib.refine.core.XBunch()
    xopts.expander = mwlib.expander.Expander("", "pagename", wikidb=wikidb)
    xopts.nshandler = mwlib.nshandling.nshandler(wikidb.get_siteinfo())

    parsed = mwlib.refine.core.parse_txt(text, xopts)
    mwlib.refine.compat._change_classes(parsed)

    s = sanitize_nodes(parsed).strip()

    return s

u_normalize = functools.partial(unicodedata.normalize, "NFC")


#ET.register_namespace("{http://www.mediawiki.org/xml/export-0.8/}", "http://www.mediawiki.org/xml/export-0.8/")

class Document(object):
    id = None
    title = None
    content = None

    def __init__(self, id, title, content):
        self.id = id
        self.title = title
        self.content = content

class Term(object):
    term_id = None
    doc_list = None

    def __init__(self, term_id, doc_list):
        self.term_id = term_id
        self.doc_list = doc_list


class DocumentIterator(object):
    pass


class WikidumpStreamDI(DocumentIterator):
    def __init__(self, bzname):
        super(WikidumpStreamDI, self).__init__()
        self.bzname = bzname

    def __iter__(self):
        bzfile = bz2.BZ2File(self.bzname, "r")

        it = ET.iterparse(bzfile, events=("start", "end", ))
        event, root = next(it)

        sw_time = time.time()
        cntr = 0
        cntr_total = 0
        for (ev, el, ) in it:
            if ev == "end" and el.tag.endswith('page'):
                cntr += 1
                try:
                    doc_id = next(el.iterfind('{http://www.mediawiki.org/xml/export-0.8/}id')).text
                    title = next(el.iterfind('{http://www.mediawiki.org/xml/export-0.8/}title')).text
                    content = next(el.iterfind('{http://www.mediawiki.org/xml/export-0.8/}revision/{http://www.mediawiki.org/xml/export-0.8/}text')).text
                    yield Document(id=doc_id, title=title, content=content)
                except Exception, e:
                    print "Error processing document", str(e)

                root.clear()

            if time.time() - sw_time > 1:
                cntr_total += cntr
                print cntr, cntr_total
                cntr = 0

                sw_time = time.time()

class WhitespaceTokenizer(object):
    def tokenize(self, content):
        for token in content.split():
            yield token




class BackgroundBuilder(object):
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

    def __init__(self):
        self.wordmap = {}
        self.tokenizer = WhitespaceTokenizer()

    def compute_termfrequency(self, content):
        return Counter(self.tokenize(content)).most_common()

    def tokenize(self, content):
        for token in self.tokenizer.tokenize(content):
            if not token in self.wordmap:
                self.wordmap[token] = len(self.wordmap)

            yield self.wordmap[token]

    def prepare_tables(self, curr):
        for tbl_stmt in self.drop_table_stmts:
            curr.execute(tbl_stmt)

        for tbl_stmt in self.create_table_stmts:
            curr.execute(tbl_stmt)

    def create_indexes(self, curr):
        curr.execute("CREATE INDEX ndx_dtf ON doc_term_freq (term_id, freq)")

    def save_doc_freq(self, curr, doc_id, frq_vector):
        for term_id, freq in frq_vector:
            tf = 1.0 + math.log(freq)
            curr.execute("INSERT INTO doc_term_freq VALUES ({term_id}, {doc_id}, {freq})".format(term_id=term_id, freq=tf, doc_id=doc_id))

    def binarize(self, lst):
        res = ''
        for item in lst:
            res += struct.pack('if', *item)
        return res

    def sliding_window_filter(self, doc_list, window_size=100, window_thresh=0.05):
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


    def save_term(self, curr, term):
        doc_list = self.sliding_window_filter(term.doc_list)
        curr.execute("INSERT INTO term VALUES(?, ?)",
            (term.term_id, buffer(self.binarize(doc_list))))

    def _normalize(self, vector, vector_sq_sum):
        for i, (doc_id, val) in enumerate(vector):
            vector[i] = (doc_id, val / vector_sq_sum)

    def iter_terms(self, curr, term_idf, min_freq=15):
        res = curr.execute("SELECT term_id, doc_id, freq FROM doc_term_freq ORDER BY term_id, freq DESC WHERE freq > ?", (min_freq, ))

        term_rec = None
        curr_term = None
        curr_doc_list = []
        curr_doc_list_sum = 0.0

        while True:
            term_rec = res.fetchone()
            if term_rec is None:
                break

            if curr_term != term_rec[0] and curr_term is not None:
                self._normalize(curr_doc_list, curr_doc_list_sum)
                yield Term(term_id=curr_term, doc_list=curr_doc_list)

                curr_doc_list = []
                curr_doc_list_sum = 0.0

            curr_term = term_rec[0]
            tfidf = term_idf[curr_term] * term_rec[2]
            curr_doc_list.append((term_rec[1], tfidf, ))
            curr_doc_list_sum += tfidf


    def save_idf(self, curr, save_curr):
        res = {}
        doc_cnt = float(curr.execute("SELECT count(*) from (select doc_id, count(*) from doc_term_freq group by doc_id)").fetchone()[0])
        for term_id, term_cnt in curr.execute("SELECT term_id,count(doc_id) FROM doc_term_freq GROUP BY term_id;"):
            idf = math.log(doc_cnt / term_cnt)
            save_curr.execute("INSERT INTO term_idf VALUES (?, ?)", (term_id, idf,) )
            res[term_id] = idf

        return res

    def save_wordmap(self, save_curr):
        for word, word_id in self.wordmap.iteritems():
            save_curr.execute("INSERT INTO term_wordmap VALUES (?, ?)",
                (word, word_id))

    def clean_doc(self, content):
        return wiki_extractor.clean(content)


    def build(self, doc_iter, outfile, skip_load=False):
        conn = sqlite3.connect(outfile)
        curr = conn.cursor()

        if not skip_load:
            self.prepare_tables(curr)
            for cntr, doc in enumerate(doc_iter):
                content = self.clean_doc(doc.content)
                term_freq = self.compute_termfrequency(content)
                self.save_doc_freq(curr, doc.id, term_freq)
                #if cntr == 1000:
                #    break
            self.create_indexes(curr)

        save_curr = conn.cursor()
        term_idf = self.save_idf(curr, save_curr)

        for term in self.iter_terms(curr, term_idf):
            self.save_term(save_curr, term)

        self.save_wordmap(save_curr)

        conn.commit()

class ESA(object):
    def __init__(self, bg_file):
        self.bg_file = bg_file
        self.tokenizer = WhitespaceTokenizer()

        self.conn = sqlite3.connect(bg_file)
        self.curr = self.conn.cursor()
        self.esa_index = None

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
        for word in self.tokenizer.tokenize(text):
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
            res_vec[dim] = sum(tv[dim] for tv in used_tvs)

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





def main():
    esa = ESA("esa_bg.db")
    v1 = esa.get_vector("work")
    v2 = esa.get_vector("is")
    print esa.similarity(v1, v2)

def main2():
    wsdi = WikidumpStreamDI('/xdisk/devel/esalib/enwiki-20130403-pages-articles.xml.bz2')

    bb = BackgroundBuilder()
    bb.build(wsdi,
        outfile="esa_bg.db",
        skip_load=False
    )


if __name__ == '__main__':
    main2()
    #main()