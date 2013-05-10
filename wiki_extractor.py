#!/usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
#  Multithread-Wikipedia-Extractor
#  For SMP based architectures
#  Version: 1.1 (May 9, 2013)
# =============================================================================
#  Copyright (c) 2012. Leonardo Souza (leonardossz@gmail.com).
# =============================================================================
#
#  Contributors:
#	Juan Manuel Caicedo (juan@cavorite.com)
#	Humberto Pereira (begini@gmail.com)
#	Siegfried-A. Gevatter (siegfried@gevatter.com), 2013
#	Pedro Assis (pedroh2306@gmail.com)
#
# =============================================================================
#  This a modified version of the orginal Wikipedia Extractor by
#  Giuseppe Attardi (attardi@di.unipi.it), University of Pisa and
#  Antonio Fuschetto (fuschett@di.unipi.it), University of Pisa, the
#  orginal work can be found at http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
# =============================================================================
#  Copyright (c) 2009. Giuseppe Attardi (attardi@di.unipi.it).
# =============================================================================
#
#  multithread-wikipedia-extractor is a free software;
#  you can redistribute it and/or modify it under the
#  terms of the GNU General Public License, version 3,
#  as published by the Free Software Foundation.
#
#  multithread-wikipedia-extractor is distributed in the hope
#  that it will be useful,  but WITHOUT ANY WARRANTY; without
#  even the implied warranty of  MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""
Multithread Wikipedia Extractor:

Extracts and cleans text from Wikipedia database dump and stores output in a
number of files of similar size in a given directory.
Each file contains several documents in the format:

	<doc id="" url="" title="">
        ...
        </doc>
"""

import Queue, threading, argparse, shutil, json
import sys, re, bz2, multiprocessing
import os.path, os, string, random, traceback
from htmlentitydefs import name2codepoint
#from lxml import etree

# compatible with the original work from the TANL project
# see http://medialab.di.unipi.it/wiki/Tanl for more info
TANL = "tanl"
# outputs json objects
JSON = "json"

class WikiCleanerThread(threading.Thread):

    _filename_lock = threading.RLock()

    def __init__(self, queue, outputdir, maxfilesize, prefix, compress, output_format):
        threading.Thread.__init__(self)
        self._queue = queue
        self._maxfilesize = maxfilesize
        self._prefix = prefix
        self._compress = compress
        self._outputdir = outputdir
        self._output_format = output_format
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        self._outfile = None

    @classmethod
    def _get_file(cls, outputdir, compress=False):
        with cls._filename_lock:
            fpath = None
            while not fpath or os.path.exists(fpath):
                fname = ''.join([random.choice(string.letters) for _ in range(16)])
                ext = ".raw" if not compress else ".raw.bz2"
                fpath = os.path.join(outputdir, fname + ext)

            if compress:
                return bz2.BZ2File(fpath, 'w')

            return open(fpath, 'w')

    def _geturl(self, wiki_id):
        return "%s?curid=%s" % (self._prefix, wiki_id)

    def _write(self, wiki_id, wiki_title, wiki_text):
        if not self._outfile:
            self._outfile = self._get_file(self._outputdir, self._compress)

        print "[%s] [%s]" % (wiki_id.encode('utf-8'), wiki_title.encode('utf-8'))

        url = self._geturl(wiki_id)

        if self._output_format == TANL:
            header = '<doc id="%s" url="%s" title="%s">%s\n' % (wiki_id, url, wiki_title, wiki_title)
            body = ' '.join(compact(clean(wiki_text))).strip()
            footer = "\n</doc>"
            self._outfile.write(header.encode("utf-8"))
            self._outfile.write(body.encode("utf-8"))
            self._outfile.write(footer.encode("utf-8"))

        elif self._output_format == JSON:
            article = dict(id=wiki_id, url=url, title=wiki_title, text=wiki_text)
            self._outfile.write(json.dumps(article, encoding='utf-8') + '\n')

        if self._outfile.tell() > self._maxfilesize:
            self._outfile.close()
            self._outfile = None

    def _clean(self, page_elem):
        # wiki xml dumps has namespace
        # use xmlns from the page element
        def TAG(tag):
            return page_elem.tag.split("page")[0] + tag

        wiki_id = page_elem.find(TAG("id")).text.strip()
        wiki_title = page_elem.find(TAG("title")).text.strip()
        revision_elem = page_elem.find(TAG("revision"))
        if revision_elem is not None:
            text_elem = revision_elem.find(TAG("text"))
            if text_elem is not None:
                wiki_text = text_elem.text.strip()
                self._write(wiki_id, wiki_title, wiki_text)

    def run(self):
        while True:
            page_elem = None
            try:
                page_elem = self._queue.get(timeout=1)
                if page_elem is not None:
                    self._clean(page_elem)
            except Queue.Empty:
                break
            except:
                traceback.print_exc(file=sys.stdout)
            finally:
                if page_elem is not None:
                    page_elem.clear()
                    self._queue.task_done()

        print "%s done" % self.name

##
# Whether to preseve links in output
#
keepLinks = False

##
# Whether to transform sections into HTML
#
keepSections = False

##
# Recognize only these namespaces
# w: Internal links to the Wikipedia
#
acceptedNamespaces = set([])

##
# Drop these elements from article text
#
discardElements = set([
        'gallery', 'timeline', 'noinclude', 'pre',
        'table', 'tr', 'td', 'th', 'caption',
        'form', 'input', 'select', 'option', 'textarea',
        'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'menu', 'dir',
        'ref', 'references', 'img', 'imagemap', 'source'
        ])

#=========================================================================
#
# MediaWiki Markup Grammar

# Template = "{{" [ "msg:" | "msgnw:" ] PageName { "|" [ ParameterName "=" AnyText | AnyText ] } "}}" ;
# Extension = "<" ? extension ? ">" AnyText "</" ? extension ? ">" ;
# NoWiki = "<nowiki />" | "<nowiki>" ( InlineText | BlockText ) "</nowiki>" ;
# Parameter = "{{{" ParameterName { Parameter } [ "|" { AnyText | Parameter } ] "}}}" ;
# Comment = "<!--" InlineText "-->" | "<!--" BlockText "//-->" ;
#
# ParameterName = ? uppercase, lowercase, numbers, no spaces, some special chars ? ;
#
#===========================================================================

selfClosingTags = [ 'br', 'hr', 'nobr', 'ref', 'references' ]

# handle 'a' separetely, depending on keepLinks
ignoredTags = [
        '', 'big', 'blockquote', 'center', 'cite', 'div', 'em',
        'font', 'h1', 'h2', 'h3', 'h4', 'hiero', 'i', 'kbd', 'nowiki',
        'p', 'plaintext', 's', 'small', 'span', 'strike', 'strong',
        'sub', 'sup', 'tt', 'u', 'var',
]

placeholder_tags = {'math':'formula', 'code':'codice'}

##
# Normalize title
def normalizeTitle(title):
  # remove leading whitespace and underscores
  title = title.strip(' _')
  # replace sequences of whitespace and underscore chars with a single space
  title = re.compile(r'[\s_]+').sub(' ', title)

  m = re.compile(r'([^:]*):(\s*)(\S(?:.*))').match(title)
  if m:
      prefix = m.group(1)
      if m.group(2):
          optionalWhitespace = ' '
      else:
          optionalWhitespace = ''
      rest = m.group(3)

      ns = prefix.capitalize()
      if ns in acceptedNamespaces:
          # If the prefix designates a known namespace, then it might be
          # followed by optional whitespace that should be removed to get
          # the canonical page name
          # (e.g., "Category:  Births" should become "Category:Births").
          title = ns + ":" + rest.capitalize()
      else:
          # No namespace, just capitalize first letter.
	  # If the part before the colon is not a known namespace, then we must
          # not remove the space after the colon (if any), e.g.,
          # "3001: The_Final_Odyssey" != "3001:The_Final_Odyssey".
          # However, to get the canonical page name we must contract multiple
          # spaces into one, because
          # "3001:   The_Final_Odyssey" != "3001: The_Final_Odyssey".
          title = prefix.capitalize() + ":" + optionalWhitespace + rest
  else:
      # no namespace, just capitalize first letter
      title = title.capitalize();
  return title

##
# Removes HTML or XML character references and entities from a text string.
#
# @param text The HTML (or XML) source text.
# @return The plain text, as a Unicode string, if necessary.

def unescape(text):
    def fixup(m):
        text = m.group(0)
        code = m.group(1)
        try:
            if text[1] == "#":  # character reference
                if text[2] == "x":
                    return unichr(int(code[1:], 16))
                else:
                    return unichr(int(code))
            else:               # named entity
                return unichr(name2codepoint[code])
        except:
            return text # leave as is

    return re.sub("&#?(\w+);", fixup, text)

# Match HTML comments
comment = re.compile(r'<!--.*?-->', re.DOTALL)

# Match elements to ignore
discard_element_patterns = []
for tag in discardElements:
    pattern = re.compile(r'<\s*%s\b[^>]*>.*?<\s*/\s*%s>' % (tag, tag), re.DOTALL | re.IGNORECASE)
    discard_element_patterns.append(pattern)

# Match ignored tags
ignored_tag_patterns = []
def ignoreTag(tag):
    left = re.compile(r'<\s*%s\b[^>]*>' % tag, re.IGNORECASE)
    right = re.compile(r'<\s*/\s*%s>' % tag, re.IGNORECASE)
    ignored_tag_patterns.append((left, right))

for tag in ignoredTags:
    ignoreTag(tag)

# Match selfClosing HTML tags
selfClosing_tag_patterns = []
for tag in selfClosingTags:
    pattern = re.compile(r'<\s*%s\b[^/]*/\s*>' % tag, re.DOTALL | re.IGNORECASE)
    selfClosing_tag_patterns.append(pattern)

# Match HTML placeholder tags
placeholder_tag_patterns = []
for tag, repl in placeholder_tags.items():
    pattern = re.compile(r'<\s*%s(\s*| [^>]+?)>.*?<\s*/\s*%s\s*>' % (tag, tag), re.DOTALL | re.IGNORECASE)
    placeholder_tag_patterns.append((pattern, repl))

# Match preformatted lines
preformatted = re.compile(r'^ .*?$', re.MULTILINE)

# Match external links (space separates second optional parameter)
externalLink = re.compile(r'\[\w+.*? (.*?)\]')
externalLinkNoAnchor = re.compile(r'\[\w+[&\]]*\]')

# Matches bold/italic
bold_italic = re.compile(r"'''''([^']*?)'''''")
bold = re.compile(r"'''(.*?)'''")
italic_quote = re.compile(r"''\"(.*?)\"''")
italic = re.compile(r"''([^']*)''")
quote_quote = re.compile(r'""(.*?)""')

# Matches space
spaces = re.compile(r' {2,}')

# Matches dots
dots = re.compile(r'\.{4,}')

# A matching function for nested expressions, e.g. namespaces and tables.
def dropNested(text, openDelim, closeDelim):
    openRE = re.compile(openDelim)
    closeRE = re.compile(closeDelim)
    # partition text in separate blocks { } { }
    matches = []                # pairs (s, e) for each partition
    nest = 0                    # nesting level
    start = openRE.search(text, 0)
    if not start:
        return text
    end = closeRE.search(text, start.end())
    next = start
    while end:
        next = openRE.search(text, next.end())
        if not next:            # termination
            while nest:         # close all pending
                nest -=1
                end0 = closeRE.search(text, end.end())
                if end0:
                    end = end0
                else:
                    break
            matches.append((start.start(), end.end()))
            break
        while end.end() < next.start():
            # { } {
            if nest:
                nest -= 1
                # try closing more
                last = end.end()
                end = closeRE.search(text, end.end())
                if not end:     # unbalanced
                    if matches:
                        span = (matches[0][0], last)
                    else:
                        span = (start.start(), last)
                    matches = [span]
                    break
            else:
                matches.append((start.start(), end.end()))
                # advance start, find next close
                start = next
                end = closeRE.search(text, next.end())
                break           # { }
        if next != start:
            # { { }
            nest += 1
    # collect text outside partitions
    res = ''
    start = 0
    for s, e in  matches:
        res += text[start:s]
        start = e
    res += text[start:]
    return res

def dropSpans(matches, text):
    """Drop from text the blocks identified in matches"""
    matches.sort()
    res = ''
    start = 0
    for s, e in  matches:
        res += text[start:s]
        start = e
    res += text[start:]
    return res

# Match interwiki links, | separates parameters.
# First parameter is displayed, also trailing concatenated text included
# in display, e.g. s for plural).
#
# Can be nested [[File:..|..[[..]]..|..]], [[Category:...]], etc.
# We first expand inner ones, than remove enclosing ones.
#
wikiLink = re.compile(r'\[\[([^[]*?)(?:\|([^[]*?))?\]\](\w*)')

parametrizedLink = re.compile(r'\[\[.*?\]\]')

# Function applied to wikiLinks
def make_anchor_tag(match):
    global keepLinks
    link = match.group(1)
    colon = link.find(':')
    if colon > 0 and link[:colon] not in acceptedNamespaces:
        return ''
    trail = match.group(3)
    anchor = match.group(2)
    if not anchor:
        anchor = link
    anchor += trail
    if keepLinks:
        return '<a href="%s">%s</a>' % (link, anchor)
    else:
        return anchor

def clean(text):

    # FIXME: templates should be expanded
    # Drop transclusions (template, parser functions)
    # See: http://www.mediawiki.org/wiki/Help:Templates
    text = dropNested(text, r'{{', r'}}')

    # Drop tables
    text = dropNested(text, r'{\|', r'\|}')

    # Expand links
    text = wikiLink.sub(make_anchor_tag, text)
    # Drop all remaining ones
    text = parametrizedLink.sub('', text)

    # Handle external links
    text = externalLink.sub(r'\1', text)
    text = externalLinkNoAnchor.sub('', text)

    # Handle bold/italic/quote
    text = bold_italic.sub(r'\1', text)
    text = bold.sub(r'\1', text)
    text = italic_quote.sub(r'&quot;\1&quot;', text)
    text = italic.sub(r'&quot;\1&quot;', text)
    text = quote_quote.sub(r'\1', text)
    text = text.replace("'''", '').replace("''", '&quot;')

    ################ Process HTML ###############

    # turn into HTML
    text = unescape(text)
    # do it again (&amp;nbsp;)
    text = unescape(text)

    # Collect spans

    matches = []
    # Drop HTML comments
    for m in comment.finditer(text):
            matches.append((m.start(), m.end()))

    # Drop self-closing tags
    for pattern in selfClosing_tag_patterns:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end()))

    # Drop ignored tags
    for left, right in ignored_tag_patterns:
        for m in left.finditer(text):
            matches.append((m.start(), m.end()))
        for m in right.finditer(text):
            matches.append((m.start(), m.end()))

    # Bulk remove all spans
    text = dropSpans(matches, text)

    # Cannot use dropSpan on these since they may be nested
    # Drop discarded elements
    for pattern in discard_element_patterns:
        text = pattern.sub('', text)

    # Expand placeholders
    for pattern, placeholder in placeholder_tag_patterns:
        index = 1
        for match in pattern.finditer(text):
            text = text.replace(match.group(), '%s_%d' % (placeholder, index))
            index += 1

    text = text.replace('<<', u'«').replace('>>', u'»')

    #############################################

    # Drop preformatted
    # This can't be done before since it may remove tags
    text = preformatted.sub('', text)

    # Cleanup text
    text = text.replace('\t', ' ')
    text = spaces.sub(' ', text)
    text = dots.sub('...', text)
    text = re.sub(u' (,:\.\)\]»)', r'\1', text)
    text = re.sub(u'(\[\(«) ', r'\1', text)
    text = re.sub(r'\n\W+?\n', '\n', text) # lines with only punctuations
    text = text.replace(',,', ',').replace(',.', '.')
    return text

section = re.compile(r'(==+)\s*(.*?)\s*\1')

def compact(text):
    """Deal with headers, lists, empty sections, residuals of tables"""
    page = []                   # list of paragraph
    headers = {}                # Headers for unfilled sections
    emptySection = False        # empty sections are discarded
    inList = False              # whether opened <UL>

    for line in text.split('\n'):

        if not line:
            continue
        # Handle section titles
        m = section.match(line)
        if m:
            title = m.group(2)
            lev = len(m.group(1))
            if keepSections:
                page.append("<h%d>%s</h%d>" % (lev, title, lev))
            if title and title[-1] not in '!?':
                title += '.'
            headers[lev] = title
            # drop previous headers
            for i in headers.keys():
                if i > lev:
                    del headers[i]
            emptySection = True
            continue
        # Handle page title
        if line.startswith('++'):
            title = line[2:-2]
            if title:
                if title[-1] not in '!?':
                    title += '.'
                page.append(title)
        # handle lists
        elif line[0] in '*#:;':
            if keepSections:
                page.append("<li>%s</li>" % line[1:])
            else:
                continue
        # Drop residuals of lists
        elif line[0] in '{|' or line[-1] in '}':
            continue
        # Drop irrelevant lines
        elif (line[0] == '(' and line[-1] == ')') or line.strip('.-') == '':
            continue
        elif len(headers):
            items = headers.items()
            items.sort()
            for (i, v) in items:
                page.append(v)
            headers.clear()
            page.append(line)   # first line
            emptySection = False
        elif not emptySection:
            page.append(line)

    return page

def handle_unicode(entity):
    numeric_code = int(entity[2:-1])
    if numeric_code >= 0x10000: return ''
    return unichr(numeric_code)

def process_data(inputdump, outputdir, maxfilesize, compress, outformat):

    # we expects large dumps so we are using iterparse method
    context = etree.iterparse(inputdump)
    context = iter(context)

    # discover prefix from the xml dump file
    # /mediawiki/siteinfo/base
    prefix = None
    for event, elem in context:
        if event == "end" and elem.tag.endswith("base"):
            prefix =  elem.text[:elem.text.rfind("/")]
            break
    print "base url: %s" % prefix

    # initialize wiki page queue
    queue = Queue.Queue(maxsize=100)

    # start worker threads
    workers = []
    for _ in range(multiprocessing.cpu_count()):
        cleaner = WikiCleanerThread(queue, outputdir, maxfilesize, prefix, compress, outformat)
        cleaner.setDaemon(True)
        cleaner.start()
        workers.append(cleaner)

    # put element pages in the queue to be processed by the cleaner threads
    for event, elem in context:
        if event == "end" and elem.tag.endswith("page"):
            queue.put(elem)

    # wait an empty queue
    queue.join()

    for w in workers:
        w.join()

    print "finished"

def main():
    global keepLinks, keepSections

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("wikidump", help="XML wiki dump file")
    parser.add_argument("outputdir", help="output directory")
    parser.add_argument("-w", "--overwrite", default=False, action="store_const", const=True, help="Overwrite existing output dir")
    parser.add_argument("-b", "--bytes", default="25M", help="put specified bytes per output file (default is %(default)s)", metavar="n[KM]")
    parser.add_argument("-c", "--compress", default=False, action="store_const", const=True, help="compress output files using bzip")
    parser.add_argument("-l", "--links", default=False, action="store_const", const=True, help="preserve links")
    parser.add_argument("-s", "--sections", default=False, action="store_const", const=True, help="preserve sections")
    parser.add_argument("-f", "--format", choices=(TANL, JSON), default=JSON, help="choose output format default is %(default)s")

    args = parser.parse_args()

    keepLinks = args.links
    keepSections = args.sections

    if not keepLinks:
        ignoreTag('a')

   # Minimum size of output files
    min_file_size = 200 * 1024
    try:
        if args.bytes[-1] in 'kK':
            file_size = int(args.bytes[:-1]) * 1024
        elif args.bytes[-1] in 'mM':
            file_size = int(args.bytes[:-1]) * 1024 * 1024
        else:
            file_size = int(args.bytes)
        if file_size < min_file_size: raise ValueError()
    except ValueError:
        print >> sys.stderr, \
        'Insufficient or invalid bytes size (minimum per output is %d bytes)' \
        % min_file_size
        return

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    else:
        if args.overwrite:
            shutil.rmtree(args.outputdir)
            os.makedirs(args.outputdir)
        else:
            raise ValueError("%s already exists, use --overwrite to recreate" % args.outputdir)

    if args.wikidump.lower().endswith("bz2"):
        with bz2.BZ2File(args.wikidump, 'r') as inputdump:
            process_data(inputdump, args.outputdir, file_size, args.compress, args.format.lower())
    else:
        with open(args.wikidump, 'r') as inputdump:
            process_data(inputdump, args.outputdir, file_size, args.compress, args.format.lower())


if __name__ == '__main__':
    main()
