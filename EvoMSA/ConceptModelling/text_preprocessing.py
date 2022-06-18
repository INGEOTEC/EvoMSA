# -*- coding: utf-8 -*-
# Copyright 2018 Sabino Miranda with collaboration of Eric S. Tellez

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import re
import os
import logging
from nltk.corpus import stopwords
import unicodedata
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
import sys
import gzip
import json
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

strPatternVowels = "áéíóúüñÁÉÍÓÚÜÑ"
strSimplePuntuation = r'«»~¡!¿?"“”,'


PATH = os.path.abspath(os.path.dirname(__file__))
EMOTICONS_FILE = os.path.join(PATH, "dicEmoticonText.txt")
UNICODE_EMOTICONS_FILE = os.path.join(PATH, "dicUnicodeEmoticonText.txt")

_SPANISH = 'spanish'
_ENGLISH = 'english'
_ITALIAN = 'italian'
_PORTUGUESE = 'portuguese'
_ARABIC = 'arabic'

PUNCTUACTION = ";:,.@\\-\"'/"
SYMBOLS = "()[]¿?¡!{}~<>|"
SKIP_SYMBOLS = set(PUNCTUACTION + SYMBOLS)
# SKIP_WORDS = set(["…", "..", "...", "...."])
_DELIMITER = ' '

strPatternVowels = "áéíóúüñÁÉÍÓÚÜÑ"
strSimplePuntuation = r'«»~¡!¿?"“”,;.\''
dicPuntuation = {"à": "á", "è": "é", "ì": "í", "ò": "ó", "ù": "ú",
                 "À": "Á", "È": "É", "Ì": "Í", "Ò": "Ó", "Ù": "Ú"}

strAllPuntuation = strSimplePuntuation + "'()[];,:/.{}~|<>"

lst_punctuation = list(strAllPuntuation)
# lst_no_stopwords = ['no', 'sí', 'sin', 'muy', 'mi' , 'mí', 'tu', 'tú', 'mis', 'yo']
lst_no_stopwords = []

_HASHTAG = '#'
_USERTAG = '@'
_sURL_TAG = 'url_tag'
_sUSER_TAG = 'user_tag'
_sHASH_TAG = 'hash_tag'
_sNUM_TAG = 'num_tag'
_sDATE_TAG = 'date_tag'
_sENTITY_TAG = 'entity_tag'
_sNEGATIVE_TAG = "negative_tag"
_sPOSITIVE_TAG = "positive_tag"
_sNEUTRAL_TAG = "neutral_tag"
_sNEGATIVE_EMOTICON = "_negativo"
_sPOSITIVE_EMOTICON = "_positivo"
_sNEUTRAL_EMOTICON = "_neutro"
_sEMOTICON_TAG = "<TAG>"

_OPTION_NONE = 0
_OPTION_DELETE = 1
_OPTION_GROUP = 2
_OPTION_USE = 3

_DEBUG_FACTOR = 0.01


class TextPreprocessingError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class TextPreprocessing():

    _BEGIN_TAG = "<s>"
    _END_TAG = "</s>"

    def __init__(self, *args, **kwargs):
        self.basic_init(*args, **kwargs)

    def basic_init(self, lang=_SPANISH,
                   sentence_delim=False,
                   **kwargs):
        if sentence_delim is False:
            self._BEGIN_TAG = ""
            self._END_TAG = ""

        self.lang = lang
        self.sentence_delim = sentence_delim

        logger.info("sws for  {}".format(lang))
        self.stopWords = self.get_stopwords(lang)
        self.tokenizer = TweetTokenizer()
        self.stemmer = None
        if self.lang in [_SPANISH, _ITALIAN, _PORTUGUESE]:
            self.stemmer = SnowballStemmer(_SPANISH, ignore_stopwords=False)
        elif self.lang == _ENGLISH:
            from nltk.stem.porter import PorterStemmer
            self.stemmer = PorterStemmer()
        elif self.lang == _ARABIC:
            from nltk.stem.isri import ISRIStemmer
            self.stemmer = ISRIStemmer()

    # def init(self, lang=_SPANISH, emoticonFile=EMOTICONS_FILE,
    #          unicodeEmoticonFile=UNICODE_EMOTICONS_FILE):

    #     self.lang = lang
    #     self.hshEmoticons = None
    #     self.reEmoticons = None
    #     self.hshUnicodeEmoticons = None
    #     self.reUnicodeEmoticons = None
    #     self.stopWords = self.get_stopwords(lang)
    #     self.loadEmoticons(emoticonFile, unicodeEmoticonFile)
    #     self.tokenizer = TweetTokenizer()
    #     self.stemmer = None
    #     if self.lang in [_SPANISH, _ITALIAN, _PORTUGUESE]:
    #         self.stemmer = SnowballStemmer(_SPANISH, ignore_stopwords=False)
    #     elif self.lang == _ENGLISH:
    #         from nltk.stem.porter import PorterStemmer
    #         self.stemmer = PorterStemmer(ignore_stopwords=False)
    #     elif self.lang == _ARABIC:
    #         from nltk.stem.isri import ISRIStemmer
    #         self.stemmer = ISRIStemmer()

    # def read_json(self, file_name):
    #     if file_name.endswith(".gz"):
    #         _file = gzip.open(file_name, 'r')
    #     else:
    #         _file = io.open(file_name, "r", encoding='utf8')

    #     items = 0
    #     while True:
    #         line = _file.readline()
    #         items += 1
    #         if not line:
    #             break
    #         if isinstance(_file, gzip.GzipFile):
    #             line = line.decode(encoding='utf-8')
    #         line.strip()
    #         if line == "" or line.startswith("#"):
    #             logger.warning("empty line => " + str(items))
    #             continue

    #         yield json.loads(line)

    # def load_corpus(self, file_name):
    #     logger.info("loading corpus... " + file_name)
    #     self.corpus = []
    #     self.targets = []
    #     items = 0
    #     self.max_doc_size = 0
    #     for data in self.read_json(file_name):
    #         text = data['text']
    #         text = self._BEGIN_TAG + " " + text + " " + self._END_TAG + " "
    #         text = self.text_transform(text,
    #                                    remove_stopwords=False,
    #                                    remove_diacritic=True,
    #                                    emo_option=_OPTION_NONE,
    #                                    url=_OPTION_NONE)
    #         tokens = text.split()
    #         doc_size = len(tokens)
    #         if doc_size > 0:
    #             items += 1
    #             self.targets.append(data['klass'])

    #             if doc_size > self.max_doc_size:
    #                 self.max_doc_size = doc_size
        
    #             self.corpus.append(text)

    #         if random.random() < _DEBUG_FACTOR:
    #             logger.debug("sampling...=>" + data['text'])
    #             logger.debug("sampling preprocesssing...=>[" + str(text) + "]")
    #             print(text)
    #             logger.info("text processed: " + str(items))

    #     logger.info("corpus loaded => OK ")

    def norm_chars(self, text, del_diac=True, del_dup=True, max_dup=2, del_punc=True):
        _DELIMITER = ' '
        L = []
        prev = _DELIMITER
        n=1
        for u in unicodedata.normalize('NFD', text):
            if del_diac:
                o = ord(u)
                if 0x300 <= o and o <= 0x036F:
                    continue
            if u in ('\n', '\r', ' ', '\t', _DELIMITER):
                u = _DELIMITER
            if del_punc and u in SKIP_SYMBOLS:
                prev = u
                continue
            if del_dup:
                if prev == u:
                    n += 1
                    if n > max_dup:
                        n = max_dup
                        continue
                    else:
                        n = 1
            prev = u
            L.append(u)
        return "".join(L)

    def get_stopwords(self, lang):
        stopW = []
        for sw in stopwords.words(lang):
            if sw not in lst_no_stopwords:
                stopW.append(sw)
        return stopW

    def filterWords(self, text):
        text = text.strip()
        text = re.sub(r"\n|\r|\t", " ", text)
        text = re.sub(r"^RT\s+", " ", text, flags=re.IGNORECASE)
        # text = re.sub(r"[^a-zA-Z\d" + strPatternVowels + "\_\s@#]", "", text, flags = re.I)
        text = re.sub(strSimplePuntuation, " ", text, flags=re.IGNORECASE)
        tokens = self.tokenizer.tokenize(text)
        t = []
        for w in tokens:
            w = re.sub(r"([a-zA-Z" + strPatternVowels + r"][a-zA-Z" + strPatternVowels + r"])\1\1+", r"\1\1\1", w, flags=re.I)
            w = re.sub(r"([a-zA-Z" + strPatternVowels + r"])\1\1+", r"\1\1\1", w, flags=re.I)
            w = re.sub(r"(^(j[aeiou](j|[aeiou])*)+(\s|\b))|((\s|\b)(j[aeiou](j|[aeiou])*)+(\s|\b))|((\s|\b)(j[aeiou](j|[aeiou])*)+$)", "jajaja", w, flags=re.I)
            w = re.sub(r"(^([aeiou]j(j|[aeiou])*)+(\s|\b))|((\s|\b)([aeiou]j(j|[aeiou])*)+(\s|\b))|((\s|\b)([aeiou]j(j|[aeiou])*)+$)", "jajaja", w, flags=re.I)
            if w not in lst_punctuation:
                if not (re.search(r"[a-zA-Z" + strPatternVowels + r"]", w, flags=re.I) and len(w) == 1):
                    t.append(w)
        text = " ".join(t)
        text = re.sub(r"\s+", " ", text, flags=re.IGNORECASE)
        return text.lower()

    def remove_stopwords(self, text):
        tokens = text.split()
        # tokens = [ tok for tok in tokens if tok.lower() not in self.stopWords and len(tok)>1]
        tokens = [tok for tok in tokens if tok.lower() not in self.stopWords]
        return " ".join(tokens)

    def text_transform(self, text,
                       remove_stopwords=True,
                       remove_diacritic=False,
                       emo_option=_OPTION_NONE,
                       url=_OPTION_NONE, stemming_comp=False):
        '''
        '''
        text = re.sub(r"\n|\r|\t", " ", text)
        # if emo_option in [_OPTION_DELETE, _OPTION_GROUP]:
        #     text = self.process_emoticons(text, emo_option)

        if url in [_OPTION_DELETE, _OPTION_GROUP]:
            if url == _OPTION_DELETE:
                text = re.sub(r"((^|\s+)(http|ftp)s?:(.+?)(\s+|$))", " ", text, flags=re.I)
            else:
                text = re.sub(r"((^|\s+)(http|ftp)s?:(.+?)(\s+|$))", " _url ", text, flags=re.I)
        if remove_stopwords:
            text = self.remove_stopwords(text)

        if remove_diacritic:
            text = self.norm_chars(text, del_diac=True, del_dup=True, max_dup=2, del_punc=True)
            # text = self.norm_chars(text)

        if stemming_comp:
            text = self.stemming_complement(text)

        text = self.filterWords(text)
        return text

    def stemming_complement(self, text):
        if not self.stemmer:
            logger.error("Stemmer is not initialized")
            sys.exit(1)

        tokens = text.split()
        stem_c = []
        for t in tokens:
            s = self.stemmer.stem(t)
            c = t.replace(s, "")
            stem_c.append(c)
        return " ".join(stem_c)

    # def process_emoticons(self, text, emo_option):
    #     if emo_option != _OPTION_NONE:
    #             text = self.filterEmoticons_plain(text, option=emo_option)
    #     return text

    # def loadEmoticons(self, emotionFile, unicodeEmoticonFile):
    #     logger.info("loading emoticons... " + emotionFile)
    #     self.loadUnicodeEmoticons(unicodeEmoticonFile)
    #     self.hshEmoticons = {}
    #     self.reEmoticons = []
        
    #     with io.open(emotionFile, encoding='utf8') as f:
    #         for line in f.readlines():
    #             line = line.strip()
    #             if line == "":
    #                 continue
    #             if line.startswith("#"):
    #                 continue
    #             # FORMATO:<TAG>
    #             # Emoción<TAG>Emoticon<TAG>texto de emocion
    #             # POSITIVO= 1
    #             # NEUTRAL = 2
    #             # INDIFERENTE = 3
    #             # NEGATIVO= 4
    #             # DESCONOCIDO= 99
    #             tag, emoticon, text = line.split(r"<TAG>")
    #             # print (tag, emoticon, text)
    #             if int(tag) == 1:
    #                 tag = _sPOSITIVE_TAG
    #             #   
    #             elif (int(tag) == 2 or
    #                   int(tag) == 3 or
    #                   int(tag) == 99):
    #                 tag = _sNEUTRAL_TAG
    #             elif int(tag) == 4:
    #                 tag = _sNEGATIVE_TAG
    #             emoticon = self.escapeEmoticons(emoticon)
    #             emoticon = emoticon.strip().lower()
    #             p = re.compile(emoticon, flags=re.I)
    #             self.reEmoticons.append(p)
    #             self.hshEmoticons[emoticon] = [tag.lower(), emoticon, text.lower()]
            
    # def loadUnicodeEmoticons(self, fileName):
    #     logger.info("loading unicode emoticons... " + fileName)
    #     self.hshUnicodeEmoticons = {}
    #     self.reUnicodeEmoticons = []
    #     with io.open(fileName, encoding='utf8') as f:
    #         for line in f.readlines():
    #             line = line.strip()
    #             if line == "":
    #                 continue
    #             if line.startswith("#"):
    #                 continue
    #             # FORMATO:
    #             # UnicodeEmoticon|texto de emocion|clase o texto de emoción
    #             unicodeEmot, text, emotion = line.split(r"|")
    #             emoticon = eval('u\"' + unicodeEmot.strip() + '\"')
    #             if emotion.startswith(_sPOSITIVE_EMOTICON):
    #                 emotion = _sPOSITIVE_TAG
    #             elif emotion.startswith(_sNEGATIVE_EMOTICON):
    #                 emotion = _sNEGATIVE_TAG
    #             elif emotion.startswith(_sNEUTRAL_EMOTICON):
    #                 emotion = _sNEUTRAL_TAG
    #             emoticon = self.escapeText(emoticon)
    #             p = re.compile(emoticon, flags=re.I)
    #             self.reUnicodeEmoticons.append(p)
    #             self.hshUnicodeEmoticons[emoticon] = [emotion.strip(), emoticon, text]
    #             # logger.info(emoticon + "=>" + emotion.strip() )

    # def filterEmoticons_plain(self, text, option):
    #     text = self.filterUnicodeEmoticons(text, option)
    #     for regex in self.reEmoticons:
    #         if regex.search(text):
    #             if option == _OPTION_DELETE:
    #                 pattern = " "
    #             else:
    #                 tags = self.hshEmoticons[regex.pattern]
    #                 if option == _OPTION_USE:
    #                     # separate the emoticon
    #                     pattern = " " + tags[1] + " "
    #                 else:
    #                     # group the emotion under label: positive, negative, neutral
    #                     pattern = " " + tags[0] + " "
    #             text = regex.sub(pattern, text)
    #     return text

    # def filterUnicodeEmoticons(self, text, option):
    #     """
    #     Groups emoticons defined by tags: _positivo, _negativo, _neutro,
    #     source: http://unicode.org/emoji/charts/full-emoji-list.html
    #     """
    #     for regex in self.reUnicodeEmoticons:
    #         if regex.search(text):
    #             if option == _OPTION_DELETE:
    #                 pattern = " "
    #             else:
    #                 tags = self.hshUnicodeEmoticons[regex.pattern]
    #                 if option == _OPTION_USE:
    #                     # separate the emoticon
    #                     pattern = " " + tags[1] + " "
    #                 else:
    #                     # group the emotion under label: positive, negative, neutral
    #                     pattern = " " + tags[0] + " "
    #             text = regex.sub(pattern, text)
    #     return text

    def escapeText(self, text):
        """
        Aplica el caracter de escape a los símbolos especiales para expresiones regulares

        """
        text = re.sub(r"\/", r"\/", text)
        text = re.sub(r"\\", r"\\\\", text)
        text = re.sub("\(", "\(", text)
        text = re.sub("\)", "\)", text)
        text = re.sub("\[", "\[", text)
        text = re.sub("\]", "\]", text)
        text = re.sub("\.", "\.", text)
        text = re.sub("\*", "\*", text)
        text = re.sub("\-", "\\-", text)
        text = re.sub("\+", "\\+", text)
        text = re.sub("\?", "\\?", text)
        text = re.sub(r"\^", r"\^", text)
        text = re.sub(r"\|", r"\|", text)
        text = re.sub(r"\*", r"\*", text)
        return text

    # def escapeEmoticons(self, text):
    #     """

    #     Aplica el caracter de escape a los símbolos especiales para expresiones regulares

    #     """
    #     if re.search(r"\\[swdb]", text, flags=re.I):
    #         return text
    #     text = re.sub(r"\\", "\\\\", text)
    #     text = re.sub(r"\\", r"\\\\", text)
    #     text = re.sub("\(", "\(", text)
    #     text = re.sub("\)", "\)", text)
    #     text = re.sub("\[", "\[", text)
    #     text = re.sub("\]", "\]", text)
    #     text = re.sub("\.", "\.", text)
    #     text = re.sub("\*", "\*", text)
    #     text = re.sub("\-", "\\-", text)
    #     text = re.sub("\+", "\\+", text)
    #     text = re.sub("\?", "\\?", text)
    #     text = re.sub(r"\^", r"\^", text)
    #     text = re.sub(r"\|", r"\|", text)
    #     return text

