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
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
# from stop_words import get_stop_words as arabic_get_stop_words
# import nltk
import numpy as np
import json
# import sys
from .text_preprocessing import TextPreprocessing,  _OPTION_NONE

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PATH = os.path.dirname(__file__)

_ENGLISH = "english"
_SPANISH = "spanish"
_ITALIAN = "italian"
_GERMAN = "german"
_ARABIC = "arabic"
_AFFECTIVE_FILE = "affective.words.json"


POS_EMOTION = "positive"
NEG_EMOTION = "negative"


class ThumbsUpDown(object):
    def __init__(self, file_name=None, lang=_SPANISH, stemming=False):
        """
        Initializes the parameters for specific language
        """
        self._text = os.getenv('TEXT', default='text')
        self.languages = [_SPANISH, _ENGLISH, _ARABIC]
        self.lang = lang
        if self.lang not in self.languages:
            raise ("Language not supported: " + lang)
        self.text_model = TextPreprocessing(lang=self.lang)
        self.stem = stemming

        if self.lang == _ENGLISH:
            self.stemmer = PorterStemmer()
        elif self.lang == _ARABIC:
            from nltk.stem.isri import ISRIStemmer
            self.stemmer = ISRIStemmer()
        else:
            self.stemmer = SnowballStemmer(self.lang)

        self.emotions = {}
        self.stem_emotions = {}
        if file_name is not None:
            emo_file = file_name
        else:
            if self.lang in [_ENGLISH, _ITALIAN, _GERMAN, _ARABIC]:
                emo_file = self.lang[:2] + "." + _AFFECTIVE_FILE
            elif self.lang == _SPANISH:
                emo_file = "es." + _AFFECTIVE_FILE
            emo_file = os.path.join(PATH, 'data', emo_file)
        self.load_emotions(emo_file)

    def get_text(self, text):
        return text[self._text]

    def stemming(self, text, option=True):
        """
        Applies the stemming process to `text` parameter
        """
        tokens = re.split(r"\s+", text)
        if not option:
            return tokens
        
        t = []
        for tok in tokens:
            if re.search(r"^(@|#|_|~)", tok, flags=re.I):
                # t.append(tok)
                continue
            else:
                try:
                    if len(tok) < 2:
                        t.append(tok)
                    else:
                            t.append(self.stemmer.stem(tok))
                except:
                    continue
        return t

    def generate_words(self, word):
        if self.lang == _SPANISH:
            return self.generate_sp_words(word)
        if self.lang == _ENGLISH:
            return self.generate_en_words(word)
        if self.lang == _ARABIC:
            return self.generate_en_words(word)

    def generate_ar_words(self, word):
        return [word]

    def generate_en_words(self, word):
        return [word]

    def generate_sp_words(self, word):
        gen_words = [word]
        if len(word.split(" ")) > 1:
            return gen_words

        if re.search(r"[aeiou][nstdv]o$", word, flags=re.I):
            gen_words.append(re.sub(r"o$", "a", word, flags=re.I))
            gen_words.append(re.sub(r"o$", "os", word, flags=re.I))
            gen_words.append(re.sub(r"o$", "as", word, flags=re.I))
            return gen_words

        if re.search(r"ble$", word, flags=re.I):
            gen_words.append(re.sub(r"ble$", "bles", word, flags=re.I))
            return gen_words

        if re.search(r"l$", word, flags=re.I):
            gen_words.append(re.sub(r"l$", "les", word, flags=re.I))
            return gen_words

        if re.search(r"z$", word, flags=re.I):
            gen_words.append(re.sub(r"z$", "ces", word, flags=re.I))
            return gen_words

        if re.search(r"ente$", word, flags=re.I):
            gen_words.append(re.sub(r"ente$", "entes", word, flags=re.I))
            return gen_words

        if re.search(r"ion$", word, flags=re.I):
            gen_words.append(re.sub(r"n$", "nes", word, flags=re.I))
            return gen_words

        if re.search(r"on$", word, flags=re.I):
            gen_words.append(re.sub(r"n$", "es", word, flags=re.I))
            gen_words.append(re.sub(r"n$", "na", word, flags=re.I))
            gen_words.append(re.sub(r"n$", "nas", word, flags=re.I))
            return gen_words
 
        if re.search(r"o$", word, flags=re.I):
            gen_words.append(re.sub(r"o$", "os", word, flags=re.I))
            gen_words.append(re.sub(r"o$", "a", word, flags=re.I))
            gen_words.append(re.sub(r"o$", "as", word, flags=re.I))
            return gen_words

        if re.search(r"a$", word, flags=re.I):
            gen_words.append(re.sub(r"a$", "as", word, flags=re.I))

        return gen_words

    def load_emotions(self, file_name):
        logger.info("loading emotions from file... {} ".format(file_name))
        if not os.path.exists(file_name):
            logger.error("Error file doesn't exist => " + file_name)
            return

        emotions_file = io.open(file_name, "r", encoding="utf8")
        for line in emotions_file:
            emot = json.loads(line)
            emotion_key = 'emotion'
            words_key = 'words'
            if not (emot.get(emotion_key) and emot.get(words_key)):
                raise Exception("Affective keys could not find in file: " + file_name + " ==> " + emotion_key + " and " + words_key)
                 
            if emot[emotion_key] not in self.emotions:
                self.emotions[emot[emotion_key]] = []
                self.stem_emotions[emot[emotion_key]] = []

            for w in emot[words_key]:
                w = self.text_model.norm_chars(text=w, del_diac=True, del_dup=False,  del_punc=False)
                for e_word in self.generate_words(w):
                    self.add_affective_word(emot[emotion_key], e_word)

        # for e in self.stem_emotions:
        # logger.debug(self.stem_emotions[e])

        # pp.pprint(self.emotions)
        # pp.pprint(self.stem_emotions)
        # pprint("words POS: " + "+".join(str(x) for x in self.stem_emotions['positive']))
        # pprint("words NEG: " + "-".join(str(x) for x in self.stem_emotions['negative']))

    def add_affective_word(self, emotion,  word):
        w = self.text_model.norm_chars(word, del_diac=True, del_dup=False, del_punc=False)
        self.emotions[emotion].append(w)
        if self.stem:
            self.stem_emotions[emotion].append(self.stemmer.stem(w))
        else:
            self.stem_emotions[emotion].append(w)

    def count_emotions(self, text):
        # check whether tweet has sentiment with stemmed words
        if self.stem:
            toks = self.stemming(text, self.stem)
        else:
            toks = re.split(r"\s+", text)
        pos = 0
        neg = 0
        for t in toks:
            if t in self.stem_emotions['positive']:
                pos += 1
                # logger.debug(t)
            if t in self.stem_emotions['negative']:
                neg += 1
                # logger.debug(t)
                
        # logger.debug(" counting emotions")
        # logger.debug("text => {}".format(text))
        # logger.debug("POS:\t" + str(pos))
        # logger.debug("NEG:\t" + str(neg))
        # logger.debug("text stem: " + ",".join(str(x) for x in toks))
        return pos, neg

    # def process_file(self, file_name):
    #     out_file = open(file_name + ".up_down", "w", encoding="utf-8")
    #     for data in self.text_model.read_json(file_name):
    #         text = data['text']
    #         text = self.text_model.text_transform(text,
    #                                               remove_stopwords=False,
    #                                               remove_diacritic=True,
    #                                               emo_option=_OPTION_NONE,
    #                                               url=_OPTION_NONE)
    #         pos, neg = self.count_emotions(text)
    #         data['up_down'] = [pos, neg]
    #         out_file.write(json.dumps(data) + "\n")

    def __getitem__(self, text):
        if isinstance(text, dict):
            text = self.get_text(text)
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        text = self.text_model.text_transform(text,
                                              remove_stopwords=False,
                                              remove_diacritic=True,
                                              emo_option=_OPTION_NONE,
                                              url=_OPTION_NONE)
        return self.count_emotions(text)
