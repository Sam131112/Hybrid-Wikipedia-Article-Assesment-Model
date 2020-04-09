import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
NLP = spacy.load('en')
MAX_CHARS = 40000
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
NLP = spacy.load('en')
MAX_CHARS = 40000


def spl_tokenizer(comment):
    comment = re.sub(r"[\*\"\n\\\+\-\/\=\(\):\[\]\|\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    words_all = [x.text for x in NLP.tokenizer(comment) if x.text != " "]
    words_rem = set([x.lemma_ for x in NLP.tokenizer(comment) if not x.is_stop])
    return (len(words_rem),(len(words_all)*1.0))

