import textstat
import re
import spacy
import pandas
from spacy.lang.en.stop_words import STOP_WORDS
NLP = spacy.load('en_core_web_sm')
MAX_CHARS = 40000
import mwparserfromhell as mw
import pandas as pd


# Readability Scores
def readibility_feats(text):
    feats  = {}
    feats["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
    feats["smog_index"]=textstat.smog_index(text)
    feats["flesch_kincaid_grade"]=textstat.flesch_kincaid_grade(text)
    feats["coleman_liau_index"]= textstat.coleman_liau_index(text)
    feats["automated_readability_index"]=textstat.automated_readability_index(text)
    feats["dale_chall_readability_score"]=textstat.dale_chall_readability_score(text)
    feats["difficult_words"]=textstat.difficult_words(text)
    feats["linsear_write_formula"]=textstat.linsear_write_formula(text)
    feats["gunning_fog"]=textstat.gunning_fog(text)

    return pd.Series(feats)


# Article length in Bytes

def article_to_bytes(s):
    bites = bytes(s,'utf-8')
    return len(bites)


# Number of references
def references(text):
    psr_obj = mw.parse(text)
    count = 0
    templates = psr_obj.filter_templates()
    for temp in templates:
        if "cite" in temp.name:
            count = count+1
    return count




# Number of links to other Wikipedia pages
def in_links(text):
    parser_obj = mw.parse(text)
    wikilinks = parser_obj.filter_wikilinks()
    count = 0
    for link in wikilinks:
        if "Category" not in link.title and "File" not in link.title:
            count=count+1
    return count




# Number of non citation templates
def num_templates(text):
    parser_obj = mw.parse(text)
    citations = references(parser_obj)
    templates = parser_obj.filter_templates()
    return len(templates) - citations - infobox(parser_obj)



# Number of categories linked to the text
def num_categories(text):
    parser_obj = mw.parse(text)
    wikilinks = parser_obj.filter_wikilinks()
    count = 0
    for link in wikilinks:
        if "Category" in link.title:
            count=count+1
    return count




# Number of images / length of the article
def image_by_article_len(txt):
    parser_obj = mw.parse(txt)
    wikilinks = parser_obj.filter_wikilinks()
    count = 0
    for link in wikilinks:
        if "File" in link.title:
            count=count+1
    article_len = article_to_bytes(txt)
    return count/(1.0*article_len)



# Information noise score
def information_to_noise(comment):
    comment = re.sub(r"[\*\"\n\\\+\-\/\=\(\):\[\]\|\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    words_all = [x.text for x in NLP.tokenizer(comment) if x.text != " "]
    words_rem = set([x.lemma_ for x in NLP.tokenizer(comment) if not x.is_stop])
    return len(words_rem)/(len(words_all)*1.0)





# Article having infobox
def infobox(text):
    psr_obj = mw.parse(text)
    count = 0
    templates = psr_obj.filter_templates()
    for temp in templates:
        if "Infobox" in temp.name:
            count = count+1
    return count



# Number of level 2 Headings
def level2head(text):
    psr_obj = mw.parse(text)
    count = 0 
    headings = psr_obj.filter_headings()
    for head in headings:
        if head.level == 2:
            count=count+1

    return count

# Number of level 3+ headings

def level3head(text):
    psr_obj = mw.parse(text)
    count = 0
    headings = psr_obj.filter_headings()
    for head in headings:
        if head.level != 2:
            count=count+1

    return count
