# standard library
import os
import csv
import re
import json
import string

# 3rd party packages
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tag import CRFTagger
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from num2words import num2words
from string import punctuation

def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b

def detect_special_char(pass_string):
    regex= re.compile("[^.,]") 
    if(regex.search(pass_string) == None): 
        res = False
    else: 
        res = True
    return(res)

class FileReader:
    """FileReader class."""
    def __init__(self, fname):
        self.fname = fname
    
    def check_type(self):
        """FileReader.check_type() should return the file extension if only the file is exists."""
        file_exists = os.path.exists(self.fname)
        if file_exists:
            print(f"The file {self.fname} exists.")
            ext = os.path.splitext(self.fname)[-1].lower()
            return ext
        else:
            print(f'The file {self.fname} does not exist.')
        
    def read_tsv(self):
        """FileReader.read_tsv() should return the data in a DataFrame."""
        try:
            print(f"File reading is in progress...")
            df = pd.read_csv(self.fname,
                            sep="\t",
                            names=["token", "ne"],
                            skip_blank_lines=False,
                            quoting=csv.QUOTE_NONE,
                            encoding='utf-8')
            
            list_tmp = []
            no = 0
            for row in df.itertuples():
                if pd.isnull(row.token):
                    list_tmp.append(np.nan)
                    no+=1
                else:
                    list_tmp.append(no)
            df["sentence"] = list_tmp
            df = df.dropna(thresh=2).reset_index(drop=True)
            df[["sentence"]] = df[["sentence"]].astype(int)
            print(f"File reading is complete.")

            return df
        except Exception as e:
            print(e)

class Preprocessing:
    """Preprocessing class."""
    def __init__(self, df):
        self.df = df
    
    def _expand_contractions(self, s, contractions_dict, contractions_re):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)

    def expand_contractions(self):
        """
        Contraction is the shortened form of a word.
        Preprocessing.expand_contractions() should return the data in a DataFrame.
        """
        try:
            print(f"Expand contractions is in progress...")
            # load Indonesian contraction dictionary
            with open("data\indo_contraction_dict.json") as file:
                contraction_dict = json.load(file)
            df_copy = self.df.copy()
            df_copy.replace({"token": contraction_dict},inplace=True)
            self.df = df_copy
            print(f"Expand contractions is complete.")
            return self.df

        except Exception as e:
            print(e)
            return self.df

    def hyphen_comma_splitting(self):
        """
        Split hyphen and comma punctuation into separate tokens.
        Preprocessing.hyphen_comma_splitting() should return the data in a DataFrame.
        """
        try:
            print(f"Hyphen and comma splitting are in progress...")
            df_copy = self.df.copy()
            df_copy["token"] = df_copy["token"].apply(lambda x:re.split("([-])", x) if isfloat(x) == False else x)
            df_copy = df_copy.explode('token').reset_index(drop=True)
            self.df = df_copy
            print(f"Hyphen and comma splitting are complete.")
            return self.df
        except Exception as e:
            print(e)
            return self.df
    
    def lowercasing(self):
        """Preprocessing.lowercasing() should return the data in a DataFrame."""
        try:
            print(f"Lowercasing is in progress...")
            df_copy = self.df.copy()
            df_copy["token"] = df_copy["token"].apply(lambda x: x.lower())
            self.df = df_copy
            print(f"Lowercasing is complete.")
            return self.df
        except Exception as e:
            print(e)
            return self.df
    
    def stemming(self):
        """
        Stemming based on PySastrawi by Hanif Amal Robbani.
        Preprocessing.stemming() should return the data in a DataFrame.
        """
        try:
            print(f"Stemming is in progress...")
            df_copy = self.df.copy()
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            df_copy["token"] = df_copy["token"].apply(lambda x:stemmer.stem(x) if any(p in x for p in punctuation) == False else x)
            self.df = df_copy
            print(f"Stemming is complete.")
            return self.df
        except Exception as e:
            print(e)
            return self.df
    
    def number2words(self):
        """
        Convert numbers to words based on num2words by Taro Ogawa.
        Preprocessing.number2words() should return the data in a DataFrame.
        """
        try:
            print(f"Number to words conversion is in progress...")
            df_copy = self.df.copy()
            for idx, val in df_copy["token"].items():
                if val != ".": 
                    t = val.replace(".", "").replace(",", ".")
                    t = re.split("(\d+)", t)
                    t = list(filter(None, t))
                    if len(t) > 1:
                        lt = []
                        for i in t:
                            if i.isdigit() == True:
                                y = num2words(int(i), lang='id')
                                y = re.split(" ", y)
                                lt.extend(y)
                            else:
                                lt.append(i)
                        df_copy.at[idx, "token"] = (lt)
                    else:
                        df_copy.at[idx, "token"] = val
            df_copy = df_copy.explode("token").reset_index(drop=True)
            self.df = df_copy
            print(f"Number to words conversion is complete.")
            return self.df
        except Exception as e:
            print(e)
            return self.df

class DatasetPreparator:
    """DatasetPreparator class."""
    def __init__(self, df, pt):
        self.df = df
        self.pt = pt
    
    def add_post(self):
        list_post = []

        ct = CRFTagger()
        ct.set_model_file(self.pt)
        series_tmp = self.df.groupby("sentence")["token"].apply(list)
        df_tmp = series_tmp.to_frame(name="token")
        for val in tqdm(df_tmp.itertuples(), total=df_tmp.shape[0]):
            post_tag = ct.tag_sents([val.token])
            list_post += [e[1] for e in post_tag[0]]
        self.df["post"] = list_post

    def check_post(self):
        """DatasetPreparator.check_post() should return True only if pos-tag column exist."""
        if ("post" or "postag") in self.df.columns:
            print(f"\nThe pos tag column exists.")
        else:
            print(f"\nThe pos tag column does not exist.")
            print(f"Pos tag column creation is in progress...")
            self.add_post()
            self.df = self.df[["sentence", "token", "post", "ne"]]
            print(f"Pos tag column creation is complete.")
            return self.df
            
    def add_ne(self):
        print(f"Represent named entity with the Begin, Inside, Outside (BIO) notation is in progress...")
        dfd = self.df.copy()
        bio_tag = []
        prev_tag = "O"
        for _, tag in self.df["ne"].iteritems():
            if tag == "O": #O
                bio_tag.append((tag))
                prev_tag = tag
                continue
            if tag != "O" and prev_tag == "O": # Begin NE
                bio_tag.append(("B-"+tag))
                prev_tag = tag
            elif prev_tag != "O" and prev_tag == tag: # Inside NE
                bio_tag.append(("I-"+tag))
                prev_tag = tag
            elif prev_tag != "O" and prev_tag != tag: # nearby NE
                bio_tag.append(("B-"+tag))
                prev_tag = tag
        
        dfd["ne"] = bio_tag
        self.df = dfd
        print(f"Represent named entity with the Begin, Inside, Outside (BIO) notation is complete.")
        return self.df

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['token'].values.tolist(), 
                                                           s['post'].values.tolist(), 
                                                           s['ne'].values.tolist())]
        self.grouped = self.data.groupby('sentence').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

def countvowels(string):
    num_vowels=0
    for char in string:
        if char in "aeiouAEIOU":
           num_vowels = num_vowels+1
    return num_vowels

# Feature set
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word': word,
        'len(word)': len(word),
        'nvowels': countvowels(word),
        'word[:4]': word[:4],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word.lower()': word.lower(),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word.lower()),
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word': word1,
            '-1:len(word)': len(word1),
            '-1:nvowels': countvowels(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word1.lower()),
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.update({
            '-2:word': word2,
            '-2:len(word)': len(word2),
            '-2:nvowels': countvowels(word2),
            '-2:word.lower()': word2.lower(),
            '-2:word[:3]': word2[:3],
            '-2:word[:2]': word2[:2],
            '-2:word[-3:]': word2[-3:],
            '-2:word[-2:]': word2[-2:],
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.ispunctuation': (word2 in string.punctuation),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
        })

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word': word1,
            '+1:len(word)': len(word1),
            '+1:nvowels': countvowels(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })

    else:
        features['EOS'] = True
    if i < len(sent) - 2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.update({
            '+2:word': word2,
            '+2:len(word)': len(word2),
            '+2:nvowels': countvowels(word2),
            '+2:word.lower()': word2.lower(),
            '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word2.lower()),
            '+2:word[:3]': word2[:3],
            '+2:word[:2]': word2[:2],
            '+2:word[-3:]': word2[-3:],
            '+2:word[-2:]': word2[-2:],
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.ispunctuation': (word2 in string.punctuation),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
        })

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for _, _, label in sent]
    
def sent2tokens(sent):
    return [token for token, _, _ in sent]