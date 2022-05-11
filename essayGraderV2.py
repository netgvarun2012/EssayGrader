import dill as pickle
import spacy
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import textwrap # utility to wrap the text
import pandas as pd
import re
import json
import os
from textblob import TextBlob
from awlify import awlify
from lexical_diversity import lex_div as ld
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet, stopwords
from os.path import abspath, join, dirname
from nltk.tokenize import word_tokenize
import string
from nltk.tag import pos_tag
from collections import defaultdict
import numpy as np


# Load spacy's vocab
nlp = spacy.load('en_core_web_sm')

# Load the saved Essay Grading model
with open('proof_reading/EssayGraderV2.pkl','rb') as fin:
    grading_model = pickle.load(fin) 

#SentenceCount feature
def count_sentences(essay):
  doc = nlp(essay)
  doc_sents = [sent for sent in doc.sents]
  return len(doc_sents)

#WordCount
def count_all_words(essay):
  doc = nlp(essay)
  return(len(doc))

#VerbCount,NounCount,AdjCount,AdverbCount,PronounCount,PunctCount
def count_feature(postag,essay,wordcount,roundval):
  doc=nlp(essay)
  pos_counts = doc.count_by(spacy.attrs.POS)
  for k,v in sorted(pos_counts.items()):
    if doc.vocab[k] == postag:
      return round(v/wordcount,roundval)
  
#NumberofActiveVoice
def check_passive_voice(inputEssay):
  doc = nlp(inputEssay)
  dep_list = []
  passive_list = []
  sents = list(doc.sents)

  for sents in doc.sents:
    dep_list = []
    for token in sents:# Tokenize the sentence into words/tokens
      dep_list.append(token.dep_)

    if ('nsubjpass') in dep_list or ('auxpass') in dep_list:
      passive_list.append(sents)

  return (round((len(sents)-len(passive_list))/len(sents),2)) # returning active voice sentences density


#NumComplex,#NumCompund,#NumSimple

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]

def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        #print(rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs

def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False

def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" or tok.pos_ == "AUX"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
          svs.append((subs[-1].orth_, "!" + v.orth_ if verbNegated else v.orth_))
            # for sub in subs:
            #     svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs

def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    #print(subs)
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated

def testSVOs(input_sent):
    tok = nlp(input_sent)
    svs = findSVs(tok)
    return (len(svs))

def getCountTypeSent(essay_text):
  s_conj = ['after','although','as','because','before','even though','if','since','though','unless','until','when','whenever','whereas','wherever','while','where','that','who','no matter']
  exceptions = ['as soon as']

  complex_comp_list=[]
  complex_list=[]
  compound_list=[]
  simple_list=[]
  dep_list = []
  ret_dict = {}
  docu = nlp(essay_text)
  num_sent = len([sent for sent in docu.sents]) # Counting the number of sentences
  simple_dict_inner = {}
  complex_comp_dict_inner = {}
  complex_dict_inner = {}
  compound_dict_inner = {}

  for sents in docu.sents:    
    for s in exceptions:
      res = re.search(r"\b" + re.escape(s) + r"\b", sents.text.lower())
      if res:
        break
    
    for s in s_conj:
      res_sj = re.search(r"\b" + re.escape(s) + r"\b", sents.text.lower())
      if res_sj:
        break

    if testSVOs(sents.text) == 1 and not res_sj:#any(tok in tok_list for tok in s_conj):                    # Simple sentence as only 1 subject verb pair and no subord conj
      simple_dict_inner["startIndex"] = sents.start_char
      simple_dict_inner["endIndex"]   = sents.end_char
      simple_dict_inner["text"]       = sents.text 
      simple_list.append(simple_dict_inner.copy()) 
    elif res_sj and not res:#any(tok in tok_list for tok in s_conj) and not res:                            # Complex sentence as more than 1 subject verb pair and subord conjunction.
      complex_dict_inner["startIndex"] = sents.start_char
      complex_dict_inner["endIndex"]   = sents.end_char
      complex_dict_inner["text"]       = sents.text 
      complex_list.append(complex_dict_inner.copy())     
    else:
      compound_dict_inner["startIndex"] = sents.start_char
      compound_dict_inner["endIndex"]   = sents.end_char
      compound_dict_inner["text"]       = sents.text 
      compound_list.append(compound_dict_inner.copy()) 

  return (len(complex_list)/num_sent,len(compound_list)/num_sent,len(simple_list)/num_sent)

#FleschScore
def get_flesch_score(essay_text):
  from readability import Readability
  r = Readability(essay_text)
  fk = r.flesch_kincaid()
  readability_score = fk.score
  return round(readability_score,2)


#UniqWordDensity
def non_punct_words(essay):
  mylist = re.findall(r'[^!?.,-;]+',essay) # To remove punctuations
  str1 = ''.join(mylist)
  return str1

def uniq_word_count(essay):
  out = []
  seen = set()
  strng = non_punct_words(essay)
  doc = nlp(strng)
  for word in doc:
    if word.text not in seen:
      if word.text != ' ':
        out.append(word)
        seen.add(word.text)
  return (len(seen)/len(doc)) 


#TellDensity
def showntell_essay(InputEssay):
  doc  = nlp(InputEssay)
  mental_tells = ['loved','realized','thought','hoped','considered','wondered','prayed','knew','saw','watched','heard','felt','could','see','seemed','appeared',
  'looked','believed','reflected','disgusted','feared','show','noticed','smelled','wonder','walked','come','hate','decided','wished','feel','see','smell',
  'fell']

  emotional_tells = ['adoration','agitation','amazement','amusement','anger','anguish','annoyance','anticipation','anxiety','confidence','conflicted','confusion',
  'contempt','curiosity','defeat','defensiveness','denial','depression','desire','desperation','determination','disappointment','disbelief','disgust','doubt','dread',
  'eagerness','elation','embarrassment','envy','excitement','fear','frustration','gratitude','guilt','happiness','hatred','hopefulness','humiliation','hurt','impatience',
  'indifference','insecurity','irritation','jealousy','loneliness','love','nervousness','nostalgia','overwhelmed','paranoia','peacefulness','pride','rage','regret','relief',
  'reluctance','remorse','resentment','resignation','sadness','satisfaction','scorn','shame','skepticism','smugness','somberness','surprise','shock','suspicion','sympathy',
  'terror','uncertainty','unease','wariness','worry']

  motivational_tells = ['decided','because','tried','when']

  emotional_adjectives = ['frustated', 'happy', 'tall', 'angry', 'sad', 'hungry', 'excited', 'embarrased', 'bright', 'shocked', 'hot', 'beautiful', 
                          'afraid', 'cold', 'interesting', 'confused', 'sweet', 'different', 'scared', 'mournful', 'furious', 'overwhelmed', 'stressed', 
                          'unique', 'overjoyed', 'scarier', 'tired', 'shy', 'giddy', 'anxious','chilly','friendly','ghastly','ghostly','holy','kingly',
                          'knightly','lonely','lovely','orderly','prickly','queenly','surly','ugly','worldly','wrinkly']

  adverbs_avoid = ['very', 'really', 'spectacularly','already' 'abruptly', 'absently', 'absentmindedly', 'accusingly', 'actually', 'adversely', 'affectionately', 
                  'amazingly', 'angrily', 'anxiously', 'arrogantly', 'bashfully', 'beautifully', 'boldly', 'bravely', 'breathlessly', 'brightly', 'briskly', 'broadly', 
                  'calmly', 'carefully', 'carelessly', 'certainly', 'cheaply', 'cheerfully', 'cleanly', 'clearly', 'cleverly', 'closely', 'clumsily', 'coaxingly', 'commonly', 
                  'compassionately', 'conspicuously', 'continually', 'coolly', 'correctly', 'crisply', 'crossly', 'curiously', 'daintily', 'dangerously', 'darkly', 'dearly', 
                  'deceivingly', 'delicately', 'delightfully', 'desperately', 'determinedly', 'diligently', 'disgustingly', 'distinctly', 'doggedly', 'dreamily', 'emptily', 
                  'energetically', 'enormously', 'enticingly', 'entirely', 'enviously', 'especially', 'evenly', 'exactly', 'excitedly', 'exclusively', 'expertly', 'extremely', 
                  'fairly', 'faithfully', 'famously', 'fearlessly', 'ferociously', 'fervently', 'finally', 'foolishly', 'fortunately', 'frankly', 'frantically', 'freely', 
                  'frenetically', 'frightfully', 'fully', 'furiously', 'generally', 'generously', 'gently', 'gleefully', 'gratefully', 'greatly', 'greedily', 'grumpily', 
                  'guiltily', 'happily', 'harshly', 'hatefully', 'heartily', 'heavily', 'helpfully', 'helplessly', 'highly', 'hopelessly', 'hungrily', 'immediately', 'importantly', 
                  'impulsively', 'inadvertently', 'increasingly', 'incredibly', 'innocently', 'instantly', 'intensely', 'intently', 'inwardly', 'jokingly', 'kindly', 'knowingly', 
                  'lawfully', 'lightly', 'likely', 'longingly', 'loudly', 'madly', 'marvelously', 'meaningfully', 'mechanically', 'meekly', 'mentally', 'messily', 'mindfully', 'miserably', 
                  'mockingly', 'mostly', 'mysteriously', 'naturally', 'nearly', 'neatly', 'negatively', 'nervously', 'nicely', 'obviously', 'occasionally', 'oddly', 'openly', 'outwardly', 
                  'partially', 'passionately', 'patiently', 'perfectly', 'perpetually', 'playfully', 'pleasantly', 'pleasingly', 'politely', 'poorly', 'positively', 'potentially', 'powerfully', 
                  'professionally', 'properly', 'proudly', 'quaveringly', 'queerly', 'quickly', 'quietly', 'quintessentially', 'rapidly', 'rapturously', 'ravenously', 'readily', 'reassuringly', 
                  'regretfully', 'reluctantly', 'reproachfully', 'restfully', 'righteously', 'rightfully', 'rigidly', 'rudely', 'sadly', 'safely', 'scarcely', 'searchingly', 'sedately', 
                  'seemingly', 'selfishly', 'separately', 'seriously', 'sharply', 'sheepishly', 'sleepily', 'slowly', 'slyly', 'softly', 'solidly', 'speedily', 'sternly', 'stingily', 'strictly', 
                  'stubbornly', 'successfully', 'superstitiously', 'surprisingly', 'suspiciously', 'sympathetically', 'tenderly', 'terribly', 'thankfully', 'thoroughly', 'thoughtfully', 'tightly', 
                  'totally', 'tremendously', 'triumphantly', 'truly', 'truthfully', 'understandably', 'unfairly', 'unfortunately', 'unhappily', 'unwillingly', 'urgently', 'usually', 'utterly', 'vastly', 
                  'venomously', 'viciously', 'violently', 'warmly', 'wearily wholly', 'wildly', 'wilfully', 'wisely', 'wonderfully', 'wonderingly', 'worriedly']

  aux_tell_list = []
  ment_tell_list = []
  det_tell_list = []
  motiv_tell_list = []
  emot_adj_tell_list  = []
  adv_tell_list  = []
  adj_tell_list = []
  sent_level_dict = {}
  count = 0

  output_dict = {}

  adj_dict_inner = {}
  adv_dict_inner = {}
  aux_dict_inner = {}
  emot_dict_inner = {}
  det_dict_inner = {}
  ment_dict_inner = {}
  motiv_dict_inner = {}

  doc_sents = [sent for sent in doc.sents]
  len_tot_sents = len(doc_sents)
  for sents in doc.sents:
    count = 0
    for ix,token in enumerate(sents):
      tok_pos=token.idx
      try:
        if token.is_sent_start:    # checking for first token
          if sents[ix].tag_ == 'PRP' or sents[ix].tag_ == 'PRP$' or sents[ix].tag_ == 'NN' :        # check for 'PRP' (Pronoun Personal) specifically with  'I','We','They','He','She'. 'You'.,check for 'PRP$' (Pronoun Possessive) my, our, your, his, her, its, and their,check for 'NN' (noun, singular or mass) i.e. 'Non-specific Nouns'
            if sents[ix + 1].pos_ == 'AUX' or sents[ix + 1].pos_ == 'MD':                           # check for the token to the right for 'AUX' and 'MD'.
              tok = (sents[ix:ix+2])     # getting the right token as well
              right_pos = sents[ix+2].idx-1 # getting the right 2 token's starting character offset.                                                    
              aux_dict_inner["startIndex"] = tok_pos
              aux_dict_inner["endIndex"]   = right_pos
              aux_dict_inner["text"]       = tok
              aux_tell_list.append(aux_dict_inner.copy())          
              count +=1
              sent_level_dict[sents] = count
            elif sents[ix + 1].pos_ == 'VERB':
              if sents[ix+1].text.lower() in mental_tells or sents[ix+1].lemma_.lower() in mental_tells: # check if the token next to the first token is a 'VERB' out of 'mental tell' verbs.
                tok = (sents[ix:ix+2])                                                                  # check for 'NN' (noun, singular or mass) i.e. 'Non-specific Nouns'
                right_pos = sents[ix+2].idx-1 # getting the right 2 token's starting character offset.                                                    
                ment_dict_inner["startIndex"] = tok_pos
                ment_dict_inner["endIndex"]   = right_pos
                ment_dict_inner["text"]       = tok
                ment_tell_list.append(ment_dict_inner.copy())                
                count +=1
                sent_level_dict[sents] = count            
          elif sents[ix].pos_ == 'DET':                                                             # check if the first token is a 'determiner' 
            if sents[ix + 1].pos_ == 'ADJ' or sents[ix + 1].tag_ == 'RBS':                          # and next one is 'adverb superlative' or an 'adjective'.
                tok = (sents[ix:ix+2])
                right_pos = sents[ix+2].idx-1 # getting the right 2 token's starting character offset.                                                    
                det_dict_inner["startIndex"] = tok_pos
                det_dict_inner["endIndex"]   = right_pos
                det_dict_inner["text"]       = tok              
                det_tell_list.append(det_dict_inner.copy())
                count +=1
                sent_level_dict[sents] = count          

        if token.text.lower() in motivational_tells or token.lemma_.lower() in motivational_tells:   # check if the token is out of the 'motivational tell' list
            tok = token.text.lower()
            right_pos = sents[ix+1].idx-1 # getting the right 2 token's starting character offset.                                                    
            motiv_dict_inner["startIndex"] = tok_pos
            motiv_dict_inner["endIndex"]   = right_pos
            motiv_dict_inner["text"]       = tok       
            motiv_tell_list.append(motiv_dict_inner.copy())
            count +=1
            sent_level_dict[sents] = count        
        elif sents[ix].text.lower() == 'to' and sents[ix + 1].pos_ == 'VERB':                        # check if the word is of the form 'to [Verb]'
            tok = (sents[ix:ix+2])
            right_pos = sents[ix+2].idx-1 # getting the right 2 token's starting character offset.                                                    
            motiv_dict_inner["startIndex"] = tok_pos
            motiv_dict_inner["endIndex"]   = right_pos
            motiv_dict_inner["text"]       = tok       
            motiv_tell_list.append(motiv_dict_inner.copy())
            count +=1
            sent_level_dict[sents] = count        
        elif any(x in sents[ix].text.lower() for x in ['with','in']) and (sents[ix + 1].pos_ == 'NOUN' or sents[ix + 1].pos_ == 'PROPN') and sents[ix + 1].text.lower() in emotional_tells: # check if the word is of the form 'with [noun] or in [noun]'
            tok = (sents[ix:ix+2])
            right_pos = sents[ix+2].idx-1 # getting the right 2 token's starting character offset.  
            emot_dict_inner["startIndex"] = tok_pos
            emot_dict_inner["endIndex"]   = right_pos
            emot_dict_inner["text"]       = tok                                                         
            emot_adj_tell_list.append(emot_dict_inner.copy())
            count +=1
            sent_level_dict[sents] = count        
        elif token.pos_ == 'ADJ' and (token.text.lower() in emotional_adjectives or token.lemma_.lower() in emotional_adjectives):  # check if the word is out of the list of 'emotional' adjective word list
          tok = token.text.lower()
          right_pos = sents[ix+1].idx-1 # getting the right 2 token's starting character offset.                                                    
          adj_dict_inner["startIndex"] = tok_pos
          adj_dict_inner["endIndex"]   = right_pos
          adj_dict_inner["text"]       = tok     
          adj_tell_list.append(adj_dict_inner.copy())
          count +=1
          sent_level_dict[sents] = count      
        elif token.pos_ == 'ADV' and (token.text.lower() in adverbs_avoid or token.lemma_.lower() in adverbs_avoid):             # check if the word is out of the list of 'adverb to avoid' word list
          tok_pos=token.idx
          right_pos = sents[ix+1].idx-1 # getting the right token's starting character offset.
          tok = token.text.lower()
          adv_dict_inner["startIndex"] = tok_pos
          adv_dict_inner["endIndex"]   = right_pos
          adv_dict_inner["text"]       = tok  
          adv_tell_list.append(adv_dict_inner.copy())
          count +=1
          sent_level_dict[sents] = count 
      except:
        continue            
      
  return (len(sent_level_dict)/count_sentences(InputEssay))

#AvgChar
def avg_char_count(essay):
  mylist = re.findall(r'[^!?.,-;]+',essay) # To remove punctuations  
  str1 = ''.join(mylist)    
  docu = nlp(str1)
  len_char = []
  for token in docu:
    #print(f'{token.idx} {token.lower_} {token.pos_} {spacy.explain(token.pos_)}  {token.tag_} {spacy.explain(token.tag_)} {token.dep_} {spacy.explain(token.dep_)}')
    len_char.append(len(token))
  return round(sum(len_char)/count_all_words(essay),2)


#Syllablesperword
def count_syllables_per_word(essay):
  import syllables
  mylist = re.findall(r'[^!?.,-;]+',essay) # To remove punctuations  
  str1 = ''.join(mylist)  
  count_s = syllables.estimate(str1)
  return round(count_s/count_all_words(essay),2)

#mtld
#from lexical_diversity import lex_div as ld
def calc_mtld(essay):
  mylist = re.findall(r'[^!?.,-;]+',essay) # To remove punctuations  
  str1 = ''.join(mylist) 
  return round(ld.mtld(str1),2)

# Percentage of named entities in all entities.
def count_ner_per_ent(essay):
  ner_list = []
  doc=nlp(essay)
  for entity in doc.ents:
    ner_list.append(entity.text)
  return round(len(ner_list)/count_all_words(essay),2)

# Avg named entities per sentence.
def count_ner_per_sent(essay):
  ner_list = []
  doc=nlp(essay)
  for entity in doc.ents:
    ner_list.append(entity.text)
  return round(len(ner_list)/count_sentences(essay),2)


# CEFR Level

def categorizeText(input_text):
    """
    :Returns: List = [MainLevel, Difficulty] (some sort of language level)
    """
    if (not(isinstance(input_text, str)) or (len(input_text) <= 0)):
        dicti = {"unknown": "NOT OKAY!", "A1": "THIS!", "A2" : "IS!", "B1": "NOT!", "B2": "A!", "C1": "TEXT!", "C2": "NO!"}
        return ["NO!", "NO!", dicti]

    # normalize text with NLP
    processed_text = processText(input_text)
    
    # store words of text lowercase in list
    words: list = [item.lower() for item in processed_text.split()]

    # count frequency of word in text
    word_frequency: dict = getWordFrequency(words)

    # Dataframe, set der Worte mit Sprachniveau
    # word, level
    set_word_table = getWordLevelDataFrameForText(set(words))

    # Viewing the distribution
    verteilung = {}
    tmp_count = 0
    
    #for each word from the text, ordered by level,
    
    for lvl in ["unknown", "A1", "A2", "B1", "B2", "C1", "C2"]:
        for word in set_word_table.loc[set_word_table['level']== lvl, "word"]:
            tmp_count += word_frequency[word]
        tmp_result = tmp_count/ len(words) * 100
        verteilung[lvl] = round(tmp_result)
        tmp_count = 0
    
    #Rank based on the highest level that contains more than n different words
    n = 4
    levels, counts = np.unique(set_word_table['level'], return_counts=True)
    
    if (len(levels) > 0):
        tmp_index, = np.where(levels == "unknown") # löschen der Stellen, an denen die Werte für UNKNOWN Worte stehen, da diese kein Sprachniveau sind
        levels = np.delete(levels, tmp_index)
        counts = np.delete(counts, tmp_index)
    max_level = np.max(levels[counts > n])
    
    #Difficulty rating of unknown words, limit: m
    count_easy = 0
    count_hard = 0
    m = 6 # siehe Wolfram alpha 5.1
    
    for word in set_word_table.loc[set_word_table['level']== "unknown", "word"]:
        if len(word) > m:
            count_hard += 1
        elif len(word) <= m:
            count_easy += 1   
    
    if count_easy <= count_hard:
        difficulty = "hard"
    else:
        difficulty = "easy"

    # return list [mainLevel, level of difficulty, language level_distribution]
    ret_list_str = ["unknown", "A1", "A2", "B1", "B2", "C1", "C2"]
    return [ret_list_str.index(m) for m in ret_list_str if m == max_level][0]
    


def getWordFrequency(words: list) -> dict:
    """
    :Return: dictionary with word and count
    """
    dici = {}

    for word in words:
        if word in dici:     
            dici[word] += 1
        else:
            dici[word] = 1
            
    return dici


def getWordLevelDataFrameForText(text):
    """
    Eingabe: set(text)
    Ausgabe: DataFrame mit word und level (A1 - C2, unknown) für das gegebene Set des Textes
    """

    # create DataFrame
    word_level_table = pd.DataFrame(columns=['word', 'level'])

    # open CEFR vocabulary file for english
    scriptDir = 'proof_reading/cefr/'
    relPath = "cefr_vocab_en.json"
    cefr_file = open(os.path.join(scriptDir, relPath))
    cefr_data = json.load(cefr_file)

    for w in set(text):

        level: str = ""

        # find the CEFR level info for the current word
        for data in cefr_data:

            if data["word"] == w:
                if data["level"]:
                    level  = data["level"]
                else:
                    level = "unknown"

        # add row WORD LEVEL
        word_level_table = word_level_table.append(
            pd.DataFrame(
                [
                    [w, level]
                ], 
                    columns=['word', 'level']
            )
        )
        
    # close cefr json file
    cefr_file.close()

    return word_level_table


def processText(text):
    mylist = re.findall(r'[^!?.,-;]+',text) # To remove punctuations  
    str1 = ' '.join(mylist)

    # lemmatize the entire text
    # first, split the text to a list of words
    words = TextBlob(str1).words
    # then, lemmatize each word
    lemmatizedText = ""
    for w in words:
        lemmatizedText += "{} ".format(w.lemmatize())

    # normalize the whitespaces for texts which include s.l. 'Title    And I am ...'
    return lemmatizedText

#AWLLevel

global awl_list
awl_list = []
def iterate_nested(d):
  for k, v in d.items():
      if isinstance(v, dict):
          iterate_nested(v)
      else:
          if k == 'awl_words':
            if v:
              for val in v:
                if isinstance(val, dict):
                  iterate_nested(val)
          elif k == 'word':
            if v not in awl_list:
              awl_list.append(v)
  return len(awl_list)

#AWLLevel
def get_count_AWL(essay):
  result = awlify(essay)
  res = json.loads(result)
  awl_count  = iterate_nested(res)
  word_count = count_all_words(essay)
  return round(awl_count/word_count,2)

#ttr
def get_ttr_stats(str1):
  tok = ld.tokenize(str1)
  flt = ld.flemmatize(str1)
  return(round(ld.ttr(flt),2),round(ld.root_ttr(flt),2),round(ld.log_ttr(flt),2))


#LexChainDensity

"""
Create a list with all the relations of each noun 
"""
def relation_list(nouns):

    relation_list = defaultdict(list)
    
    for k in range (len(nouns)):   
        relation = []
        for syn in wordnet.synsets(nouns[k], pos = wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split('.')[0])
        relation_list[nouns[k]].append(relation)
    return relation_list
    


"""
Compute the lexical chain between each noun and their relation and 
apply a threshold of similarity between each word. 
""" 
def create_lexical_chain(nouns, relation_list):
    lexical = []
    threshold = 0.5
    for noun in nouns:
        flag = 0
        for j in range(len(lexical)):
            if flag == 0:
                for key in list(lexical[j]):
                    if key == noun and flag == 0:
                        lexical[j][noun] +=1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
        if flag == 0: 
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical
 

"""
Prune the lexical chain deleting the chains that are more weak with 
just a few words. 
"""       
def prune(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1: 
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain


def get_lexical_chain(essay_text):
    """
    Read the .txt in this folder.
    """
        
    """
    Return the nouns of the entire text.
    """
    position = ['NN', 'NNS', 'NNP', 'NNPS']
    
    sentence = nltk.sent_tokenize(essay_text)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [tokenizer.tokenize(w) for w in sentence]
    tagged =[pos_tag(tok) for tok in tokens]
    nouns = [word.lower() for i in range(len(tagged)) for word, pos in tagged[i] if pos in position ]
        
    relation = relation_list(nouns)
    lexical = create_lexical_chain(nouns, relation)
    final_chain = prune(lexical)
    """
    Print the lexical chain. 
    """   
    # for i in range(len(final_chain)):
    #     print("Chain "+ str(i+1) + " : " + str(final_chain[i]))
    
    return (final_chain)

#Total number of lexical chains normalized with text length.
def get_avg_count_chain(text):
  mylist = re.findall(r'[^!?.,-;]+',text) # To remove punctuations  
  str1 = ' '.join(mylist)
  return round(len(get_lexical_chain(str1))/count_all_words(text),2)

#Average/maximum lexical chain length. 
def get_avg_chain_length(text):
  count_chains = []
  mylist = re.findall(r'[^!?.,-;]+',text) # To remove punctuations  
  str1 = ' '.join(mylist)  
  for l in get_lexical_chain(str1):
    count_chains.append(len(l.keys()))
  return(round(np.mean(count_chains),2))


def score_main_v2(input_essay):
    ret_dict = {}
    lines={}
    wrapper = textwrap.TextWrapper()
    word_list = wrapper.wrap(text=input_essay)
    label_list = ["Bad (D-F)","Average (B-C)","Excellent (A+)"]


    text = ''.join(str(v) for v in word_list)
    lines[0] = text     # creating a dummy dictionary to be later converted to Dataframe.

    df_test = pd.DataFrame(lines.values(),columns=['Essay Text'])
    df_test['SentenceCount'] = df_test['Essay Text'].apply(count_sentences)
    df_test['WordCount'] = df_test['Essay Text'].apply(count_all_words)
    df_test['VerbCount']  = df_test.apply(lambda x: count_feature('VERB',x['Essay Text'], x['WordCount'],2), axis=1)
    df_test['NounCount']  = df_test.apply(lambda x: count_feature('NOUN',x['Essay Text'], x['WordCount'],2), axis=1)
    df_test['AdjCount']  = df_test.apply(lambda x: count_feature('ADJ',x['Essay Text'], x['WordCount'],2), axis=1)
    df_test['AdverbCount']  = df_test.apply(lambda x: count_feature('ADV',x['Essay Text'], x['WordCount'],2), axis=1)
    df_test['PronounCount']  = df_test.apply(lambda x: count_feature('PRON',x['Essay Text'], x['WordCount'],2), axis=1)
    df_test['PunctCount']  = df_test.apply(lambda x: count_feature('PUNCT',x['Essay Text'], x['WordCount'],2), axis=1) 
    df_test['NumberofActiveVoice'] = df_test.apply(lambda row: pd.Series(check_passive_voice(row['Essay Text'])),axis=1) 
    df_test[['NumComplex','NumCompund','NumSimple']] = df_test.apply(lambda row: pd.Series(getCountTypeSent(row['Essay Text'])),axis=1) 
    df_test['FleschScore'] = df_test.apply(lambda x: get_flesch_score(x['Essay Text']),axis=1) 
    df_test['UniqWordDensity']  = df_test.apply(lambda x: uniq_word_count(x['Essay Text']), axis=1)
    df_test['TellDensity']  = df_test.apply(lambda x: showntell_essay(x['Essay Text']), axis=1)
    df_test['AvgChar']  = df_test.apply(lambda x: avg_char_count(x['Essay Text']), axis=1)
    df_test['Syllablesperword']  = df_test.apply(lambda x: count_syllables_per_word(x['Essay Text']), axis=1)  
    df_test['mtld']  = df_test.apply(lambda x: calc_mtld(x['Essay Text']), axis=1)  
    df_test['namedEntities']  = df_test.apply(lambda x: count_ner_per_ent(x['Essay Text']), axis=1)    
    df_test['nerPerSent']  = df_test.apply(lambda x: count_ner_per_sent(x['Essay Text']), axis=1)      
    df_test['CEFRLevel']  = df_test.apply(lambda x: categorizeText(x['Essay Text']), axis=1)      
    df_test['AWLLevel']  = df_test.apply(lambda x: get_count_AWL(x['Essay Text']), axis=1)      
    df_test[['ttr','root_ttr','log_ttr']] = df_test.apply(lambda row: pd.Series(get_ttr_stats(row['Essay Text'])),axis=1) 
    df_test['LexChainDensity']  = df_test.apply(lambda x: get_avg_count_chain(x['Essay Text']), axis=1)      
    df_test['AvgLexChainLen']  = df_test.apply(lambda x: get_avg_chain_length(x['Essay Text']), axis=1)      
    
    Xinput = df_test.drop(['Essay Text'], axis = 1).round(2)

    d = Xinput.to_dict(orient='records')

    ret_dict['predictions'] = [ round(elem, 2) for elem in grading_model.predict_proba(Xinput).tolist()[0]]
    ret_dict['classes'] = [label_list[i] + ':' + str(i)   for i in grading_model.classes_.tolist()]
    ret_dict['values'] = d

    return ret_dict