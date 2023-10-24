# -*- coding: utf-8 -*-
"""
Created on Mon Jan 2023

@author: Spume.co
"""

#%% LIB
import io
import logging

import pandas as pd
from pandas import DataFrame
import regex
import libp.text_util as tutil
import logging
import json
import random


LANGUAGE='EN'
language_rules = json.load(open(f'asset/language_asset/{LANGUAGE}_rules.json'))


# p = r'\blignin.*\b|\bligniin.*\b|\blignos.*\b|\bnanolign.*\b|\bblack liquor\b|\blicor negro\b|\bschwarzlauge\b|\bmustalipeä\b|\bgemuse\b|\bkasvis\b|\bhedelmää\b|\bFrucht\b|\bfruit.*\b|\bbenef.*\b|\badvantag.*\b|\bimprov.*\b|\bmelhor.*\b|\bverbessern\b|\bparantaa\b|\bhyötyä\b|\bNutzen\b|\bsaud.*\b|\bsalud.*\b|\bhealth.*\b|\bterveys\b|\bvointi\b|\bgesundheit\b|\bblack liquor\b|\blicor negro\b|\bSchwarzlauge\b|\bmustalipeä\b|\bdisast.*\b|\bkatast.*\b|\bcatastroph\b|\bkill.*\b|\bmort.*\b|\bsurmata\b|\btappaa\b|\btoten\b|\bumbringen\b|\bvernichten\b|\bMarktanteil\b|\bMarktforschung\b|\bLignina\b|\bpatent\b'
regp = r'\blignin[\p{L}-]*\b|\bligniin[\p{L}-]*\b|\blignos[\p{L}-]*\b|\bnanolign[\p{L}-]*\b|\b“black liquor”\b|\blicor negro\b|\bschwarzlauge\b|\bmustalipeä\b'
regp = r'(\W|^)[\p{L}\-]*-lignin[\p{L}\-]*(\W|$)|(\W|^)[\p{L}\-]*-lignin(\W|$)|((\W|^)lignin[\p{L}\-]*(\W|$))'


def _exec(words: list=None) :
    result_ = {}
    
    excel = pd.read_excel('lexis.xlsx', 'Sheet1')
    
    txt_list = list(excel['content'])
    
    
    """
    words = ['lignin',
            'ligniin',
            'lignos',
            'nanolign',
            'black liquor',
            'licor negro',
            'schwarzlauge',
            'mustalipeä']
    """    
    
    for w in words :
        print(f'Searching for {w}..')
        
        filtered_text_list = get_filtered_text_list(get_word_regex(w),
                                                    txt_list)
        
        ap = get_train_phrases_from_text_list( filtered_text_list['t'] )        
        
        result_[w] = ap
        print(f'Found {len(ap)} phrases for {w}.. ')
                        
    return result_

def filter_texts_by_wordlist(word_list:list, text_list:list, with_index:bool=False) :
    filtered_text_list = {}
    for w in word_list :
        w = w.strip()
        print(f'Searching for {w}..')
        
        filtered_text_list[w] = get_filtered_text_list(get_word_regex(w),
                                                       text_list,
                                                       with_index=with_index)
    
    return filtered_text_list


def apply_language_rules(text:str) :
    pass


def get_word_regex(word: str, restrict: bool=False) :
    return_ = r'(\W|^)[\p{L}\-]*-' + word + r'[\p{L}\-]*(\W|$)|(\W|^)[\p{L}\-]*-' + word + r'(\W|$)|((\W|^)' + word + r'[\p{L}\-]*(\W|$))'
    
    if ' ' in word:
        wl = word.split(' ')
        return_ = f'{return_}|((\W|^)' + wl[0] + r'[\p{L}\-]* ' + wl[1] + '[\p{L}\-]*(\W|$))'
    
    if restrict :
        return_ = r'(\W|^)' + word + r'(\W|$)'
        
    return return_


def create_word_regex(word:str) :
    pass


def get_filtered_text_list(reg, text_list, with_index = False):
    result = {'t': [], 'nan': 0}
    texts = []
    total = len(text_list)
    print(f'\r Initializing Filtering for {reg}... ', end='\r')
    for i, txt in enumerate( text_list):
                
        print(f'\rFiltering {i}/{total}... ', end='\r')
                
        txt = keep_only_word_characters( tutil.clean_html( str(txt) ))
        mat = regex.search(reg, str(txt), flags=regex.IGNORECASE | regex.MULTILINE | regex.UNICODE)
        if mat:
            if with_index:
                result['t'].append((mat.group(0).strip(), txt, i))
            else :
                result['t'].append((mat.group(0).strip(), txt))

        if 'nan' == str(txt):
            result['nan'] += 1    

    return result


def get_train_phrases_from_text_list(text_list) :
    phrases = []
    total = len(text_list)
    for i, t in enumerate(text_list):        
        print(f'\rExtracting phrases {i}/{total}... ', end='\r')        
        
        mat, txt = t
        
        #Apply Language text transform
        #txt = apply_language_rules( txt )
        
        if txt :
            phrases.append(get_word_context_phrases(txt, mat, i))
        
    return phrases


def keep_only_word_characters(text: str, keep_chars: str = None):
    """

    :param text:
    :param keep_chars:
    :return:
    """
    reg = r'\p{L}\s\-'
    if keep_chars:
        reg += keep_chars
    sub = regex.sub(r'[^' + reg + r']', ' ', text, flags=regex.IGNORECASE | regex.UNICODE)
    sub = regex.sub(r'\s+', ' ', sub, flags=regex.UNICODE)
    
    return sub


def get_word_context_phrases(text: str, word: str, index: int, word_distance: int = 15):
    """

    :param text: Cleaned text
    :param word:
    :param index:
    :param word_distance:
    :return:
    """
    return_ = []
    pat = get_word_regex( tutil.addslashes( word ))
    text = tutil.clean_html(text)
    
    logging.warning(f'match {index}')    
    # pat = r'\blignin[\p{L}\-]*\b|\bligniin[\p{L}\-]*\b|\blignos[\p{L}\-]*\b|\bnanolign[\p{L}\-]*\b|\b“black liquor”\b|\blicor negro\b|\bschwarzlauge\b|\bmustalipeä\b'
    #pat = r'(-' + word + r')?((\W|^)[\p{L}\-]*-' + word + r'[\p{L}\-]*(\W|$))|((\W|^)' + word + r'[\p{L}\-]*(\W|$))'
    #pat = r'(\W|^)[\p{L}\-]*-' + word + r'[\p{L}\-]*(\W|$)|(\W|^)[\p{L}\-]*-' + word + r'(\W|$)|((\W|^)' + word + r'[\p{L}\-]*(\W|$))'
        
    #clean text e aplit by space
    try :
        
        text_word_list = text.split(' ')
    except TypeError as te:
        logging.error(te)
        logging.error(f'word: {word} t: {text} idex: {index}')
        text = ""
        text_word_list = []
        
    text_word_list_last_index = len(text_word_list) - 1
        
    last_index = 0
    for match in regex.finditer(pat, text, flags=regex.IGNORECASE | regex.MULTILINE | regex.UNICODE):
        word_text_index_list = None
        text_match = match.group(0).strip()
        logging.debug(f'\rmatch: {match}')
        try:
            match_index = get_word_match_index(text_match, text_word_list, last_index)[1] #index list lisition                       
            
            if match_index :
                #Go through text and find REGEX match positions
                word_text_index_list = match_index
                start_word_text_index = word_text_index_list[0]
                end_word_text_index = word_text_index_list[-1] #last
                
        except (ValueError,TypeError) as e:
            logging.warning('Phrase not considered: {} {} {}'.format(text_match, pat, index))
            logging.error(e)
            e_c_index = e.args[1]
            last_index = e_c_index
        except IndexError as ie:
            # occurs if get_word_match_index not find next compund 
            # combination found by regex finditer
            logging.error('Index out of range')
            return []

        if word_text_index_list :
            start_pos = max( [(start_word_text_index - word_distance), 0])
            end_pos = min([ (end_word_text_index + word_distance),  text_word_list_last_index])
            
            #Append match phrases on return list
            return_.append({'before': text_word_list[start_pos: start_word_text_index],
                            'match': text_word_list[start_word_text_index: (end_word_text_index + 1)],
                            'after': text_word_list[( end_word_text_index + 1) : min([(end_pos+1), text_word_list_last_index])]})
            
            last_index = end_word_text_index+1

    return return_


def get_word_match_index(word, text_word_list:list, last_index: int=0) :
    index_list = []
    c_index=0
    logging.debug(f'INIT word: {word} c_index : {c_index} lasti: {last_index}')
    i=0
    word_split = word.split(' ')
    while i < (len( word_split)) :
        logging.debug(f'WHILE INIT word: {word} i: {i} c_index : {c_index} lasti: {last_index}')
        term = word_split[i]
        try :
            c_index = text_word_list.index(term, max(last_index, c_index))
        except ValueError:
            logging.error('No more Index to search on text_word_list')
            return (None,None,None)
                        
        #verifies if compound term is sequencially valid
        if i > 0:
            prev_index = index_list[i-1]
            logging.debug(f'prev_index: {prev_index} c_index : {c_index} i: {i}')
            if c_index == (prev_index+1) :
                index_list.append(c_index)
                i += 1
                logging.debug(f'Second term added to list term:{term} c_index : {c_index} i: {i}')
            else:                
                logging.debug(f'EXCEPTION word: {word} term: {term} c_index : {c_index} i: {i} lasti: {last_index}')
                logging.debug(f'returning to the first term {word[i-1]}..')
                i = 0
                c_index = index_list[0] + 2 #force go to next words
                index_list = []
                                
        else :
            logging.debug (f'added to index list w: {term} c_index: {c_index}')
            index_list.append(c_index)
            i += 1
        
    logging.debug(f'index_list: {index_list}')
    
    #return INDEX position of REGEX match that called this method
    return (word, index_list, len(index_list))
        

def get_random_list_items(text_list:list, size:int=100, not_list:list=[]) :
    total = len(text_list)
    i=0
    return_ = []
    while i < size :
        r = random.randint(0, total-1)
        if r not in not_list:
            if not str( text_list[r] ) == 'nan' :
                return_.append(str(text_list[r]))
                i += 1
            
    return return_

   
    
    
#%% TEST
"""
BULD TRAIN SET
"""


"""
TESTS

test = get_train_phrases_from_text_list(regp, txt_list)
testtttt = get_train_phrases_from_text_list(regp, txt_list)

test = get_word_context_phrases(keep_only_word_characters(result2['t'][0]), 'lignin')
test = keep_only_word_characters(result2['t'][0])

lines = []
for plist in test:
    for p in plist:
        b = p['before']
        m = p['match']
        a = p['after']

        lines.append( '<li>{} <strong>{}</strong> {}</li>'.format(' '.join(b), ' '.join(m), ' '.join(a)) )

html = '<html><head></head>'
html += '<style> li { ' \
        'padding: 20px 10px; ' \
        'border-radius:10px; border: solid 1px #fff; ' \
        'box-shadow: 5px 2px 10px gray; ' \
        'width:70%;  ' \
        'font-family: Helvetica, Arial;  ' \
        'margin: 15px auto;' \
        ' } ' \
        'strong { padding: 3px; background: yellow; }' \
        '</style><body><ul>'

html += '\n'.join( lines )
html += '</ul></body></html>'

with open('tests/phrases_result.html', mode='w', encoding='utf-8') as f :
    f.write(html)
    f.close()
    
sum( [ len(l) for l in test ] )

"""