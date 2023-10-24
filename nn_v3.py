# -*- coding: utf-8 -*-
"""
Created on Jan 2023

@author: Miguel
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.saving import hdf5_format
#from keras.backend import clear_session
from tensorflow.keras.backend import clear_session
import pickle
import pandas as pd
import h5py
import logging
from io import BytesIO
from pandas import DataFrame
import random
import json
import libp.text_util as tutil
import regex
import numpy as np
import os
from tensorflow.python.client import device_lib

def print_tf_info():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

#LANGUAGE='EN'
#language_rules = json.load(open(f'asset/language_asset/{LANGUAGE}_rules.json'))
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def randomize_loaded_train(text_list ) -> DataFrame :
        
    len_ = len(text_list)
    rand_list = [ random.triangular(0, 1) for i in list(range(0,len_))]
    
    sentences = [ i[0] for i in text_list]
    labels    = [ i[1] for i in text_list]
    
    datatable = DataFrame({'sentence' :sentences, 'label' : labels, 'rand':rand_list}, columns=['sentence', 'label', 'rand'])
    sorted_ = datatable.sort_values(by=['rand'], ascending=True)
    
    return {'sentence' : list(sorted_['sentence']), 'label' : list(sorted_['label']) }
    

def filter_texts_by_wordlist(word_list:list, text_list:list, with_index:bool=False) :
    filtered_text_list = {}
    for w in word_list :
        w = w.strip()
        print(f'Searching for {w}..')
                
        strict = w.split('|')
        strict = True if 'STRICT' in strict else False
        word = w.split('|')[0]
        
        filtered_text_list[w] = get_filtered_text_list(get_word_regex(word, restrict=strict),
                                                       text_list,
                                                       with_index=with_index)
    return filtered_text_list


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

def get_word_regex(word: str, restrict: bool=False) :
    return_ = r'(\W|^)[\p{L}\-]*-' + word + r'[\p{L}\-]*(\W|$)|(\W|^)[\p{L}\-]*-' + word + r'(\W|$)|((\W|^)' + word + r'[\p{L}\-]*(\W|$))'
    
    if ' ' in word:
        wl = word.split(' ')
        return_ = f'{return_}|((\W|^)' + wl[0] + r'[\p{L}\-]* ' + wl[1] + '[\p{L}\-]*(\W|$))'
    
    if restrict :
        return_ = r'(\W|^)' + word + r'(\W|$)'
        
    return return_


def fit_model_training_set(excel_file_name,
                           words:list,
                           sheet='Sheet1',                             
                           out_scope_excel_file_name=None,
                           scope_excel_file_name=None,
                           epochs = 25,
                           negative_filler_size = 2000,
                           batch_size=1) :
    """
    BULD TRAIN SET
    """
    excel = pd.read_excel(excel_file_name, sheet_name=sheet)
    txt_list = list(excel['content'])

    if out_scope_excel_file_name:
        out_scope_excel = pd.read_excel(out_scope_excel_file_name, 'Sheet1')
        out_scope_excel_txt_list = list(out_scope_excel['content'])
    else :
        out_scope_excel_txt_list = []
        
    if scope_excel_file_name:
        scope_excel_file = pd.read_excel(scope_excel_file_name, 'Sheet1')
        scope_excel_txt_list = list(scope_excel_file['content'])
    else :
        scope_excel_txt_list = []

    negative_text_list = []
    positive_text_list = []

    filtered_text_list = filter_texts_by_wordlist(words, txt_list, with_index=True)
    
    positive_index = [] 
    for wt in filtered_text_list.values() :
        for m, txt, rindex in wt['t'] :
            txt = tutil.clean_html(keep_only_word_characters( txt ))
            
            positive_index.append(rindex)
            positive_text_list.append((txt, 1))
    
    scope_excel_txt_list = [ ( tutil.clean_html(keep_only_word_characters( t )), 1) 
                             for t in scope_excel_txt_list]
    
    positive_text_list = positive_text_list + scope_excel_txt_list
    
    logging.warning('Positive List created')
    
    train_filler_size = negative_filler_size
    negative_text_list = get_random_list_items(txt_list, train_filler_size, not_list=positive_index) 


    train_negative_filler_list = [ ( tutil.clean_html(keep_only_word_characters( t )), 0) 
                                   for t in (negative_text_list + out_scope_excel_txt_list)]
    
    logging.warning('Negative List created')
    
    train_list = positive_text_list + train_negative_filler_list 

    train_set = randomize_loaded_train(train_list)

    sentences = train_set['sentence']
    labels = train_set['label']

    len_sentences = len(sentences)
    train_size_factor = 0.8

    training_size = round(len_sentences * train_size_factor) #.75
    training_sentences = sentences[0:training_size]        
    testing_sentences = sentences[-round(len_sentences*(1-train_size_factor)):]

    training_labels = labels[0:training_size]
    testing_labels = labels[-round(len_sentences*(1-train_size_factor)):]
    
    vocab_size = 10000    
    
    tokenizer = get_tokenizer(sentences, vocab_size=vocab_size)
    
    logging.warning('Sentences Tokenized.. ')
        
    # Setting the padding properties
    max_length = 2500
    trunc_type='post'
    padding_type='post'
    # Creating padded sequences from train and test data
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    #training_padded = training_padded / len(word_indexx)
    #testing_padded = testing_padded / len(word_indexx)
    
    # Setting the model parameters
    embedding_dim = 300
    hidden_layer_neurons = 32
    model = tf.keras.Sequential([    
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense( hidden_layer_neurons, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
        tf.keras.layers.Dense( 1, activation='sigmoid' ) #Probability
        
    ])
    
    model.summary()
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    logging.warning('Model compiled..')
    
    """
    MODEL FIT
    """    
    
    history = model.fit( x = training_padded, 
                         y = np.asarray(training_labels).astype('float32'),
                         epochs = epochs,
                         batch_size = 32,
                         validation_data = (testing_padded, np.asarray(testing_labels).astype('float32')) )
    
    words_founded = []
    for v in  filtered_text_list.values() :
        for i in v['t'] :
            words_founded.append(i[0])
            
    return {'model'                   : model, 
            'tokenizer'               : tokenizer, 
            'found-words-from-filter' : set(words_founded),
            'set-length' :  {'positive': positive_text_list, 
                             'negative': len(negative_text_list), 
                             'total'   : len(txt_list) },
            'total-filtered'          : len(set(positive_index)),
            'filtered_text_list'      : filtered_text_list,
            'model-history'           : history} 


def get_tokenizer(sentences, vocab_size = 10000, oov_tok = "<oov>"):
    # Setting tokenizer properties
        
    # Fit the tokenizer on Training data
    tokenizer = Tokenizer(num_words=vocab_size, 
                          oov_token=oov_tok,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789°º',
                          lower=True,
                          char_level = False
                          )
    
    tokenizer.fit_on_texts(sentences)
    
    return tokenizer


def predict(model, tokenizer, txt) :
    txt = tutil.keep_only_word_characters(
            tutil.clean_html( txt ))

    words_total = len( txt.split(' ') )

    padded = pad_sequences(tokenizer.texts_to_sequences( [ txt ] ), 
                  maxlen=2500,
                  padding='post', 
                  truncating='post' )

    return model.predict(padded, batch_size=1)
    

def export(name, model, tokenizer):
    model_name = f'{name}-model.hdf5'
    tokenizer_name = f'{name}-tokenizer.dat'

    hdf5_file = h5py.File(f'asset/{model_name}', 'w') 
    hdf5_format.save_model_to_hdf5( model, hdf5_file ) 
    hdf5_file.close()

    pickle.dump(tokenizer, open(f'asset/{tokenizer_name}', 'wb'))

    print(f'model {model_name} Exported...')


def load_model(name) :
    ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

    model_name = f'{name}-model.hdf5'
    tokenizer_name = f'{name}-tokenizer.dat'

    reconstructed_model = hdf5_format.load_model_from_hdf5(f'{ROOT_PATH}/asset/{model_name}')
    # Load tokenizer
    reconstructed_model_tokenizer = pickle.load(open(f'{ROOT_PATH}/asset/{tokenizer_name}', 'rb'))
    
    return {'model' : reconstructed_model, 'tokenizer' : reconstructed_model_tokenizer}


def predict_from_loaded_model (txt, model_name) :
    load = load_model(model_name)    
    return predict(load['model'], load['tokenizer'], txt)


def fit_export_model (model_name, excel_file_name, words_list, scope_excel_file_name=None, out_scope_excel_file_name=None, negative_filler_size=2000) :
    logging.warning(f'Fitting {model_name} using {excel_file_name}')
    fit = fit_model_training_set( excel_file_name,
                                  words_list,
                                  out_scope_excel_file_name=out_scope_excel_file_name,
                                  scope_excel_file_name= scope_excel_file_name,
                                  negative_filler_size=negative_filler_size)
    
    logging.warning(f'Exporting {model_name} on asset folder')
    export(model_name, fit['model'], fit['tokenizer'])
    
    return fit

def predict_from_new_fit_model (txt, model_name, excel_file_name, words_list, out_scope_excel_file_name ) :
    
    fit = fit_model_training_set(excel_file_name , words_list, 
                                 out_scope_excel_file_name=out_scope_excel_file_name)

    model = fit['model']
    tokenizer = fit['tokenizer']
    res = predict(model, tokenizer, txt)
    
    return {'res': res, 'model':model, 'tokenizer': tokenizer} 
