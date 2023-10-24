# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 08:52:32 2023

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
from app.models.connection.blobstorage import BlobStorage
from app.controllers.assetloader import AssetLoader
import h5py
import logging
from io import BytesIO
from pandas import DataFrame
import random
import json
import libp.text_util as tutil
import numpy as np
import os
import nn_v3 as nn
import time

POSITIVE_BIAS = 0.74

#%% Execution



print( nn.predict_from_loaded_model('JAMES Latham Ltd (Lathams), one of the UK’s leading independent distributors of panel products, has announced the introduction of WISA®’s new BioBond technology to its full range of WISA-Spruce plywood.WISA BioBond is the latest bonding innovation from UPM, a globally-renowned manufacturer of sustainable architectural plywood, now used in the entire WISA-Spruce range. A landmark development in plywood adhesive, BioBond replaces at least 50% of standard glue’s fossil-based phenol with lignin, timber’s inherent bonding agent.Obtained as a by-product of the Kraft Process, this partial substitution reduces the carbon footprint of WISA-Spruce by approximately 10% without compromising technical performance or visual appeal.WISA-Spruce is manufactured using UPM’s proprietary BioBond technology, and mirrors the qualities of plywood produced using the traditional higher-carbon bonding method. This means it offers a like-for-like greener alternative. As with all WISA’s plywood, WISA-Spruce with BioBond technology has undergone rigorous testing to guarantee its high performance qualities, meeting superior standards of strength, resistance and sustainability.More than just a new form of glue, BioBond has reduced CO2 across WISA’s plywood portfolio. Already available for birch, and now spruce, plywood, UPM plans to roll out BioBond across all its plywood mills, gradually covering its entire range. As one of the UK’s most sustainable materials distributors, Lathams is keen to introduce UK specifiers to the low-carbon advantages of WISA-Spruce with BioBond technology. By incorporating this innovation to its ever-expanding collection of green architectural materials, the company is demonstrating its ongoing commitment to supporting sustainable design and build.Commenting on the introduction of BioBond to the WISA-Spruce range, Nick Widlinski, panels director at Lathams says, “There’s no doubt timber and wood-based materials are helping architects and designers tackle global climate change through making lower-emission material choices. However a question around the carbon intensity of glues and adhesives used in the production of engineered wood persists, and WISA BioBond tackles it head on. Its introduction and standardisation across the brand’s high-performance spruce range is a game-changer, offering the best quality with a reduced carbon footprint. Not only is it helping us to promote more sustainable construction methods, it’s also supporting a wider drive toward a Net Zero society.”UPM’s VP of strategy and business development, Susanna Rinne, concludes, “Sustainability is at the heart of our ethos and guides our ongoing R&D. We are the first manufacturer in the world to use a lignin-based solution for spruce and birch plywood, offering a no-compromise sustainable material solution. It’s imperative we work with those who have similar values. … Latham’s longstanding reputation for championing sustainable specification make them a great partner to help us introduce BioBond and its unique properties to the UK and Irish markets.”Providing further confidence in WISA-Spruce’s green credentials and certification, the product category scored one of the best ratings on Lathams’ new Carbon Calculator tool. An academically developed formula which scores the embodied carbon of each Lathams’-stocked timber product from cradle to purchase, BioBond WISA-Spruce achieved top ranking across the board, providing third party verifications for the material’s sustainability claims.For contact details, see page 54 of our October/November 2022 issue on our Back Issues page.', 
                                     model_name='lignin-2'
                                   ))


print( nn.predict_from_loaded_model(txt, 
                                     model_name='lignin-2'
                                   ))


print( nn.predict_from_loaded_model('Twelve clean fuel projects in B.C. will be funded under a new $800 million federal fund clean fuel fund announced today in B.C. The 12 projects include the production of five types of low carbon fuels: hydrogen, renewable natural gas, sustainable aviation fuel, biodiesel and ethanol. Natural Resources Minister Jonathan Wilkinson made the announcement Monday. The new $800 million is part of the earlier announced $1.5 billion Clean Fuels Fund. It will be spent on 60 projects across Canada, including 12 in B.C. It was not divulged how much of the $800 million will go to B.C. projects. Wilkinson also noted that NRCan had already previously invested $9 million in six organizations for 10 new hydrogen and natural gas fueling stations stations in British Columbia, Ontario and Alberta. Those fueling stations include a new HTEC hydrogen fueling station that serves drayage trucks at the Port of Vancouver, and a new FortisBC compressed natural gas (CNG) fueling station for commercial trucks on Annacis Island. “While electrification will be a chosen route in some sectors, clean fuels will play a role, a very significant role, in many areas going forward,” Wilkinson said. Heavy duty long-haul trucking, for example, is one sector that would be very difficult to electrify, which is why hydrogen fuel cells are considered the most likely option for decarbonizing that sector. Aviation is also one of those hard-to-abate sectors. For that sector, low or zero carbon fuels are likely to be one option. Carbon Engineering, for example, is working with a U.S. partner to produce a carbon neutral aviation fuel from captured CO2 and hydrogen. Clean or low-carbon hydrogen can be produced either through electrolysis (using water and electricity) or from natural gas. The CO2 from the latter can be captured and stored to make blue hydrogen. But as Wilkinson noted, for carbon capture, geological sequestration capacity is needed. Dedicated CO2 pipelines are also needed. Alberta has a dedicated CO2 pipeline -- the Alberta Carbon Trunk line -- that is already in use. B.C. has geological sequestration capacity in old natural gas wells, but has no dedicated carbon pipeline. Regionally, each of our provinces and territories has a unique mix of their own natural resources, and so the opportunities that are available to them are going to be different across the country, Wilkinson said. What will be a big opportunity in Alberta will be very different in terms of the opportunities that present themselves in Nova Scotia. Wilkinson noted there is another way to produce hydrogen from natural gas, without requiring carbon capture and storage, and named a B.C. company that has developed such a process: Ekona Power, which uses methane pyrolisis to produce both hydrogen and solid carbon from natural gas. The issue right now is we produce it (hydrogen) from natural gas and we dont capture the CO2, Wilkinson noted. So the first step is actually putting in place systems where we actually are utilizing it for existing applications where were actually either capturing the CO2, or doing what companies like Ekona are trying to do, which is drop the carbon out as a solid, so that you can actually use hydrogen without the the carbon emissions that cause climate change.', 
                                     model_name='biocrude-pyrolisis'
                                   ))



#%% LIGNIN Acc:

lignin_scope_excel = pd.read_excel('lignin-scope-milena.xlsx', 'Sheet1')
lignin_scope_excel_txt_list = list(lignin_scope_excel['content'])

acc = []
for i,txt in enumerate( lignin_scope_excel_txt_list ) :
    prob = nn.predict_from_loaded_model(txt,  model_name='lignin-2' )[0][0]
    print('{} : {}'.format(i, prob))
    
    if prob > POSITIVE_BIAS :
        acc.append(1)
    else :
        acc.append(0)
    #time.sleep(0.1)
    
print('Result: {}'.format( (sum(acc) / len(lignin_scope_excel_txt_list))*100 ))





#%% Train

"""
LIGNIN
"""

words = ['lignin',
        'ligniin',
        'lignos',
        'nanolign',
        'black liquor',
        'licor negro',
        'schwarzlauge',
        'mustalipeä'
        ]


fit_lignin = nn.fit_export_model(model_name='lignin-3', 
                                 excel_file_name='lexis.xlsx', 
                                 words_list=words,
                                 out_scope_excel_file_name='lignin-out-of-scope-milena.xlsx',
                                 scope_excel_file_name='lignin-scope-milena.xlsx',
                                 negative_filler_size=600)



#%% Biocrude & Pyrolisis Train
"""
BIOCRUDE & PYROLISIS Oil
"""
words = ['pirolis',
         'pyroly',
         'liquefação hidro',
         'hydro liquefac',
            'basur',
            'hardwood',
            'softwood',
            'eucalyptus',
            'pine',
            'forest based',
            'annual crop',
            'perennial crop',
            'energy crop',
            'swtichgrass',
            'miscanthus',
            'bagasse',
            'sawdust',
            'cellulos',
            'aquatic crop',
            'vegetable oil',
            'seed oil',
            'Tall Oil',
            'palm oil',
            'fatty acid',
            'rapeseed oil',            
            'bio-based',
            'plant based',
            'plant derived',
            #'biodegrad',
            'eucalipto',
            'pinheiro',
            #'base florest',
            'pó de serra',
            'hemicelulos',
            'cultura aquatica',
            'plant aquat',
            'oleo vegetal',
            'Biomass',
            'base biológica',
            'origen vegetal',
            'derivad planta']

#words = ['palm oil', 'piroli', 'pyroly']

fit1 = nn.fit_export_model(model_name='biocrude-pyrolisis', 
                           excel_file_name='lexis.xlsx', 
                           words_list=words,
                           negative_filler_size=1500)

#excel = pd.read_excel('lexis.xlsx', sheet_name='Sheet1')
#txt_list = list(excel['content'])
#r = nn.filter_texts_by_wordlist(words, txt_list)

#%% Carbon Markets
"""
Carbon Markets
"""
words = [ 
    "Carbon credit",
    "crédito de carbono|STRICT",
    "carbon market",
    "mercado de carbono|STRICT",
    "mercado regulado de carbono|STRICT",
    "carbon offset",
    "BECCS",
    "Bioenergy with carbon capture and storage|STRICT"
    ]

fit3 = nn.fit_export_model( model_name='carbon-markets', 
                            excel_file_name='lexis.xlsx', 
                            words_list=words,
                            negative_filler_size=2000)


#%%PLOT PERFORMANCE

import matplotlib.pyplot as plt

def plotloss(model):
    history_dict = model.history
    
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12,9))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plotacc(model):
    history_dict = model.history
    
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12,9))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim((0.5,1))
    plt.show()

plotloss(fit_lignin['model-history'])
plotacc(fit_lignin['model-history'])

plotloss(fit1['model-history'])
plotacc(fit1['model-history'])

plotloss(fit3['model-history'])
plotacc(fit3['model-history'])

