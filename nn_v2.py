# -*- coding: utf-8 -*-

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
from read_lexis import *
import numpy as np


def randomize_loaded_train(text_list ) -> DataFrame :
        
    len_ = len(text_list)
    rand_list = [ random.triangular(0, 1) for i in list(range(0,len_))]
    
    sentences = [ i[0] for i in text_list]
    labels    = [ i[1] for i in text_list]
    
    datatable = DataFrame({'sentence' :sentences, 'label' : labels, 'rand':rand_list}, columns=['sentence', 'label', 'rand'])
    sorted_ = datatable.sort_values(by=['rand'], ascending=True)
    
    return {'sentence' : list(sorted_['sentence']), 'label' : list(sorted_['label']) }
    


#%% BUILS TRAIN SET
"""
BULD TRAIN SET
"""
excel = pd.read_excel('lexis.xlsx', 'Sheet1')
txt_list = list(excel['content'])

lignin_excel_out_scope_ = pd.read_excel('lignin-out-of-scope-.xlsx', 'Sheet1')
lignin__out_scope_txt_list = list(lignin_excel_out_scope_['content'])

words = ['lignin',
        'ligniin',
        'lignos',
        'nanolign',
        'black liquor',
        'licor negro',
        'schwarzlauge',
        'mustalipeä']

negative_text_list = []
positive_text_list = []

filtered_text_list = filter_texts_by_wordlist(words, txt_list, with_index=True)

positive_index = [] 
for wt in filtered_text_list.values() :
    for m, txt, rindex in wt['t'] :
        positive_index.append(rindex)
        positive_text_list.append((txt, 1))

train_filler_size = 1000
negative_text_list = get_random_list_items(txt_list, train_filler_size, not_list=positive_index) 


train_negative_filler_list = [ ( tutil.clean_html(keep_only_word_characters( t )), 0) 
                               for t in (negative_text_list + lignin__out_scope_txt_list)]

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

#%% MODEL

# Setting tokenizer properties
vocab_size = 10000
oov_tok = "<oov>"

# Fit the tokenizer on Training data
tokenizer = Tokenizer(num_words=vocab_size, 
                      oov_token=oov_tok,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789°º',
                      lower=True,
                      char_level = False
                      )

tokenizer.fit_on_texts(sentences)

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
hidden_layer_neurons = 512
model = tf.keras.Sequential([    
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense( hidden_layer_neurons, activation=tf.nn.relu),
    tf.keras.layers.Dense( 1, activation='sigmoid' ) #Probability
    
])

model.summary()
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#%% MODEL FIT
epochs = 10
history = model.fit( x = training_padded, 
                     y = np.asarray(training_labels).astype('float32'),
                     epochs = epochs,
                     batch_size = 32,
                     validation_data = (testing_padded, np.asarray(testing_labels).astype('float32')) )


#%% PREDICT TEST

txt = 'JAMES Latham Ltd (Lathams), one of the UK’s leading independent distributors of panel products, has announced the introduction of WISA®’s new BioBond technology to its full range of WISA-Spruce plywood.WISA BioBond is the latest bonding innovation from UPM, a globally-renowned manufacturer of sustainable architectural plywood, now used in the entire WISA-Spruce range. A landmark development in plywood adhesive, BioBond replaces at least 50% of standard glue’s fossil-based phenol with lignin, timber’s inherent bonding agent.Obtained as a by-product of the Kraft Process, this partial substitution reduces the carbon footprint of WISA-Spruce by approximately 10% without compromising technical performance or visual appeal.WISA-Spruce is manufactured using UPM’s proprietary BioBond technology, and mirrors the qualities of plywood produced using the traditional higher-carbon bonding method. This means it offers a like-for-like greener alternative. As with all WISA’s plywood, WISA-Spruce with BioBond technology has undergone rigorous testing to guarantee its high performance qualities, meeting superior standards of strength, resistance and sustainability.More than just a new form of glue, BioBond has reduced CO2 across WISA’s plywood portfolio. Already available for birch, and now spruce, plywood, UPM plans to roll out BioBond across all its plywood mills, gradually covering its entire range. As one of the UK’s most sustainable materials distributors, Lathams is keen to introduce UK specifiers to the low-carbon advantages of WISA-Spruce with BioBond technology. By incorporating this innovation to its ever-expanding collection of green architectural materials, the company is demonstrating its ongoing commitment to supporting sustainable design and build.Commenting on the introduction of BioBond to the WISA-Spruce range, Nick Widlinski, panels director at Lathams says, “There’s no doubt timber and wood-based materials are helping architects and designers tackle global climate change through making lower-emission material choices. However a question around the carbon intensity of glues and adhesives used in the production of engineered wood persists, and WISA BioBond tackles it head on. Its introduction and standardisation across the brand’s high-performance spruce range is a game-changer, offering the best quality with a reduced carbon footprint. Not only is it helping us to promote more sustainable construction methods, it’s also supporting a wider drive toward a Net Zero society.”UPM’s VP of strategy and business development, Susanna Rinne, concludes, “Sustainability is at the heart of our ethos and guides our ongoing R&D. We are the first manufacturer in the world to use a lignin-based solution for spruce and birch plywood, offering a no-compromise sustainable material solution. It’s imperative we work with those who have similar values. … Latham’s longstanding reputation for championing sustainable specification make them a great partner to help us introduce BioBond and its unique properties to the UK and Irish markets.”Providing further confidence in WISA-Spruce’s green credentials and certification, the product category scored one of the best ratings on Lathams’ new Carbon Calculator tool. An academically developed formula which scores the embodied carbon of each Lathams’-stocked timber product from cradle to purchase, BioBond WISA-Spruce achieved top ranking across the board, providing third party verifications for the material’s sustainability claims.For contact details, see page 54 of our October/November 2022 issue on our Back Issues page.'

txt = '2022 DEC 06 (NewsRx) -- By a News Reporter-Staff News Editor at Agriculture Daily -- Researchers detail new data in Agriculture - Industrial Crops and Products. According to news reporting originating from Nanjing, People s Republic of China, by NewsRx correspondents, research stated, Lignin is extensively studied to be used as an energy storage material due to its intrinsic catechin structure. However, the limited content of catechin groups in industrial lignin is not enough to satisfy the demand for highly charged storage devices.     Financial supporters for this research include National Key Research and Development Project of the 13th Five-Year Plan, Jiangsu Provincial Postgraduate Research and Practice Innovation Program, Jiangsu Provincial Higher Education Key Discipline Construction Grant Program (PAPD).   Our news editors obtained a quote from the research from Nanjing Forestry University, Herein, the pseudocapacitance of lignin itself is further improved by demethylation and cleavage of aryl ether bonds of alkali lignin (AL) using the HBr/LiBr system. The phenolic hydroxyl (Ar-OH) group content of the demethylation alkali lignin (DAL, up to 2.99 mmol.L-1) is 1.85 times higher than that of AL. The as-prepared DAL@ reduced graphene oxide (rGO) composites obtained from DAL and graphene oxide (GO) showed excellent energy storage capacity (414.5 F.g(-1)), which increased by 96.60% compared to the AL@rGO composites. More significantly, the DAL@rGO materials displayed excellent rate capacity, and its capacitance retention rate reached 76.91%.   According to the news editors, the research concluded: Therefore, this strategy of structural modification of lignin is effective for improving its energy storage performance, which will provide a possibility for the development of lignin into higher-performance energy storage materials.   This research has been peer-reviewed.   For more information on this research see: Structural Modification of Alkali lignin Into Higher Performance Energy Storage Materials: Demethylation and Cleavage of Aryl Ether Bonds. Industrial Crops and Products, 2022;187. Industrial Crops and Products can be contacted at: Elsevier, Radarweg 29, 1043 Nx Amsterdam, Netherlands. (Elsevier - www.elsevier.com; Industrial Crops and Products - www.journals.elsevier.com/industrial-crops-and-products/)   The news editors report that additional information may be obtained by contacting Hongqi Dai, Nanjing Forestry University, Jiangsu Coinnovat Ctr Efficient Proc & Utilizat F, Nanjing 210037, People s Republic of China. Additional authors for this research include Mengya Sun, Xiu Wang, Liang Jiao, Huiyang Bian and Shuzhen Ni.   The direct object identifier (DOI) for that additional information is: https://doi.org/10.1016/j.indcrop.2022.115441. This DOI is a link to an online electronic document that is either free or for purchase, and can be your direct source for a journal article and its citation.   Keywords for this news article include: Nanjing, People s Republic of China, Asia, Industrial Crops and Products, Agriculture, Nanjing Forestry University.   Our reports deliver fact-based news of research and discoveries from around the world. Copyright 2022, NewsRx LLC'

#6% parece nao ser correto
txt = 'Greif Paper Packaging and Services Division announceinvestment in recycling facilitiesGreif Paper Packaging and Services (PPS) business has announced two significant investments in its recycling operations to support its growth strategy in sustainable paper packaging solutions.A new 81,000-square-foot paper recycling facility for collecting, processing, and baling has opened in Florence, Kentucky, bringing Greif total number of recycling facilities across North America to 19.In addition, Greif has more than doubled the size of its paper fiber recycling plant in Nashville, Tennessee, to enhance efficiency and support accelerated growth. The expanded facility covers a floor space of 72,000 square feet.Both plants are located in areas where demand for a collection of waste fiber and the supply of recycled paper and board is experiencing continued growth."These exciting projects will help strengthen Greif PPS division and are an important part of our investment strategy. Together they are expected to grow Greif fiber basket by more than 5,000 tons per month. The paper fiber recycling plants are located near Greif-operated paperboard mills and other containerboard and recycled paper product manufacturers, and where greater growth opportunities are possible," explained John Grinnell, Vice President and General Manager of the Recycling Group at Greif. "By positioning capacity closer to our suppliers and customers, we can better support their sustainability targets while aligning with our own environmental goals."Greif paper fiber recycling facilities offer complete outsourcing solutions for pulp and paper fiber procurement, transportation, and administration. Approximately 50 percent of the fiber Greif collects through its recycling business is used to supply Greif containerboard mills, where the paper bales are made into various grades of recycled paperboard and containerboard. The remainder is sold to other US and international producers of recycled paper products.Greif Industrial Products Group converts these rolls and sheets from the Greif paper mill into finished paper products and packaging, including corrugated boxes, tubes and cores, and other various paperboard products.Due to its integrated capabilities, Greif holds a central position in the paper recycling industry and operates as a net-positive recycler.Since acquiring Caraustar Industries in 2019, Greif  PPS division now has 19 paper fiber recycling facilities. PPS also operates 14 paper mills, and between its other business units, Core Choice, IPG, and Adhesives, it has more than 40 facilities across North America.AttachmentsDisclaimerGreif Inc. published this content on 01 November 2022 and is solely responsible for the information contained therein. Distributed by , unedited and unaltered, on 01 November 2022 14:39:03 UTC.'

txt = 'Lignin valorization has become a top technological challenge. Lignin is one of the three main components of wood, the ubiquitous and carbon capturing resource. The other two are cellulose and hemicellulose. Lignin, accounting for 15-30% of woody biomass, is the least developed of these. Although the past fifteen years have seen a spectacular rise in lignin understanding, industrial applications don’t pop up. What are the problems and how do we find solutions?  An underutilized resource In 2016, Utrecht University and Platform Agro-Paper-Chemistry issued a report ‘Lignin Valorization, The Importance of a Full Value Chain Approach’. It offers an excellent summary of recent research. Lignin, according to this paper, is a very interesting biopolymer. It has unique properties because of its highly aromatic nature and low oxygen content compared to cellulose and hemicellulose. It could make a very useful resource for renewable building blocks, fuels and materials. But although this resource is in ample supply, its isolation technology and treatment pose formidable problems.  So far, these problems stood in the way of mass application of this material, except for heat production. The global pulp industry for instance produces 55 million tons of lignin each year, but the vast majority of this is burnt. As we can foresee that the era of fossil fuels is drawing to its end, proper lignin valorization has become important. But then, we should be able to isolate high-quality, useful lignin without compromising the quality of those other two main components. We will need to develop a proper wood biorefinery that valorizes all wood components. For it is often noted that ‘in order to create economically feasible biorefineries and overcome the initial energy cost associated with processing and pretreatment, all three major constituents should be fully valorized.’  A complex polymer Lignin is a complex cross-linked polymer consisting mainly of coumaryl-, coniferyl-and sinapyl phenolic structures, also called monolignols. It is formed in the cell walls of wood or agricultural crops and plants, providing structural strength. But the term ‘lignin’ denotes a vast variety of substances. Their composition depends on the specific plant or tree species, the season, the climate and the plant age. Both the incidence of the most important building blocks and the ways in which they are linked together depend strongly on the plant species. Lignin monomers can connect in a number of ways (notably by direct carbon linking and the ether bond) and at different locations in the molecules.  To this variability, researchers and industries add their differences, owing to the processes used to treat the substance. We call the substances resulting from these treatment processes ‘technical lignins’. They differ from the original substance, and from each other, in characteristics like molecular weight, water solubility and degree of contamination (e.g. the incorporation of non-native elements, such as sulphur). As a result, also their physical and chemical properties vary. Properties like solubility in different solvents and molecular weight, greatly determine the possibilities for lignin valorization.  A large amount of lignin is produced as a side product to paper production. Kuusankoski paper mill, 1987. Photo: Felix O, Wikimedia Commons. A large amount of lignin is produced as a side product to paper production. Kuusankoski paper mill, 1987. Photo: Felix O, Wikimedia Commons. Industrial processes in lignin valorization Industry uses a number of processes to treat wood as a resource. Often, these processes are not particularly suited for lignin valorization; cellulose (and to a lesser extent) hemicellulose are the main products. New processes however that may produce better quality lignin, tend to produce low-quality cellulose. Processes that deliver good qualities of all three main components haven’t yet been developed.  – Acidic pulping is done with sulphite or bisulphite and a hydroxide. The lignin ends up dissolved in the pulping liquor (black liquor) as lignosulfonate, together with some degraded carbohydrates. Most lignin obtained in this way is burnt in order to recover the chemicals. – Alkaline pulping is done with sodium hydroxide, often together with sodium sulphide (Kraft process). Normally, the lignin in the black liquor is burnt for recovery of chemicals and energy production and thus not freely available. The Lignoboost process however, has been developed to isolate part of the lignin from black liquor in a rather pure way. – The Bergius-Rheinau process uses concentrated hydrochloric acid and produces a highly insoluble technical lignin. – In steam explosion processing, woody biomass is treated with steam at about 200oC and high pressure for a short period of time followed by a rapid decompression; this results in a quite native lignin though with significantly reduced molecular weight. – Organosolv pulping involves biomass treatment at rather high temperatures with a mixture of water and an organic solvent like ethanol or acetic acid. The resulting lignin is quite well dissolvable. – Hydrolysis of woody biomass will treat cellulose and hemicellulose and leave a residue of non-soluble lignin. – Modern methods use Deep Eutectic Solvents (DES) or Ionic Liquids to dissolve wood in its entirety; treatment of such solutions may render valuable substances. However, no industrial process emerged from these attempts yet.  Applications In principle, lignin can be extracted in a rather pure form and used as such. However, no industrial applications emerged so far. Much research is being done, but applications are ‘at early stages of development’. The other processes hardly produce lignin of commercial use, other than for incineration. For instance, of the technical lignin from the Kraft process, just 2% is commercially used for products, such as dispersing or binding agents. Lignosulfonates from the acidic pulping process are used more widely, for more or less the same purposes.  Applications include use as a dispersant; for instance in paints. As an emulsifier it is added to asphalt for preparation of temperature-stable emulsions. Lignins or products derived thereof are applied in plastics and foams. One producer (Borregaard) produces vanillin from lignin. Nevertheless, lignin remains a very underutilized resource.  Prospects for lignin valorization In present wood treatment technologies, lignin is looked upon as a by-product; the main process is primarily intended to valorize cellulose. In principle, wood could be biorefined with an eye on valorizing the lignin component too. This would mean a complete overhaul of processes; in order to obtain lignin in an optimal functionality. Then, we could produce higher value compounds from it, including fine chemicals, building blocks for pharmaceuticals, flavouring agents and fragrances. The remaining lignin could still produce bulk chemicals and fuel additives, and finally a product that can be burnt. But in the short term, such processes are unlikely to be developed for commercial use.  It is hard to develop lignin valorization. Firstly, ‘lignin’ as such is very poorly defined. It differs chemically between species. Season, weather conditions and position in the field all influence the specific lignin makeup. On top of that, chemical treatment will also influence lignin’s molecular composition. There is a need for a databank listing all these varieties. The International Lignin Institute (ILI) gathers precisely such information. And there is a Lignin Club that unites lignin producers and customers world-wide. The club keeps score of a lignin matrix that shows what kind of lignin one will need for a specific application. And it oversees a major data base of lignin varieties and their properties. The foundation for lignin matchmaking.  Conclusion: the sector hopes that in the course of time, we will be able to produce commercial chemicals from this resource. And new applications (of rather pure lignin) may arise in the form of new materials made from the resource.'

txt = 'Prime Minister Pham Minh Chinh received European Commissioner for Environment, Oceans, and Fisheries Virginijus Sinkevičius in Ho Chi Minh City on November 28, affirming the importance Vietnam attaches to green development and the realisation of international commitments on forest, seas, oceans, and climate change fight. PM Pham Minh Chinh (R) meets with European Commissioner for Environment, Oceans, and Fisheries Virginijus Sinkevičius in HCM City on November 28. (Photo: ) HCM City – Prime Minister Pham Minh Chinh received European Commissioner for Environment, Oceans, and Fisheries Virginijus Sinkevičius in Ho Chi Minh City on November 28, affirming the importance Vietnam attaches to green development and the realisation of international commitments on forest, seas, oceans, and climate change fight. PM Chinh said as Vietnam, especially its Mekong Delta, is highly vulnerable to climate change, it has built programmes, strategies, and plans for green development. However, as a developing country with a low starting point, fairness and justice need to be guaranteed during the transition process.He called on the international community to assist Vietnam in finance, science - technology, personnel training, governance experience, and building of a legal basis that matches Vietnam’s conditions and international practices.Regarding illegal, unreported and unregulated (IUU) fishing, he said after the European Union gave recommendations, Vietnam has actively taken measures for addressing this issue.Aside from aligning legal regulations on sea and ocean protection and anti-IUU fishing with international rules, the country has increased communications to improve citizens’ awareness of and compliance with regulations, created beneficial livelihoods for people to reduce sea exploitation, and installed monitoring devices on vessels to deal with any actions of IUU fishing, the Government leader elaborated.PM Chinh expressed his hope that the EU will cooperate with Vietnam, which has more than 3,000km of coastline, in developing sea-based economic activities, including sea farming, fishing, logistics, protecting and exploiting maritime resources, and ensuring security and safety of navigation and overflight at sea.He also called on his guest to help strengthen the Vietnam - EU cooperation in an increasingly substantive and effective manner, including promoting EU countries’ ratification of the EU - Vietnam Investment Protection Agreement (EVIPA).At the meeting, Sinkevičius, who was in Vietnam to attend the Green Economy Forum & Exhibition (GEFE) 2022 held by the European Chamber of Commerce, highly valued the country’s efforts and commitments to implement green and anti-climate change policies, including the roadmap for achieving net zero emissions by 2050.Vietnam has worked actively and fruitfully with the EU in the IUU fishing combat, he noted, adding that this is a long-term issue and the country should keep striving to achieve desired results.He also asked Vietnam to continue paying attention to and coordinating with the EU in protecting and developing forests, and conserving forest and marine biodiversity, including preventing plastic waste.Agreeing with the commissioner’s opinions, PM Chinh appreciated the EU’s assistance with the IUU fishing combat, adding that Vietnam is working hard to recover forests, protect the ocean, and prevent plastic waste.The country does not pursue pure economic growth at the expense of the environment, but takes the people as the centre, player, target, momentum, and resource of development, he stated.He expressed his belief that the Vietnam - EU relations, including in fighting climate change and conserving forest, marine, and agricultural biodiversity, will become increasingly substantive and effective./."'

txt = tutil.keep_only_word_characters(
        tutil.clean_html( txt ))

words_total = len( txt.split(' ') )

padded = pad_sequences(tokenizer.texts_to_sequences( [ txt ] ), 
              maxlen=2500,
              padding='post', 
              truncating='post' )

model.predict(padded)

#%% MODEL EXPORT

model_name = 'lignin-model.hdf5'
tokenizer_name = 'lignin-tokenizer.dat'

hdf5_file = h5py.File(f'asset/{model_name}', 'w') 
hdf5_format.save_model_to_hdf5( model, hdf5_file ) 
hdf5_file.close()

pickle.dump(tokenizer, open(f'asset/{tokenizer_name}', 'wb'))

print(f'model {model_name} Exported...')


