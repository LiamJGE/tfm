import wandb

# import tfm.scripts.OWL2VecStar.owl2vec_star.owl2vec_star as owl2vec_star
from owl2vec_star import owl2vec_star
from gensim.models import KeyedVectors

import pandas as pd
import pickle
import os



# datasets = ['forbes', 'aaup', 'auto_mpg', 'cars93']
datasets = ['auto_mpg', 'cars93']
# datasets = ['forbes']
# query_types = ['full']
query_types = ['full', 'simple']
# mods = ['', 'property', 'attribute', 'attribute_corr']
# mods = ['']
mods = ['real_class_corr']

for dataset in datasets:
    print('Dataset:', dataset)
    for query_type in query_types:
        print('Query type:', query_type)
        for mod in mods:
            print('Mod:', mod if mod != '' else 'None')

            config = {

                'DATASET': dataset,
                'QUERY_TYPE': query_type,
                'GRAPH_MOD': mod,
                'SCHEMA': '../data/statements.rdf',
                'URI_DF': f'../data/{dataset}/complete_dataset.tsv',
                'LABEL': 'label',

                # 'BASIC': {'ontology_file': f'../../data/{dataset}/{query_type}_query_{mod}mod/graph.owl',
                #         'uri_file': f'../../data/{dataset}/complete_dataset.tsv',
                #         'embedding_file': f'../../data/{dataset}/'},

                'BASIC': {'ontology_file': f'../../data/{dataset}/{query_type}_query_{mod}mod/graph.owl',
                        'uri_file': f'../../data/{dataset}/complete_dataset.tsv',
                        'embedding_file': f'../../data/{dataset}/'},

                'DOCUMENT': {'cache_dir': './cache/',
                            'ontology_projection': 'yes',
                            'projection_only_taxonomy': 'no',
                            'multiple_labels': 'yes',
                            'avoid_owl_constructs': 'no',
                            'save_document': 'yes',
                            'axiom_reasoner': 'hermit',
                            #  'pre_entity_file': '../data/forbes_entities.txt',
                            'walker': 'random',
                            'walk_depth': 4,
                            'URI_Doc': 'yes',
                            'Lit_Doc': 'yes',
                            'Mix_Doc': 'yes',
                            'Mix_Type': 'random',
                            },

                'MODEL': { # Pretraining
                        'embed_size': 300, 
                        'iteration': 15, 
                        'window': 5, 
                        'min_count': 1,
                        'negative': 25, 
                        'seed': 42,
                        'sg': 1,
                        
                        # Fine tuning
                        'epoch': 100}
            }

            # if (dataset != 'forbes' or mod!=''):
                 
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="owl2vec_training",

                # track hyperparameters and run metadata
                config=config
            )

            gensim_model = owl2vec_star.extract_owl2vec_model(config['BASIC']['ontology_file'], config, True, True, True, wandb)

            output_folder="./cache/output/"

            # Gensim format
            gensim_model.save(output_folder+"ontology.embeddings")
                #Txt format
            gensim_model.wv.save_word2vec_format(output_folder+"ontology.embeddings.txt", binary=False)

        
            #Embedding vectors generated above
            model = KeyedVectors.load("./cache/output/ontology.embeddings", mmap='r')
            wv = model.wv
            df = pd.read_csv(config['BASIC']['uri_file'], sep='\t')
            uris_of_interest = list(set(df['uri'].values))
            emb_dict = {}
            for uri in uris_of_interest:
                if uri in wv.index_to_key:
                    emb_dict[uri] = wv[uri]

            config['EMBEDDINGS'] = config['BASIC']['embedding_file'] + config['QUERY_TYPE'] + '_query_' + config['GRAPH_MOD'] + 'mod/embeddings.pkl'
            if not os.path.exists(config['BASIC']['embedding_file'] + config['QUERY_TYPE'] + '_query_' + config['GRAPH_MOD'] + 'mod'):
                os.makedirs(config['BASIC']['embedding_file'] + config['QUERY_TYPE'] + '_query_' + config['GRAPH_MOD'] + 'mod')
            with open(config['BASIC']['embedding_file'] + config['QUERY_TYPE'] + '_query_' + config['GRAPH_MOD'] + 'mod/embeddings.pkl', 'wb') as f:
                    pickle.dump(emb_dict, f)

            wandb.finish()

            

            print(f'----------------- MOD {mod} DONE ----------------')
            print()
        print(f'----------------- QUERY TYPE {query_type} DONE ----------------')
        print()
    print(f'----------------- DATASET {dataset} DONE ----------------')
    print()