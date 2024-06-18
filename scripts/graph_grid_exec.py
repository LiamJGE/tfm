import graph_builder
import os


# datasets = ['forbes', 'aaup', 'auto_mpg', 'cars93']
datasets = ['auto_mpg', 'cars93']
query_types = ['full', 'simple']
# query_types = ['full']
# mods = ['', 'property', 'attribute']
mods = ['']
mods = ['real_class_corr', 'real_property_corr']

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
                'URI_DF': f'../data/{dataset}/complete_dataset_real_corr_feat.tsv',
                'LABEL': 'label',
            }

            g_builder = graph_builder.GraphBuilder(config=config)
            g_builder.run()
            print(f'----------------- MOD {mod} DONE ----------------')
            print()
        print(f'----------------- QUERY TYPE {query_type} DONE ----------------')
        print()
    print(f'----------------- DATASET {dataset} DONE ----------------')
    print()