import wandb
from trainer import Trainer
import os

# datasets = ['forbes', 'auto_mpg', 'cars93']
datasets = ['auto_mpg', 'cars93']
# datasets = ['aaup']
query_types = ['full', 'simple']
# query_types = ['simple']
# mods = ['', 'property_corr', 'property_corr_less', 'property_corr_less_less', 'attribute_corr', 'class_corr', 'class_corr_less', 'class_corr_less_less', 'real_class_corr']
mods = ['real_class_corr']
# mods = ['real_property_corr']
not_cols = {
    'forbes': ["Market_Value","Country","Industry","Sales","Assets","label","uri","id","Company", 'label_num'],
    # 'aaup': ['State', 'College_name', 'Type', 'uri', 'id', 'label_salary', 'label', 'label_comp', 'Average_salary_full_professors', 
    #          'Average_salary_associate_professors', 'Average_salary_assistant_professors', 'Average_salary_all_ranks',
    #          'Average_compensation_full_professor', 'Average_compensation_associate_professors',
    #          'Average_compensation_assistant_professors', 'Average_compensation_all_ranks'],
    'aaup': ['State', 'College_name', 'Type', 'uri', 'id', 'label_salary', 'label', 'label_comp'],
    'auto_mpg': ['mpg_label', 'car', 'uri', 'id', 'label', 'label_num', 'horsepower'],
    'cars93': ["label","car","uri","id","Type","Midrange_Price","Minimum_Price",'Luggage_capacity', 'Rear_seat_room', 'label_num', 'Horsepower']
}

for dataset in datasets:
    print('Dataset:', dataset)
    for query_type in query_types:
        print('Query type:', query_type)
        for mod in mods:
            if query_type == 'full' and mod == '' and dataset != 'auto_mpg' and dataset != 'cars93':
                extra_mods = ['_less_props', '_no_literals', '']
            else:
                extra_mods = ['']
            print('Mod:', mod if mod != '' else 'None')

            for extra_mod in extra_mods:
                print('Extra Mod:', extra_mod if extra_mod != '' else 'None')

                if os.path.exists(f'../data/{dataset}/{query_type}_query_{mod}mod{extra_mod}/embeddings.pkl'):
                    config = {

                        'DATASET': dataset,
                        'QUERY_TYPE': query_type,
                        'GRAPH_MOD': mod + extra_mod,
                        'SCHEMA': '../data/statements.rdf',
                        'URI_DF': f'../data/{dataset}/complete_dataset.tsv',
                        'LABEL': 'label',
                        'NOT_COLS': not_cols[dataset],

                        'EMBEDDINGS': f'../data/{dataset}/{query_type}_query_{mod}mod{extra_mod}/embeddings.pkl',
                    }

                    print(config['EMBEDDINGS'])

                    if os.path.exists(config['EMBEDDINGS']):
                        # start a new wandb run to track this script
                        wandb.init(
                            # set the wandb project where this run will be logged
                            project="fine_tuning_owl_mult_trials",

                            # track hyperparameters and run metadata
                            config=config
                        )

                        if config['LABEL'] == 'label':
                            wandb.define_metric("test_f1", summary="max")
                        else:
                            wandb.define_metric("test_mse", summary="min")
                            wandb.define_metric("test_mae", summary="min")
                            wandb.define_metric("test_rmse", summary="min")


                        trainer_ = Trainer(config, wandb, config['NOT_COLS'])
                        trainer_.run()

                        wandb.finish()
                        # print()
