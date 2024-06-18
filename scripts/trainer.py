import pandas as pd
import numpy as np
import pickle
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV


import tensorflow as tf
from tensorflow.keras import layers, callbacks

from wandb.keras import WandbCallback, WandbMetricsLogger
import wandb

class Trainer():
    def __init__(self, config, wandb, not_cols):
        self.config = config
        self.wandb = wandb
        self.df = pd.read_csv(config['URI_DF'], sep='\t')
        self.label = config['LABEL']
        self.embeddings = self.load_embeddings(config['EMBEDDINGS'])
        self.embeddings_df = ''
        self.joined_df = self.join_on_uri()
        self.SEED = 42
        self.not_cols = not_cols
        self.num_trials = 10

    def run(self):
        if self.config['DATASET'] == 'forbes':
            self.map_industry_forbes()

        self.joined_df.columns = self.joined_df.columns.astype(str)
        self.joined_df.dropna(inplace=True)
        self.df = self.df[[col for col in self.df.columns if col not in self.not_cols or 'label' in col]]
        self.df.dropna(inplace=True)

        X, y = self.split_X_y(not_x_cols=self.not_cols)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_train_test_val(X, y)

        if self.label == 'label':
            self.train_nn_classification(X_train, X_val, X_test, y_train, y_val, y_test)
            # self.train_svc_classification(X_train, X_val, X_test, y_train, y_val, y_test)
        else:
            self.train_nn_regression(X_train, X_val, X_test, y_train, y_val, y_test)
            # self.train_svr_regression(X_train, X_val, X_test, y_train, y_val, y_test)

        tsne_df = self.calculate_tsne(not_cols=self.not_cols) 

        self.tsne_plot(tsne_df, colour_col='label')

        if self.config['DATASET'] == 'forbes':
            self.tsne_plot(tsne_df, colour_col='Industry')

    def load_embeddings(self, dir):
        with open(dir, 'rb') as f:
            embeddings = pickle.load(f)
            
        return embeddings

    def join_on_uri(self):
        print(type(self.embeddings))
        if 'tsne' in self.config['EMBEDDINGS']:
            self.embeddings_df = self.embeddings.transpose()
        else:
            self.embeddings_df = pd.DataFrame.from_dict(self.embeddings, orient='index')
        
        self.embeddings_df.index.name = 'uri'
        self.embeddings_df.reset_index(inplace=True)
        return self.df.merge(self.embeddings_df, on='uri')
    
    def map_industry_forbes(self):
        industry_mapping = {
            'Regional Banks': 'Financial Services',
            'Oil & Gas Operations': 'Energy & Utilities',
            'Telecommunications services': 'Technology & Telecommunications',
            'Major Banks': 'Financial Services',
            'Electric Utilities': 'Energy & Utilities',
            'Real Estate': 'Real Estate & Construction',
            'Diversified Metals & Mining': 'Manufacturing & Industrial',
            'Investment Services': 'Financial Services',
            'Food Processing': 'Consumer Goods & Retail',
            'Construction Services': 'Real Estate & Construction',
            'Electronics': 'Technology & Telecommunications',
            'Pharmaceuticals': 'Healthcare & Pharmaceuticals',
            'Diversified Insurance': 'Financial Services',
            'Specialized Chemicals': 'Manufacturing & Industrial',
            'Semiconductors': 'Technology & Telecommunications',
            'Diversified Chemicals': 'Manufacturing & Industrial',
            'Conglomerates': 'Miscellaneous',
            'Iron & Steel': 'Manufacturing & Industrial',
            'Oil Services & Equipment': 'Energy & Utilities',
            'Broadcasting & Cable': 'Entertainment & Leisure',
            'Beverages': 'Consumer Goods & Retail',
            'Life & Health Insurance': 'Financial Services',
            'Aerospace & Defense': 'Manufacturing & Industrial',
            'Computer Services': 'Technology & Telecommunications',
            'Airline': 'Miscellaneous',
            'Auto & Truck Parts': 'Miscellaneous',
            'Medical Equipment & Supplies': 'Healthcare & Pharmaceuticals',
            'Auto & Truck Manufacturers': 'Miscellaneous',
            'Construction Materials': 'Real Estate & Construction',
            'Specialty Stores': 'Consumer Goods & Retail',
            'Other Industrial Equipment': 'Miscellaneous',
            'Food Retail': 'Consumer Goods & Retail',
            'Business & Personal Services': 'Real Estate & Construction',
            'Household/Personal Care': 'Real Estate & Construction',
            'Other Transportation': 'Miscellaneous',
            'Heavy Equipment': 'Manufacturing & Industrial',
            'Software & Programming': 'Technology & Telecommunications',
            'Railroads': 'Manufacturing & Industrial',
            'Communications Equipment': 'Technology & Telecommunications',
            'Electrical Equipment': 'Technology & Telecommunications',
            'Department Stores': 'Miscellaneous',
            'Computer Hardware': 'Technology & Telecommunications',
            'Trading Companies': 'Miscellaneous',
            'Consumer Financial Services': 'Miscellaneous',
            'Natural Gas Utilities': 'Energy & Utilities',
            'Healthcare Services': 'Healthcare & Pharmaceuticals',
            'Tobacco': 'Consumer Goods & Retail',
            'Apparel/Footwear Retail': 'Consumer Goods & Retail',
            'Consumer Electronics': 'Consumer Goods & Retail',
            'Biotechs': 'Healthcare & Pharmaceuticals',
            'Property & Casualty Insurance': 'Financial Services',
            'Diversified Utilities': 'Energy & Utilities',
            'Apparel/Accessories': 'Miscellaneous',
            'Managed Health Care': 'Financial Services',
            'Restaurants': 'Consumer Goods & Retail',
            'Paper & Paper Products': 'Manufacturing & Industrial',
            'Computer & Electronics Retail': 'Consumer Goods & Retail',
            'Containers & Packaging': 'Manufacturing & Industrial',
            'Printing & Publishing': 'Manufacturing & Industrial',
            'Hotels & Motels': 'Entertainment & Leisure',
            'Casinos & Gaming': 'Entertainment & Leisure',
            'Computer Storage Devices': 'Technology & Telecommunications',
            'Home Improvement Retail': 'Consumer Goods & Retail',
            'Air Courier': 'Manufacturing & Industrial',
            'Discount Stores': 'Consumer Goods & Retail',
            'Internet & Catalog Retail': 'Consumer Goods & Retail',
            'Advertising': 'Miscellaneous',
            'Business Products & Supplies': 'Miscellaneous',
            'Aluminum': 'Manufacturing & Industrial',
            'Drug Retail': 'Consumer Goods & Retail',
            'Security Systems': 'Manufacturing & Industrial',
            'Recreational Products': 'Entertainment & Leisure',
            'Thrifts & Mortgage Finance': 'Financial Services',
            'Household Appliances': 'Consumer Goods & Retail',
            'Environmental & Waste': 'Manufacturing & Industrial',
            'Precision Healthcare Equipment': 'Manufacturing & Industrial',
            'Trucking': 'Manufacturing & Industrial',
            'Rental & Leasing': 'Manufacturing & Industrial',
            'Furniture & Fixtures': 'Manufacturing & Industrial'
        }
        
        self.joined_df['Industry'] = self.joined_df['Industry'].map(industry_mapping)

    def encode_cats(self, cols):
        self.joined_df = pd.get_dummies(self.joined_df, columns=cols)
    
    def split_X_y(self, not_x_cols):

        if 'emb' not in not_x_cols:
            X, y = self.joined_df[[col for col in self.joined_df.columns if col not in not_x_cols]], self.joined_df[self.label]
            X_emb =  X[[col for col in X.columns if col in [str(i) for i in range(0, self.embeddings_df.shape[1])]]]
            print('Embedding dim:', str(len(X_emb)))
            X_no_emb = X[[col for col in X.columns if col not in [str(i) for i in range(0, self.embeddings_df.shape[1])]]]

            # Initialize the scaler
            if X_no_emb.shape[1] > 0:
                scaler = StandardScaler()
                X_no_emb = scaler.fit_transform(X_no_emb)
                X = np.concatenate((X_emb.to_numpy(), X_no_emb), axis=1)
            else:
                X = X_emb
        # X = X_no_emb
        else:
            X, y = self.df[[col for col in self.df.columns if col not in not_x_cols]], self.df[self.label]
            print(X.columns)


        # Convert string labels to numeric values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        return X, y

    def split_train_test_val(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=self.SEED)
        X_train = np.concatenate([X_train]*10)
        y_train = np.concatenate([y_train]*10)

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_nn_classification(self, X_train, X_val, X_test, y_train, y_val, y_test):

        # Define number of classes
        num_classes = len(self.df['label'].unique())  # Adjust based on your dataset
        print('Number of classes:', num_classes)

        all_f1_scores = []

        for i in range(self.num_trials):
            # Create a sequential model for classification with regularization
            model_ = tf.keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dropout(0.6),
                layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(num_classes, activation='softmax')
            ])

            # Compile the model
            model_.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            # Define early stopping callback
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)

            # Train the model with early stopping
            history = model_.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=4, epochs=50,
                                callbacks=[early_stopping, WandbMetricsLogger()])

            # Evaluate the model
            loss, accuracy = model_.evaluate(X_test, y_test)
            print(f'Test Accuracy: {accuracy}')

            # Predict classes for test data
            y_pred = model_.predict(X_test)
            y_pred_classes = y_pred.argmax(axis=1)

            # Compute F1 score
            f1 = f1_score(y_test, y_pred_classes, average='weighted')
            all_f1_scores.append(f1)
        print(f'Weighted F1 Score: {sum(all_f1_scores)/len(all_f1_scores)}')

        wandb.log({'test_f1':sum(all_f1_scores)/len(all_f1_scores)})

        # conf_matrix = confusion_matrix(y_test, y_pred_classes)
        # print(conf_matrix)
    
    def train_svc_classification(self, X_train, X_val, X_test, y_train, y_val, y_test):
        # Define number of classes
        num_classes = len(self.df['label'].unique())  # Adjust based on your dataset
        print('Number of classes:', num_classes)

        all_f1_scores = []

        for i in range(self.num_trials):
            # Create a pipeline with a standard scaler and SVC
            model_ = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1, probability=True))

            # Train the model
            model_.fit(X_train, y_train)

            # Evaluate the model
            accuracy = model_.score(X_test, y_test)
            print(f'Test Accuracy: {accuracy}')

            # Predict classes for test data
            y_pred = model_.predict(X_test)

            # Compute F1 score
            f1 = f1_score(y_test, y_pred, average='weighted')
            all_f1_scores.append(f1)

        print(f'Weighted F1 Score: {sum(all_f1_scores)/len(all_f1_scores)}')

        wandb.log({'test_f1': sum(all_f1_scores) / len(all_f1_scores)})


    def train_nn_regression(self, X_train, X_val, X_test, y_train, y_val, y_test):

        all_mse_scores = []
        all_mae_scores = []
        all_rmse_scores = []

        for i in range(self.num_trials):
            # Create a sequential model for regression with regularization
            model_ = tf.keras.Sequential([
                layers.Dense(4, activation='relu', input_shape=(X_train.shape[1],)),
                # layers.Dropout(0.6),
                # layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                # layers.Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(1)  # Output layer for regression
            ])

            # Compile the model
            model_.compile(optimizer='adam',
                        loss='mean_squared_error')

            # Define early stopping callback
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=2)

            # Train the model with early stopping
            history = model_.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=50,
                                callbacks=[early_stopping, WandbMetricsLogger()])

            # Evaluate the model
            loss = model_.evaluate(X_test, y_test)
            print(f'Test Mean Squared Error: {loss}')

            # Predict values for test data
            y_pred = model_.predict(X_test)

            # Compute Mean Squared Error
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            all_mse_scores.append(mse)
            all_mae_scores.append(mae)
            all_rmse_scores.append(rmse)
        print(f'Mean Squared Error: {sum(all_mse_scores)/len(all_mse_scores)}')

        wandb.log({'test_mse':sum(all_mse_scores)/len(all_mse_scores)})
        wandb.log({'test_mae':sum(all_mae_scores)/len(all_mae_scores)})
        wandb.log({'test_rmse':sum(all_rmse_scores)/len(all_rmse_scores)})

    def train_svr_regression(self, X_train, X_val, X_test, y_train, y_val, y_test):

        all_mse_scores = []

        for i in range(self.num_trials):
            # Define the SVR model with a grid search to find the best hyperparameters
            svr = SVR()
            param_grid = {
                'kernel': ['linear']
            }
            
            # Perform Grid Search with cross-validation
            grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_svr = grid_search.best_estimator_
            
            # Evaluate the model on the validation set
            y_val_pred = best_svr.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            print(f'Validation Mean Squared Error: {val_mse}')
            
            # Evaluate the model on the test set
            y_test_pred = best_svr.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            print(f'Test Mean Squared Error: {test_mse}')
            
            all_mse_scores.append(test_mse)

        # Calculate and log the mean test MSE
        mean_test_mse = sum(all_mse_scores) / len(all_mse_scores)
        print(f'Mean Test Mean Squared Error: {mean_test_mse}')
        wandb.log({'test_mse': mean_test_mse})

    def calculate_tsne(self, not_cols):
        # if self.config['DATASET'] == 'forbes':
        #     tsne_df = self.joined_df[[col for col in self.joined_df.columns if col not in not_cols and 'Industry' not in str(col) and 'Country' not in str(col)]]
        # else:
        #     tsne_df = self.joined_df[[col for col in self.joined_df.columns if col not in not_cols]]

        # Step 1: Standardize the data
        # scaler = StandardScaler()
        # scaled_data = scaler.fit_transform(tsne_df)
        X = self.joined_df[[col for col in self.joined_df.columns if col not in not_cols]]
        X_emb = X[[col for col in X.columns if col in [str(i) for i in range(0, self.embeddings_df.shape[1])]]]

        # Step 4: Perform PCA with the chosen number of components
        tsne = TSNE()
        tsne_result = tsne.fit_transform(X_emb)

        # Step 5: Create a DataFrame with PCA results
        # tsne_df = pd.DataFrame(data=tsne_result, columns=[f'PC{i}' for i in range(1, num_components + 1)])
        tsne_df = pd.DataFrame(data=tsne_result)

        # Concatenate with 'Market_Value' column from the original DataFrame
        tsne_df['label'] = self.joined_df['label']
        if self.config['DATASET'] == 'forbes':
            tsne_df['Industry'] = self.joined_df['Industry']

        return tsne_df
    
    def tsne_plot(self, tsne_df, colour_col):
        plt.figure(figsize=(10, 6))
        for name, subgroup in tsne_df.groupby(by=colour_col):
            plt.scatter(subgroup[subgroup.columns[0]], subgroup[subgroup.columns[1]], label=name)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'tSNE Scatter Plot (Colored by {colour_col})')
        plt.legend(title=colour_col)
        plt.grid(True)

        figure = plt.gcf()
        
        # Log image to wandb
        wandb.log({'TSNE ' + colour_col: wandb.Image(figure)})
        # plt.show()