import json
import pandas as pd
import os
import string
import nltk

import warnings
warnings.filterwarnings('ignore')
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


def run(args):
    df_train, df_test = load_data(args)
    hyper_params = {
        'learning_rate': 1e-3,  # default for Adam
        'epochs': 1000,
        'batch_size': 64,
        'layers': [64, 32],
        #'layers': [256, 128, 64, 32],
        'dim': 10000,
        'dropout_rate': 0.3,
        'checkpoint_path': os.path.join(args.result, 'model_cpk.hdf5'),
    }
    train_x, train_y, test_x, test_y = prepare_data(df_train, df_test, hyper_params)
    
    model = build(hyper_params)
    model.summary()
    
    if args.task == 'nlp_train':
        callbacks = define_callbacks(hyper_params)
        model.fit(
            train_x, 
            train_y, 
            epochs=100,
            validation_data=[test_x, test_y],
            batch_size=hyper_params['batch_size'],
            callbacks=callbacks,
        )

    elif args.task == 'nlp_test':
        model.load_weights(hyper_params['checkpoint_path'])
        loss,acc = model.evaluate(test_x,  test_y, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    elif args.task == 'nlp_cache':
        model.load_weights(hyper_params['checkpoint_path'])
        
        results = model.predict(test_x)
        output = []
        i = 0
        for vec in results:
            for idx in range(5):
                if vec[idx] == max(vec):
                    output.append([df_test['user_id'][i], df_test['review_id'][i], df_test['stars'][i], df_test['useful'][i], df_test['funny'][i], df_test['cool'][i], idx+1])
                    i += 1
                    continue

        df = pd.DataFrame(output, columns = ['user_id', 'review_id', 'stars', 'useful', 'funny', 'cool', 'sentiment'])
        df.to_csv(os.path.join(args.result,'sentiment.csv'))
        
    else:
        assert False, 'Unknown task'
        
    return model,(df_train, df_test,(train_x, train_y, test_x, test_y))

        
    
def load_data(args):
    review_data_path = os.path.join(args.data,"yelp_academic_dataset_review.json")
    column_names = get_superset_of_column_names_from_file(review_data_path)

    train_start = 1;
    train_end = args.train_size
    test_start = args.train_size+1
    test_end = args.train_size+args.test_size+1

    df_train = convert_to_dataframe(review_data_path, column_names, train_start, train_end) # for training
    df_test = convert_to_dataframe(review_data_path, column_names, test_start, test_end) # for validation

    # test
    print(df_train.head())
    print(df_test.head())
    # test

    nltk.data.path = [args.tmp]
    nltk.download('stopwords',download_dir=args.tmp)
    nltk.download('punkt',download_dir=args.tmp)
    nltk.download('wordnet',download_dir=args.tmp)

    return df_train, df_test
    

def prepare_data(df_train, df_test, hyper_params):
    # prepare data
    # 1.tokenize
    
    # test case
    test_str = "I bought several books yesterday<br /> and I really love them!"
    preprocessing(test_str)
    # test

    for df in df_train, df_test:
        df['text_prep'] = df['text'].progress_apply(preprocessing)

    # 2.vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        min_df=2, # ignore word that only appears in 1 review
        ngram_range=(1, 2), # consider both uni-gram and bi-gram
    )

    train_x = tfidf_vectorizer.fit_transform(df_train['text_prep'])
    test_x = tfidf_vectorizer.fit_transform(df_test['text_prep'])

    def convert(x):
        return int(x)-1
    
    train_y = df_train['stars'].apply(convert)
    test_y = df_test['stars'].apply(convert)

    print(train_y.value_counts())

    # 3.Dimensionality Reduction

    #DIM = 10000 # Dimensions to keep, a hyper parameter
    dim = hyper_params['dim']

    # Create a feature selector
    # By default, f_classif algorithm is used
    # Other available options include mutual_info_classif, chi2, f_regression etc. 

    selector = SelectKBest(k=dim)
    # The feature selector also requires information from labels
    # Fit on training data
    train_x = selector.fit_transform(train_x, train_y)
    test_x = selector.fit_transform(test_x, test_y)
    print(train_y.shape)

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    print(train_y.shape)
    print(test_y.shape)

    return train_x, train_y, test_x, test_y
    

def convert_to_dataframe(json_file_path, column_names, begin, end):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    
    df = pd.DataFrame(columns=column_names)
    line_counter = 0
    with open(json_file_path, "rb") as fin:
        for line in fin:
            line_counter += 1
            if line_counter < begin:
                continue
            elif line_counter in range(begin, end+1):
                line_contents = json.loads(line)
                df = df.append(line_contents, ignore_index = True)
            else:
                return df


def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""
    with open(json_file_path, "rb") as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names = line_contents.keys()
            return column_names


def preprocessing(line: str) -> str:
    # 建立translation table， 把标点符号转换成空白符
    transtbl = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # 出现的频率非常高，但对训练没什么用的词，以后可以剔除掉
    stopwords = nltk.corpus.stopwords.words('english')
    # nltk包需要提前安装， lemmatizer是已经训练好的词性还原
    lemmatizer = nltk.WordNetLemmatizer()
    
    """
    Take a text input and return the preprocessed string.
    i.e.: preprocessed tokens concatenated by whitespace
    """
    # 剔除换行符，以及标点符号
    line = line.replace('<br />', '').translate(transtbl)
    
    # list 剔除在stopwords里面的token， 'v' 是在句子中作的成分也就是动词词性还原，
    # 如果不是动词，会再做名词词性还原
    tokens = [lemmatizer.lemmatize(t.lower(),'v')
              for t in nltk.word_tokenize(line)
              if t.lower() not in stopwords]
    
    return ' '.join(tokens)


def build(hyper_params):
    model = build_model(
        input_dim=hyper_params['dim'],
        layers=hyper_params['layers'],
        output_dim=5,
        dropout_rate=hyper_params['dropout_rate'],
    )
    model.compile(
        optimizer=Adam(lr=hyper_params['learning_rate']),
        loss='categorical_crossentropy',
        #loss='binary_crossentropy',
        metrics=['acc'],
    )
    return model

    
# add an extra dropout layer after each dense layer
def build_model(input_dim, layers, output_dim, dropout_rate):
    # Input layer
    X = Input(shape=(input_dim,))
    
    # Hidden layer(s)
    H = X
    for layer in layers:
        H = Dense(layer, activation='relu')(H)
        H = Dropout(rate=dropout_rate)(H)
    
    # Output layer
    activation_func = 'softmax' if output_dim > 1 else 'sigmoid'
    
    Y = Dense(output_dim, activation=activation_func)(H)
    return Model(inputs=X, outputs=Y)


def define_callbacks(hyper_params):
    early_stopping_hook = EarlyStopping(
        # what metrics to track
        monitor='val_loss',
        # maximum number of epochs allowed without imporvement on monitored metrics 
        patience=10,
    )

    # path to store checkpoint
    CPK_PATH = hyper_params['checkpoint_path']

    model_cpk_hook = ModelCheckpoint(
        CPK_PATH,
        monitor='val_loss',
        # Only keep the best model
        save_best_only=True,
    )
    
    return [early_stopping_hook, model_cpk_hook]
