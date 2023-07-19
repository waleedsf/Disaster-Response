import sys
import pandas as pd
from sqlalchemy import *
import nltk
import re
import pickle
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import precision_score, recall_score, f1_score

nltk.download('stopwords')


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
   tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
   stop_words = set(stopwords.words("english"))
   ps = PorterStemmer()

   clean_tokens = []
   for tok in tokens:
       clean_tok = ps.stem(tok).lower().strip()
       if clean_tok.isalpha() and clean_tok not in stop_words:
          clean_tokens.append(clean_tok)
   return clean_tokens

def build_model():
    pipeline = Pipeline([
    
    ("vect" , CountVectorizer(tokenizer=tokenize)),
    ("tfidf" , TfidfTransformer()),
    ("clf" , MultiOutputClassifier(RandomForestClassifier()))

])
    
#     parameters = {
#     'clf__estimator__n_estimators': [50, 100],
#     'clf__estimator__min_samples_split': [2, 3]
# }

#     cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    evaluation = {}
    for column in Y_test.columns:
        evaluation[column] = []
        evaluation[column].append(precision_score(Y_test[column], y_pred_df[column]))
        evaluation[column].append(recall_score(Y_test[column], y_pred_df[column]))
        evaluation[column].append(f1_score(Y_test[column], y_pred_df[column]))
    print(pd.DataFrame(evaluation))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()