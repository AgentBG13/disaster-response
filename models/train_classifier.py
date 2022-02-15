import sys
import pandas as pd
import nltk
import re
import pickle
from sqlalchemy import create_engine
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM disaster_response_message', con=engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
   
    return X, y, category_names



def tokenize(text):
    text = re.sub(r'[^0-9A-Za-z]', ' ', text).lower()
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tf-idf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, class_weight="balanced")))])
    parameters = {'clf__estimator__n_estimators': [200, 250],
              'clf__estimator__max_depth': [100, 150],
              'clf__estimator__min_samples_split': [20, 25],
              'clf__estimator__min_samples_leaf': [5, 10]
             }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring='f1_weighted', n_jobs=-1)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


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