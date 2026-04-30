from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import string
import joblib

def preProcess_text(lines):
    all_lemmatized_sentences = []
    for line in lines:
      punctuations = string.punctuation
      text = line.lower()

      tokens = word_tokenize(text)

      filteredText = []
      for token in tokens:
        if token not in stopwords.words("english"):
          filteredText.append(token)

      clean_tokens = [word for word in filteredText if word not in punctuations]
      filteredText_copy = list(filteredText)
      for word in filteredText_copy:
        if word in punctuations:
          filteredText.remove(word)

      wnet = WordNetLemmatizer()
      lemmatizeWord = []
      for word in filteredText:
        lemmatizeWord.append(wnet.lemmatize(word, "v"))

      all_lemmatized_sentences.append(lemmatizeWord)

    return all_lemmatized_sentences

def do_prediction(user_msg):
    vectorizer=joblib.load("intent_classifier_vectorizer.pkl")
    model=joblib.load("intent_classifier_model.pkl")        
    processed = preProcess_text([user_msg])
    user_vetor = vectorizer.transform([" ".join(tokens) for tokens in processed])
    predict = model.predict(user_vetor)
    return predict[0]
