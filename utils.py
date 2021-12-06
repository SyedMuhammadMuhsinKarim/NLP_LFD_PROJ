import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def dependency():
    sp = spacy.load('en_core_web_sm')
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    return sp

class DataCleaner:
    def __init__(self):
        self.sp = dependency()
        
    def remove_stop_words(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        return " ".join(filtered_sentence)

    def remove_punctuation(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        return " ".join(tokenizer.tokenize(text))

    def roots_words_stem(self, text):
        stemmer = SnowballStemmer("english")
        text = " ".join([stemmer.stem(word.lower()) for word in text.split()])    
        return text
        
    def data_cleaning(self, text):
        text = text.lower() # lowercase/'
        text = re.sub(r'https?:\/\/.\S+', "", text) # remove links
        text = re.sub(r'[0-9]+', "", text) # remove numbers
        text = self.roots_words_stem(text)
        text = self.remove_stop_words(text)
        text = self.remove_punctuation(text)
        return text

def model_metrics(model, test_x, test_y , train_y, train_x):
    pred = model.predict(test_x)
    pred_train = model.predict(train_x)

    accuracy = accuracy_score(test_y, pred)
    accuracy_train = accuracy_score(train_y, pred_train)

    precision = precision_score(test_y, pred, average='macro')
    precision_train = precision_score(train_y, pred_train, average='macro')

    recall = recall_score(test_y, pred, average='macro')
    recall_train = recall_score(train_y, pred_train, average='macro')

    f1 = f1_score(test_y, pred, average='macro')
    f1_train = f1_score(train_y, pred_train, average='macro')

    cm = confusion_matrix(test_y, pred)
    cm_train = confusion_matrix(train_y, pred_train)

    test_score = model.score(test_x, test_y)
    train_score = model.score(train_x, train_y)

    senstivity = cm[0][0] / (cm[0][0] + cm[0][1])
    specificity = cm[1][1] / (cm[1][0] + cm[1][1])

    senstivity_train = cm_train[0][0] / (cm_train[0][0] + cm_train[0][1])
    specificity_train = cm_train[1][1] / (cm_train[1][0] + cm_train[1][1])
    
    matrics_parameters = {
        "accuracy": {"test": accuracy, 'train': accuracy_train},
        "precision": {"test": precision, 'train': precision_train},
        "recall": {"test": recall, 'train': recall_train},
        "f1": {"test": f1, 'train': f1_train},
        "sensitivity": {"test": senstivity, 'train': senstivity_train},
        "specificity": {"test": specificity, 'train': specificity_train},
        "score": {"test": test_score, 'train': train_score}
    }

    return pd.DataFrame(matrics_parameters).T * 100

def ploting(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Category {topic_idx +1}", fontdict={"fontsize": 20})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

def topic_graph(features, components, words, data_set):
  tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=features)
  tfidf = tfidf_vectorizer.fit_transform(data_set)
  nmf = NMF(n_components=components, random_state=1, alpha=0.1, l1_ratio=0.5).fit(tfidf)
  tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
  ploting(nmf, tfidf_feature_names, words, "Topics Graph")

class NamedEntityRecognition:
    def __init__(self):
        self.nlp = dependency()

    def get_named_entities(self, text):
        doc = self.nlp(text)
        return [(X.text, X.label_) for X in doc.ents]
    
    def pos_tagging(self, text):
        text = [(token.text, token.pos_ )for token in self.nlp(text)]
        return text
    
    def get_token_named_entities(self, text):
        doc = self.nlp(text)
        return [(X, X.ent_iob_, X.ent_type_) for X in doc]
    
    def get_render_text(self, text):
        doc = self.nlp(text)
        sentences = [x for x in doc.sents]
        return displacy.render(sentences, style='ent')
    
    def get_render_text_dep(self, text):
        doc = self.nlp(text)
        sentences = [x for x in doc.sents]
        return displacy.render(sentences, style='dep', options = {'distance': 120})