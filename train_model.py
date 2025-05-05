### ENHANCED: train_model.py (NER-safe augmentation added)

import json
import random
import numpy as np
import nltk
import string
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Load intents
with open('dataset/information_intents.json', 'r') as f:
    information_intents = json.load(f)

with open('dataset/events_intents.json', 'r') as f:
    events_intents = json.load(f)

with open('dataset/support_intents.json', 'r') as f:
    support_intents = json.load(f)

# Combine all intents
all_intents = information_intents['intents'] + events_intents['intents'] + support_intents['intents']

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word, tag):
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag[0], wordnet.NOUN)

def is_named_entity(word, text):
    doc = nlp(text)
    for ent in doc.ents:
        if word in ent.text:
            return True
    return False

def synonym_replacement(sentence, n=1):
    words = word_tokenize(sentence)
    new_words = words.copy()
    random.shuffle(new_words)
    num_replaced = 0

    for word in new_words:
        if word in string.punctuation or is_named_entity(word, sentence):
            continue
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def random_deletion(sentence, p=0.1):
    words = word_tokenize(sentence)
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return ' '.join(new_words) if new_words else random.choice(words)

def word_order_change(sentence):
    words = word_tokenize(sentence)
    if len(words) <= 3:
        return sentence
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def augment_training_data(intents, augment_factor=2):
    X_train = []
    y_train = []

    for intent in intents:
        original_patterns = intent["patterns"]
        for pattern in original_patterns:
            X_train.append(pattern)
            y_train.append(intent["tag"])

        for pattern in original_patterns:
            if len(pattern.split()) > 3:
                for _ in range(augment_factor):
                    X_train.append(synonym_replacement(pattern))
                    y_train.append(intent["tag"])

                    X_train.append(random_deletion(pattern))
                    y_train.append(intent["tag"])

                    X_train.append(word_order_change(pattern))
                    y_train.append(intent["tag"])

    return X_train, y_train

# Generate data with augmentation
X_train, y_train = augment_training_data(all_intents)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Model training with tuning
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0], 'fit_prior': [True, False]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_vec, y_train)
best_model = grid_search.best_estimator_

# Validation accuracy
val_acc = best_model.score(X_val_vec, y_val)
print(f"Validation Accuracy: {val_acc:.4f}")

# Train on all data
X_all_vec = vectorizer.fit_transform(X_train + X_val)
y_all = y_train + y_val
final_model = MultinomialNB(**grid_search.best_params_)
final_model.fit(X_all_vec, y_all)

# Save model and vectorizer
dump(final_model, 'model/chatbot_model.joblib')
dump(vectorizer, 'model/vectorizer.joblib')
print("Model training complete. Files saved.")
