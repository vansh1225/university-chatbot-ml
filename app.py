### ENHANCED: app.py (adding ML features to chatbot)

from flask import Flask, render_template, request, jsonify
import json
import random
import datetime
import numpy as np
from joblib import load
from textblob import TextBlob
import spacy

# Load NLP model for NER
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Load trained ML models and vectorizers for different categories
general_model = load('model/chatbot_model.joblib')
general_vectorizer = load('model/vectorizer.joblib')

# Load intent files for different categories
with open('dataset/intents.json', 'r') as f:
    general_intents = json.load(f)

with open('dataset/information_intents.json', 'r') as f:
    information_intents = json.load(f)

with open('dataset/events_intents.json', 'r') as f:
    events_intents = json.load(f)

with open('dataset/support_intents.json', 'r') as f:
    support_intents = json.load(f)

CONFIDENCE_THRESHOLD = 0.6
CATEGORY_MAP = {
    "information": information_intents,
    "events": events_intents,
    "support": support_intents
}
INTENT_TO_CATEGORY = {}
for category, intents_data in CATEGORY_MAP.items():
    for intent in intents_data['intents']:
        INTENT_TO_CATEGORY[intent['tag']] = category

def get_greeting():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning! "
    elif 12 <= current_hour < 17:
        return "Good afternoon! "
    else:
        return "Good evening! "

def get_options_menu():
    return "Please select one of the following options:\n1. Information\n2. Events\n3. Support"

def predict_intent_with_confidence(user_input):
    input_vector = general_vectorizer.transform([user_input])
    intent_probabilities = general_model.predict_proba(input_vector)[0]
    max_prob_index = np.argmax(intent_probabilities)
    predicted_intent = general_model.classes_[max_prob_index]
    confidence = intent_probabilities[max_prob_index]
    return predicted_intent, confidence

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

def detect_multiple_intents(user_input):
    clauses = [clause.strip() for clause in user_input.split(" and ") if clause.strip()]
    responses = []
    for clause in clauses:
        intent, confidence = predict_intent_with_confidence(clause)
        if confidence > CONFIDENCE_THRESHOLD:
            category = INTENT_TO_CATEGORY.get(intent)
            if category:
                for i in CATEGORY_MAP[category]['intents']:
                    if i['tag'] == intent:
                        responses.append(random.choice(i['responses']))
    return responses

def chatbot_response(user_input, category=None):
    if user_input.lower() in ["menu", "options", "help"]:
        return {"response": get_options_menu()}

    if user_input.lower() in ["exit", "quit", "bye", "no", "end"]:
        return {"response": "Thank you for chatting with us. Goodbye."}

    if user_input.lower() in ["yes", "yeah", "sure", "ok", "okay"]:
        return {"response": get_options_menu()}

    if user_input in ["1", "information", "info"]:
        return {"response": "You've selected Information.", "category": "information"}
    if user_input in ["2", "events", "event"]:
        return {"response": "You've selected Events.", "category": "events"}
    if user_input in ["3", "support", "help"]:
        return {"response": "You've selected Support.", "category": "support"}

    # Apply ML features
    original_input = user_input
    user_input = correct_spelling(user_input)
    sentiment = analyze_sentiment(user_input)
    entities = extract_entities(user_input)

    multi_responses = detect_multiple_intents(user_input)
    if len(multi_responses) > 1:
        return {"response": "<br>".join(multi_responses) + "<br><br>Would you like to ask something else?"}

    predicted_intent, confidence = predict_intent_with_confidence(user_input)
    if confidence < CONFIDENCE_THRESHOLD:
        return {"response": "I'm not sure I understand. Please try again or choose a category:", "options_menu": get_options_menu()}

    if category is None and predicted_intent in INTENT_TO_CATEGORY:
        category = INTENT_TO_CATEGORY[predicted_intent]

    if category not in CATEGORY_MAP:
        return {"response": "Please choose a valid category.\n" + get_options_menu()}

    intents = CATEGORY_MAP[category]
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            response += f"\n(Sentiment: {sentiment}, Entities: {entities})"
            return {
                "response": response + "\nWould you like to ask something else?",
                "category": category,
                "intent": predicted_intent,
                "confidence": confidence
            }

    return {"response": "I'm sorry, I don't understand. Try again or select another option."}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    category = request.form.get('category', None)

    if user_input.lower() == "initial_greeting":
        return get_greeting() + "Welcome to Illinois Institute of Technology. " + get_options_menu()

    response_data = chatbot_response(user_input, category)
    if isinstance(response_data, dict):
        if 'suggestions' in response_data:
            suggestions_html = "<ul>" + "".join([f"<li>{q}</li>" for q in response_data['suggestions']]) + "</ul>"
            return response_data['response'] + suggestions_html
        elif 'options_menu' in response_data:
            return response_data['response'] + "<br><br>" + response_data['options_menu']
        else:
            return response_data['response']
    else:
        return response_data

if __name__ == '__main__':
    app.run(debug=True)
