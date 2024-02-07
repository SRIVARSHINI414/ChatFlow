import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('/Users/srivarshinig/Documents/Project/Chatflow/intents.json').read())
words, classes, model = pickle.load(open('words.pkl', 'rb')), pickle.load(open('classes.pkl', 'rb')), load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    return [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sentence)]

def bag_of_words(sentence):
    return np.array([1 if word in clean_up_sentence(sentence) else 0 for word in words])

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [{"intent": classes[i], "probability": str(r)} for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: float(x["probability"]), reverse=True)
    return results

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

print("GO! Bot is running!")

while True:
    message = input("")
    predicted_intents = predict_class(message)
    response = get_response(predicted_intents, intents)
    print(response)