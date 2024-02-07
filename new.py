import random
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = sorted(set(lemmatizer.lemmatize(word.lower()) for intent in intents['intents'] for pattern in intent['patterns'] for word in nltk.word_tokenize(pattern) if word not in ['?', '!', '.', ',']))
classes = sorted(set(intent['tag'] for intent in intents['intents']))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = [(bag, output) for intent in intents['intents'] for pattern in intent['patterns'] for word in nltk.word_tokenize(pattern) if word not in ['?', '!', '.', ','] for bag in [1 if word in lemmatizer.lemmatize(word.lower()) else 0 for word in words]]

random.shuffle(training)
training = np.array(training)

trainX, trainY = training[:, :len(words)], training[:, len(words):]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print('Done')