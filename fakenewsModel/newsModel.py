import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
from collections import Counter

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop = set(stopwords.words("english"))


def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split()
                      if word.lower not in stop]
    return " ".join(filtered_words)


def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


fake_news = pd.read_csv('./Fake.csv')['text'].to_frame()
real_news = pd.read_csv('./True.csv')['text'].to_frame()

fake_news['isFake'] = [1 for text in fake_news['text']]
real_news['isFake'] = [0 for text_ in real_news['text']]

combine = [fake_news, real_news]
all_news = pd.concat(combine)

all_news["text"] = all_news.text.map(remove_stopwords)

X = all_news.drop('isFake', axis=1)
y = all_news['isFake']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, random_state=42
)

preprocessing = tf_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoding = tf_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = preprocessing(text_input)
outputs = encoding(preprocessed_text)

# Neural network layers
layer1 = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
layer2 = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(layer1)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs=[layer2])

metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=metrics)


model.fit(X_train, y_train, epochs=10)
model.save("nn.h5")
