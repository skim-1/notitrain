import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
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

train_content = X_train.text.to_numpy()
train_classification = y_train.to_numpy()
test_content = X_test.text.to_numpy()
test_classification = y_test.to_numpy()

uwc = len(counter_word(all_news.text))
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=uwc)
tokenizer.fit_on_texts(train_content)

train_sequences = tokenizer.texts_to_sequences(train_content)
test_sequences = tokenizer.texts_to_sequences(test_content)

max_length = 2500

train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences, maxlen=max_length, padding='post', truncating='post'
)

test_padded = tf.keras.preprocessing.sequence.pad_sequences(
    test_sequences, maxlen=max_length, padding='post', truncating='post'
)

fn_model = tf.keras.models.Sequential()

fn_model.add(tf.keras.layers.Embedding(uwc, 32, input_length=max_length))
fn_model.add(tf.keras.layers.LSTM(64, dropout=0.1))
fn_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

fn_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

fn_model.fit(train_padded, train_classification, epochs=20,
             validation_data=(test_padded, test_classification), verbose=1)

predictions = fn_model.predict(train_padded)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]

print(train_classification[10:20])
print(binary_predictions[10:20])
