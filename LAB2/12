# Step 1: Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Masking, LSTM, Bidirectional, Dense

# Step 2: Conversation pairs (input-output)
pairs = [
    ('hi', 'hello'),
    ('what are you', 'i am a chatbot'),
    ('what do you do', 'i do nothing'),
    ('bye', 'goodbye')
]

# Step 3: Separate inputs and outputs
x_texts, y_texts = zip(*pairs)

# Step 4: Tokenize and prepare vocabulary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_texts + y_texts)
vocab_size = len(tokenizer.word_index) + 1

# Step 5: Convert texts to padded sequences
x_seq = tokenizer.texts_to_sequences(x_texts)
y_seq = tokenizer.texts_to_sequences(y_texts)

max_len = max(max(len(s) for s in x_seq), max(len(s) for s in y_seq))
x = pad_sequences(x_seq, maxlen=max_len)
y = pad_sequences(y_seq, maxlen=max_len)

# Step 6: One-hot encode the output
y_cat = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Step 7: Define the model
model = Sequential([
    Masking(mask_value=0, input_shape=(max_len,)),
    Embedding(input_dim=vocab_size, output_dim=64),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dense(vocab_size, activation='softmax')  # Use softmax to get word probabilities
])

# Step 8: Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y_cat, epochs=100)

# Step 9: Chat function
def chat(user_input):
    seq = tokenizer.texts_to_sequences([user_input])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)[0]
    words = [tokenizer.index_word.get(np.argmax(p), '') for p in pred if np.argmax(p) != 0]
    return ' '.join(words).strip()

# Step 10: Run chatbot
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    msg = input("You: ")
    if msg.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    print("Bot:", chat(msg) or "Sorry, I don't understand.")
