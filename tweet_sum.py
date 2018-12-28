from __future__ import division
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
import numpy as np
import nltk
import pickle
import random
import argparse
import csv
import os
import spacy
from numpy import dot
from numpy.linalg import norm
import re

#spacy_nlp = spacy.load('en_core_web_lg')
glove_embeddings = None
with open("/Users/Odie/Desktop/RNNTwitter/glove/glove.twitter.27B.200d.txt", "rb") as lines:
    glove_embeddings = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}
print("LOADED GLOVE VECTORS TWEET-SUM")

allWords = []

MAX_TWEET_LEN = 20
EMBEDDING_SIZE = len(glove_embeddings.itervalues().next()) #200


# p is the index into our training data for where we are now.
p = 0
accuracy_print_interval = 20

# number of neurons in each layer
input_num_units = 0
hidden_num_units = 1500
hidden_num_units_rnn = 64
output_num_units = MAX_TWEET_LEN

epochs = 121
batch_size = 32
learning_rate = 0.001

def removePunctuation(text):
    '''
    Removes punctuation, changes to lower case and strips leading and trailing
    spaces.

    Args:
        text (str): Input string.

    Returns:
        (str): The cleaned up string.
    '''
    a=0
    while(a==0):
        if(text[0]==' '):
            text=text[1:]
        else:
            a=1
    while(a==1):
        if(text[-1]==' '):
            text=text[0:-1]
        else:
            a=0
    text=text.lower()
    return re.sub('[^0-9a-zA-Z ]', ' ', text)

def preprocess_text(txt):
    # Set to lower case
        txt = txt.lower()

        # Remove mentions
        txt = re.sub('\B@[a-z0-9_-]+', '<USER_MENTION>', txt)

        # Rempve hashtags
        #txt = re.sub('(?:(?<=\s)|^)#(\w*[A-Za-z_]+\w*)', '<HASHTAG>', txt)

        # Remove Url
        txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', txt)

        # Remove punctuation
        txt = removePunctuation(txt)

        #TODO: USE STANFORD TWITTER GLOVE AND EMOJI

        return txt


def load_tweets_and_summaries(path="Extended_Labelled_Data.csv", TextVecorizationLevel="sentence"):
    '''
    Loads texts and their summaries into separate parallel arrays from a specified
    folder. The specified folder should contain .txt files with the summary on
    the first line and the text on subsequent ones, or another folder following
    the same specified format.

    @param path path to csv file with tweets
    @return Tuple (text_list, summary_list)
    '''
    texts = []
    summaries = []

    # Load texts
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        i = 1
        for row in reader:
            tweet = {}
            texts.append(row[0])
            summaries.append(row[2])
            i+=1
            if(i >= 250):
                break

    # Preprocess text
    # tweet at idx 19 has emoji
    #print(texts[19])
    texts = [preprocess_text(t) for t in texts]
    summaries = [preprocess_text(s) for s in summaries]
    #print(texts[19])

    return (texts, summaries)

def extract_sentences_from_paragraph(paragraph):
    sentences = paragraph.split(".")
    return [s for s in sentences if len(s.strip()) != 0]

def embedding_for_word(word_text):
    '''
    For a given word, this method returns the relevant computed vector using
    the simple TensorFlow nn implementation.

    @param word_text: Word to fetch vector for.
    @return: vector representation of word.
    '''
    # Load saved word vectors
    global glove_embeddings

    if(word_text in glove_embeddings):
        return glove_embeddings[word_text]
    # If word doesnt have vector, return empty vector with zeros with same dimensionality as vector
    else:
        return np.zeros(shape=(len(glove_embeddings.itervalues().next())))

def vectorize_texts_and_summaries(texts, summaries):
    '''
    Takes a list of texts and a list of summaries and vectorizes them, returning
    two parallel lists of the vectorized texts and summaries. Texts are
    vectorised such that the word embedding for each word in the sentence is
    inferred and all the constituent embeddings are concatenated up until MAX_TWEET_LEN
    words. If a text has less than MAX_TWEET_LEN it is padded with zeros. This
    brings setence vectors to be of shape MAX_TWEET_LEN-times-EMBEDDING_SIZE.
    Summaries arent vectorised so to speak. A target is created for each sentence
    in the text based on the summary. This target is whether or not a word in the
    sentence appears in the summary. This target is vectorised. As such in the returned
    summaries matrix, we have vectors of size MAX_TWEET_LEN with 1s and 0s.

    Each sntence in a text has the same target (the text summary)
    which is returned in the parallel summaries vectors list

    @param texts List<List<String>> a list containing another list of Strings
        where each string in the nested list represents a sentence from the text.
        (Another way to think about it is a list of sentences where a sentence is List<String>)
    @param summaries List<String> a list parallel to @param<texts> which contains
        a string summary of the texts.
    @return Tuple(np.array((len(texts), MAX_TWEET_LEN, EMBEDDING_SIZE)), np.zeros((len(summaries), MAX_TWEET_LEN)))

    '''
    global MAX_TWEET_LEN
    nested_list_len = lambda x: sum(len(list) for list in x)
    source_text_vectors = np.zeros((len(texts), MAX_TWEET_LEN, EMBEDDING_SIZE))
    target_summary_vectors = np.zeros((len(summaries), MAX_TWEET_LEN))
    vec_idx = 0

    for i in xrange(len(texts)):
        text = texts[i]
        summary = summaries[i]
        sentence_vector = np.zeros((MAX_TWEET_LEN, EMBEDDING_SIZE),dtype=np.float32)
        target_vector = np.zeros((MAX_TWEET_LEN))
        idx = 0
        for w in text.split(" "):
            if(idx < MAX_TWEET_LEN):
                sentence_vector[idx] = embedding_for_word(w)
                target_vector[idx] = int(w in summary)
                idx += 1
            else:
                break
        while(idx < MAX_TWEET_LEN):
            sentence_vector[idx] = np.zeros(EMBEDDING_SIZE)
            target_vector[idx] = 0
            idx += 1
        source_text_vectors[vec_idx] = sentence_vector
        target_summary_vectors[vec_idx] = target_vector
        vec_idx+=1

    return (source_text_vectors, target_summary_vectors)

def vectorize_text(text_string):
    '''
    Takes a sentece and vectorizes it such that the word embedding for each word
    in the embedding is inferred and all the constituent embeddings are
    concatenated up until MAX_TWEET_LEN
    words. If a string has less than MAX_TWEET_LEN it is padded with zeros

    @param String Text to be vectorized
    @return np.array of shape (MAX_TWEET_LEN-times-EMBEDDING_SIZE)

    '''
    global MAX_TWEET_LEN
    global EMBEDDING_SIZE

    sentence_vector = np.zeros((MAX_TWEET_LEN, EMBEDDING_SIZE),dtype=np.float32)
    idx = 0
    for w in text_string.split(" "):
        if(idx < MAX_TWEET_LEN):
            sentence_vector[idx] = embedding_for_word(w)
            idx += 1
        else:
            break
    while(idx < MAX_TWEET_LEN):
        sentence_vector[idx] = np.zeros(EMBEDDING_SIZE)
        idx += 1
    return sentence_vector

def next_batch(all_inputs, all_outputs):
    '''
    Iterates in batches through the dataset, given as parallel arrays
    all_inputs and all_outputs, wrapping around as needed.

    @param all_inputs  np.array((batch_size, input_num_units)) sentences from texts
    @param all_outputs np.array((batch_size, input_num_units)) summaries for texts
    @return batch to be used by TensorFlow model
    '''
    global p, MAX_TWEET_LEN, EMBEDDING_SIZE

    batch_x = np.zeros((batch_size, MAX_TWEET_LEN, EMBEDDING_SIZE))
    batch_y = np.zeros((batch_size, MAX_TWEET_LEN))

    i = 0
    while(i < batch_size):
        batch_x[i] = all_inputs[p]
        batch_y[i] = all_outputs[p]
        p += 1
        i += 1
        # Wrap back around if reached end
        if(p  >= len(all_outputs)):
            p = 0

    return (batch_x, batch_y)

def shuffle_data(inputs, outputs):
    rng_state = np.random.get_state()
    np.random.shuffle(inputs)
    np.random.set_state(rng_state)
    np.random.shuffle(outputs)

def classes_from_sigmoid_probs(sigmoid_probs, threshold=0.6):
    sigmoid_probs = tf.cast(sigmoid_probs, tf.float32)
    threshold = float(threshold)
    return tf.cast(tf.greater(sigmoid_probs, threshold), tf.float32)

def train(all_inputs, all_outputs):
    global learning_rate
    global batch_size
    global epochs
    global input_num_units
    global hidden_num_units, hidden_num_units_rnn
    global output_num_units
    global accuracy_print_interval
    global glove_embeddings
    global EMBEDDING_SIZE, MAX_TWEET_LEN

    # tf Graph input. None means that the first dimension can be of any size so it represents the batch size
    x = tf.placeholder(tf.float32, [None, MAX_TWEET_LEN, EMBEDDING_SIZE])
    y = tf.placeholder(tf.float32, [None, MAX_TWEET_LEN])
    keep_prob = tf.placeholder(tf.float32)

    # Attention mechanism
    rnn_outputs, _ = bi_rnn(tf.nn.rnn_cell.LSTMCell(hidden_num_units_rnn),tf.nn.rnn_cell.LSTMCell(hidden_num_units_rnn), inputs=x, sequence_length=None, dtype=tf.float32)
    fw_outputs, bw_outputs = rnn_outputs

    W = tf.Variable(tf.random_normal([hidden_num_units_rnn], stddev=0.1))
    H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
    M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

    alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, hidden_num_units_rnn]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, MAX_TWEET_LEN)))  # batch_size x seq_len
    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_TWEET_LEN, 1]))
    r = tf.squeeze(r,2)
    h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

    # Dropout
    h_drop = tf.nn.dropout(h_star, keep_prob)

    input_num_units = hidden_num_units_rnn

    # define weights and biases of the neural network
    weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units])),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units]))
    }

    biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units])),
    'output': tf.Variable(tf.random_normal([output_num_units]))
    }

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    # Now create our neural networks computational graph
    # Wx.Wh + Bh
    hidden_layer = tf.add(tf.matmul(h_drop, weights['hidden']), biases['hidden'])
    #activation function to hidden layer calculation
    hidden_layer = tf.nn.relu(hidden_layer)
    # Output calc
    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

    # probabilities
    pred_probs = tf.sigmoid(output_layer)

    # Define loss and optimizer
    # sigmoid_cross_entropy_with_logits is used because its a multi label problem
    # (i.e. a input can have a target of more than one class eg [1,1,0] instead of just [0,1,0])
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=y), axis=1)
    cost = tf.reduce_mean(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(classes_from_sigmoid_probs(pred_probs), tf.round(y))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            total_cost = 0
            num_batches = int(len(all_outputs)/batch_size)
            for i in range(num_batches):
                batch_x, batch_y = next_batch(all_inputs, all_outputs)
                c = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
                #print(sess.run(pred_probs, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0}))
                #print(sess.run(batch_y, feed_dict={x: batch_x, y: batch_y}))
                total_cost += c
            avg_cost = total_cost/num_batches
            print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
            if(epoch % accuracy_print_interval == 0):
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
                print "Training Accuracy: ", "{:.5f}".format(acc)
            shuffle_data(all_inputs, all_outputs)
        print "\nTraining complete!"
        save_path = saver.save(sess, "Serial/tweet_sum.ckpt")
        print("Model saved in file: %s" % save_path)

        text = "I would like to stop wheezing now"
        unit_batch_x = np.zeros((1, MAX_TWEET_LEN, EMBEDDING_SIZE))
        unit_batch_x[0] = vectorize_text(text)
        results = sess.run(tf.round(pred_probs), feed_dict={x: unit_batch_x, keep_prob:1.0})
        attn = sess.run(alpha, feed_dict={x: unit_batch_x, keep_prob:1.0})
        print(results)
        for res in results:
            print( " ".join([word for idx,word in enumerate(text.split(" ")) if(res[idx] == 1)]) )
        print("\n{}\n".format(attn[0]))

def test(batch_x):
    global learning_rate
    global batch_size
    global epochs
    global input_num_units
    global hidden_num_units, hidden_num_units_rnn
    global output_num_units
    global accuracy_print_interval
    global glove_embeddings
    global EMBEDDING_SIZE, MAX_TWEET_LEN

    # tf Graph input. None means that the first dimension can be of any size so it represents the batch size
    x = tf.placeholder(tf.float32, [None, MAX_TWEET_LEN, EMBEDDING_SIZE])
    y = tf.placeholder(tf.float32, [None, MAX_TWEET_LEN])
    keep_prob = tf.placeholder(tf.float32)

    # Attention mechanism
    rnn_outputs, _ = bi_rnn(tf.nn.rnn_cell.LSTMCell(hidden_num_units_rnn),tf.nn.rnn_cell.LSTMCell(hidden_num_units_rnn), inputs=x, sequence_length=None, dtype=tf.float32)
    fw_outputs, bw_outputs = rnn_outputs

    W = tf.Variable(tf.random_normal([hidden_num_units_rnn], stddev=0.1))
    H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
    M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

    alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, hidden_num_units_rnn]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, MAX_TWEET_LEN)))  # batch_size x seq_len
    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_TWEET_LEN, 1]))
    r = tf.squeeze(r,2)
    h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

    # Dropout
    h_drop = tf.nn.dropout(h_star, keep_prob)

    input_num_units = hidden_num_units_rnn

    # define weights and biases of the neural network
    weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units])),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units]))
    }

    biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units])),
    'output': tf.Variable(tf.random_normal([output_num_units]))
    }

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    # Now create our neural networks computational graph
    # Wx.Wh + Bh
    hidden_layer = tf.add(tf.matmul(h_drop, weights['hidden']), biases['hidden'])
    #activation function to hidden layer calculation
    hidden_layer = tf.nn.relu(hidden_layer)
    # Output calc
    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

    # probabilities
    pred_probs = tf.sigmoid(output_layer)

    # Define loss and optimizer
    # sigmoid_cross_entropy_with_logits is used because its a multi label problem
    # (i.e. a input can have a target of more than one class eg [1,1,0] instead of just [0,1,0])
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=y), axis=1)
    cost = tf.reduce_mean(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(classes_from_sigmoid_probs(pred_probs), tf.round(y))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        save_path = "Serial/tweet_sum.ckpt"
        saver.restore(sess, save_path)

        return sess.run(classes_from_sigmoid_probs(pred_probs), feed_dict={x: batch_x, keep_prob:1.0})

def param_search(learning_rates=[0.01], hidden_layer_sizes=[1500]):
    global learning_rate
    global batch_size
    global epochs
    global input_num_units
    global hidden_num_units
    global output_num_units
    global accuracy_print_interval

    t,s = load_tweets_and_summaries()
    inputs,outputs = vectorize_texts_and_summaries(t, s)

    for lr in learning_rates:
        for size in hidden_layer_sizes:
            learning_rate = lr
            hidden_num_units = size
            print("\n------- [PARAM SEARCH] --------")
            print("LEARNING RATE: {}".format(lr))
            print("HIDDEN LAYER SIZE: {}".format(size))
            print("-------------------------------\n")
            train(inputs,outputs)

# t,s = load_tweets_and_summaries()
# inputs,outputs = vectorize_texts_and_summaries(t, s)
# train(inputs,outputs)

#param_search(learning_rates=[0.1,0.01,0.001,0.0001], hidden_layer_sizes=[500,1000,1500,2000,3000])


#Test on one sentence
text = "Getting sent home cuz i had an asthma attack wasnt my plan today, been so bad recently"
unit_batch_x = np.zeros((1, MAX_TWEET_LEN, EMBEDDING_SIZE))
unit_batch_x[0] = vectorize_text(text)
results = test(unit_batch_x)
for res in results:
    print( " ".join([word for idx,word in enumerate(text.split(" ")[:MAX_TWEET_LEN]) if(res[idx] == 1)]) )
