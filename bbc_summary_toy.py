from __future__ import division
import tensorflow as tf
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

spacy_nlp = spacy.load('en_core_web_lg')
allWords = []

MAX_SENTENCE_LEN = 25
MAX_SUMMARY_LEN = 10
EMBEDDING_SIZE = 300


# p is the index into our training data for where we are now.
p = 0
accuracy_print_interval = 5

# number of neurons in each layer
input_num_units = MAX_SENTENCE_LEN*EMBEDDING_SIZE
hidden_num_units = 9000
output_num_units = MAX_SENTENCE_LEN

epochs = 60
batch_size = 20
learning_rate = 0.01

def load_texts_and_summaries(path, TextVecorizationLevel="sentence"):
    texts = []
    summaries = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(path+"/"+filename, 'r') as f:
                x = f.readlines()
                summaries.append(x[0].strip())
                if(TextVecorizationLevel == 'sentence'):
                    paragraph = " ".join(x[1:]).strip()
                    texts.append(extract_sentences_from_paragraph(paragraph))
                elif(TextVecorizationLevel == 'paragraph'):
                    paragraph = " ".join(x[1:]).strip()
                    texts.append(paragraph)
                else:
                    paragraph = " ".join(x[1:]).strip()
                    texts.append(paragraph)
        #break
    return (texts, summaries)

def extract_sentences_from_paragraph(paragraph):
    sentences = paragraph.split(".")
    return [s for s in sentences if len(s.strip()) != 0]

def vectorize_texts_and_summaries(texts, summaries):
    global spacy_nlp
    t_vectors = []
    source_text_vectors = []
    s_vectors = []
    target_summary_vectors = []

    if(type(texts[0]) == list):
        for i in xrange(len(texts)):
            summary = summaries[i]
            sentences = texts[i]
            sentences_container = []

            # Get text vector
            for s in sentences:
                sentence_vector = np.array([])
                target_vector = np.zeros((2))
                for w in s:
                    if(len(sentence_vector) < MAX_SENTENCE_LEN*EMBEDDING_SIZE):
                        sentence_vector = np.append(sentence_vector, spacy_nlp(unicode(w,errors='replace')).vector)
                    else:
                        break
                while(len(sentence_vector) < MAX_SENTENCE_LEN*EMBEDDING_SIZE):
                    sentence_vector = np.append(sentence_vector,np.zeros(EMBEDDING_SIZE))
                source_text_vectors.append(sentence_vector)
                if(len(s) == 20):
                    target_vector[1] = 1
                else:
                    target_vector[0] = 1
                target_summary_vectors.append(target_vector)
    elif(type(texts[0]) == str):
        for i in xrange(len(texts)):
            text = texts[i]
            t_vectors.append(spacy_nlp(unicode(text,errors='replace')).vector)
            summary = summaries[i]
            s_vectors.append(spacy_nlp(unicode(summary,errors='replace')).vector)
    else:
        raise BadValueError("Texts should either be a list of strings or a list of string lists")

    return (source_text_vectors, target_summary_vectors)

def create_inputs_and_outputs(texts, summaries, textVecorizationLevel="sentence"):
    '''
    Creates inputs and outputs such that input is a list of sentence vectors where
    a sentence vector is a concatenation of its constituent word vectors. The output
    is of the same length as the input but has a 1 when that constituent word should
    appear in the summary and 0 when not.
    '''
    inputs = []
    outputs = []
    if(textVecorizationLevel == 'sentence'):
        for i in xrange(len(summaries)):
            sentences = texts[i]
            for s in sentences:
                inputs.append(s)
                outputs.append(summaries[i])
    else:
        for i in xrange(len(summaries)):
            inputs.append(text[i])
            outputs.append(summaries[i])
    return (inputs, outputs)


def next_batch(all_inputs, all_outputs):
    global p

    batch_x = np.zeros((batch_size, input_num_units))
    batch_y = np.zeros((batch_size, 2))

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


def vec2word(vec):
    '''
    Fetches the word for a given vector by checking for the most similar vector
    in the vocabulary.

    @param vec a spacy vector (300-long np array)
    @return String the word for that vector
    '''
    global allWords
    if(len(allWords) == 0):
        allWords = list({w for w in spacy_nlp.vocab if w.has_vector and w.orth_.islower()})

    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    allWords.sort(key=lambda w: cosine(w.vector, vec))
    allWords.reverse()
    return allWords[:0].orth_


def train(all_inputs, all_outputs):
    global learning_rate
    global batch_size
    global epochs
    global input_num_units
    global hidden_num_units
    global output_num_units
    global accuracy_print_interval



    # tf Graph input. None means that the first dimension can be of any size so it represents the batch size
    x = tf.placeholder(tf.float32, [None, input_num_units])
    y = tf.placeholder(tf.float32, [None, 2])

    # define weights and biases of the neural network
    weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units])),
    'output': tf.Variable(tf.random_normal([hidden_num_units, 2]))
    }

    biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units])),
    'output': tf.Variable(tf.random_normal([2]))
    }

    # Now create our neural networks computational graph
    # Wx.Wh + Bh
    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    #activation function to hidden layer calculation
    hidden_layer = tf.nn.relu(hidden_layer)
    # Output calc
    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(output_layer,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            total_cost = 0
            num_batches = int(len(all_outputs)/batch_size)
            for i in range(num_batches):
                batch_x, batch_y = next_batch(all_inputs, all_outputs)
                c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                total_cost += c
            avg_cost = total_cost/num_batches
            print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
            # show training accuracy after every 2nd epoch
            if(epoch % accuracy_print_interval == 0):
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                print "Training Accuracy: ", "{:.5f}".format(acc)
        print "\nTraining complete!"





t,s = load_texts_and_summaries("bbc/sport")
inputs,outputs = vectorize_texts_and_summaries(t, s)
# print(len(inputs[0]))
#print(len(outputs))
train(inputs,outputs)

# doc1 = spacy_nlp(u"European leaders say Asian states must let their currencies rise against the US dollar to ease pressure on the euro.")
# doc2 = spacy_nlp(u"The European single currency has shot up to successive all-time highs against the dollar over the past few months. Tacit approval from the White House for the weaker greenback, which could help counteract huge deficits, has helped trigger the move. But now Europe says the euro has had enough, and Asia must now share some of the burden.")
# similarity = doc1.similarity(doc2)
# print(doc1.text, doc2.text, similarity)
