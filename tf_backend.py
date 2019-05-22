from __future__ import absolute_import, division, print_function, unicode_literals

#!pip install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

do_not_train_words = 1

if do_not_train_words:
	PATH = os.path.abspath('.')+'/train2014/'

	annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'


	# read the json file
	with open(annotation_file, 'r') as f:
	    annotations = json.load(f)

	# storing the captions and the image name in vectors
	all_captions = []
	all_img_name_vector = []

	for annot in annotations['annotations']:
	    caption = '<start> ' + annot['caption'] + ' <end>'
	    image_id = annot['image_id']
	    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

	    all_img_name_vector.append(full_coco_image_path)
	    all_captions.append(caption)

	# shuffling the captions and image_names together
	# setting a random state
	train_captions, img_name_vector = shuffle(all_captions,
	                                          all_img_name_vector,
	                                          random_state=1)

	# selecting the first 30000 captions from the shuffled set
	num_examples = 30000
	train_captions = train_captions[:num_examples]
	img_name_vector = img_name_vector[:num_examples]

	# This will find the maximum length of any caption in our dataset
	def calc_max_length(tensor):
	    return max(len(t) for t in tensor)

	# The steps above is a general process of dealing with text processing

	# choosing the top 5000 words from the vocabulary
	top_k = 5000
	tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
	                                                  oov_token="<unk>",
	                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
	tokenizer.fit_on_texts(train_captions)
	train_seqs = tokenizer.texts_to_sequences(train_captions)

	tokenizer.word_index['<pad>'] = 0
	tokenizer.index_word[0] = '<pad>'

	# creating the tokenized vectors
	train_seqs = tokenizer.texts_to_sequences(train_captions)

	# padding each vector to the max_length of the captions
	# if the max_length parameter is not provided, pad_sequences calculates that automatically
	cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

	# calculating the max_length
	# used to store the attention weights
	max_length = calc_max_length(train_seqs)

	# Create training and validation sets using 80-20 split
	img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
	                                                                    cap_vector,
	                                                                    test_size=0.2,
	                                                                    random_state=0)


BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
if do_not_train_words:
	vocab_size = 8236
else:
	vocab_size =  len(tokenizer.word_index) + 1
	num_steps = len(img_name_train) // BATCH_SIZE

# shape of the vector extracted from InceptionV3 is (64, 2048)
# these two variables represent that
features_shape = 2048
attention_features_shape = 64

if do_not_train_words:
	# loading the numpy files
	def map_func(img_name, cap):
	  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
	  return img_tensor, cap

	dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

	# using map to load the numpy files in parallel
	dataset = dataset.map(lambda item1, item2: tf.numpy_function(
	          map_func, [item1, item2], [tf.float32, tf.int32]),
	          num_parallel_calls=tf.data.experimental.AUTOTUNE)

	# shuffling and batching
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # we get 1 at the last axis because we are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

"""## Checkpoint"""

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder = encoder,
                           decoder = decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0

skip_training = False
ckpt.restore(ckpt_manager.latest_checkpoint)
if ckpt_manager.latest_checkpoint:
	skip_training = True
else:
	print("ERROR: You've not copied checkpoints from Google drive. Please refer to README for future details")
	exit(1)

"""## Training

* We extract the features stored in the respective `.npy` files and then pass those features through the encoder.
* The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
* The decoder returns the predictions and the decoder hidden state.
* The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
* Use teacher forcing to decide the next input to the decoder.
* Teacher forcing is the technique where the target word is passed as the next input to the decoder.
* The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
"""


# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []

@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss

EPOCHS = 20

if not skip_training:
	for epoch in range(start_epoch, EPOCHS):
	    start = time.time()
	    total_loss = 0

	    for (batch, (img_tensor, target)) in enumerate(dataset):
	        batch_loss, t_loss = train_step(img_tensor, target)
	        total_loss += t_loss

	        if batch % 100 == 0:
	            print ('Epoch {} Batch {} Loss {:.4f}'.format(
	              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
	    # storing the epoch end loss value to plot later
	    loss_plot.append(total_loss / num_steps)

	    if epoch % 5 == 0:
	    	ckpt_manager.save()

	    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
	                                         total_loss/num_steps))
	    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

	plt.plot(loss_plot)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss Plot')
	plt.show()

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def process_image(image_path):
	result, attention_plot = evaluate(image_path)
	print ('Prediction Caption inside function:', ' '.join(result))
	plot_attention(image_path, result, attention_plot)
	# opening the image

	return result


def image_url_to_path(url):
	return tf.keras.utils.get_file(url.rsplit('/', 1)[-1],
                                     origin=url)


def get_caption_from_image(image_url, show_image = False):
	path_to_image = image_url_to_path(image_url)
	result = process_image(path_to_image)

	if show_image:
		Image.open(path_to_image)


	return result


# examples of usage 

image_giraffe = 'https://i.dailymail.co.uk/i/pix/2017/11/22/22/469A374A00000578-5109115-A_pair_of_lions_risked_their_lives_in_an_attempt_to_take_down_a_-a-95_1511388548555.jpg'

res = get_caption_from_image(image_giraffe)

print("Result is: " + str(res))


image_seagull = 'https://www.wonderplugin.com/videos/demo-image0.jpg'

res = get_caption_from_image(image_seagull, True)

print("Result is: " + str(res))


