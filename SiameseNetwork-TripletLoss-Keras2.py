from keras.models import load_model
import keras
from tensorflow.keras import layers
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.keras import metrics

from datetime import datetime
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
shutil.rmtree('logs/', ignore_errors=True)
# K = tf.keras.backend
from keras import backend as B

epochs = 50
learning_rate = 0.00006
batch_size = 8
target_shape = (160, 160)
base_dir = "./logs"

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=32, shuffle=True):
        self.dataset = self.create_dataset(dataset_path)
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.no_of_people = len(list(self.dataset.keys()))
        self.min_length = min([len(self.dataset[i]) for i in self.dataset.keys()])

    def __getitem__(self, index):
        people = list(self.dataset.keys())
        P = []
        A = []
        N = []
        for i in range(self.batch_size*index):
            person = people[random.randrange(self.no_of_people)]

            anchor_index = random.randint(0, len(self.dataset[person])-1)
            a = self.get_image(person, anchor_index)

            positive_index = random.randint(0, len(self.dataset[person])-1)
            while positive_index == anchor_index:
                positive_index = random.randint(0, len(self.dataset[person])-1)
            p = self.get_image(person, positive_index)

            people.remove(person)
            negative_person = random.choice(people)
            # print(person,negative_person)
            negative_index = random.randint(
                0, len(self.dataset[negative_person])-1)
            n = self.get_image(negative_person, negative_index)
            P.append(p)
            A.append(a)
            N.append(n)
            people = list(self.dataset.keys())        
        return [A, P, N]   

    def __len__(self):
        return self.min_length // self.batch_size

    def create_dataset(self, dataset_path):
        dataset = {}
        for root, directories, files in os.walk(dataset_path):
            for directory in directories:
                for _, __, file in os.walk(os.path.join(dataset_path, directory)):
                    dataset[directory] = file
        return dataset
    
    def get_image(self, person, index):
        # print(os.path.join(self.dataset_path, os.path.join( person, self.dataset[person][index])))
        img = os.path.join(self.dataset_path, os.path.join(
            person, self.dataset[person][index]))
        return img

data_gen = DataGenerator(dataset_path='data', batch_size=batch_size)

def gen_data():
    data = data_gen.__getitem__(data_gen.__len__())
    anchor_images, positive_images, negative_images = data
    # print(len(anchor_images))
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)

    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(len(anchor_images) * 0.8))
    val_dataset = dataset.skip(round(len(anchor_images) * 0.8))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset

train_dataset, val_dataset = gen_data()
# print(train_dataset)

base_cnn = load_model('model/facenet_keras.h5')
base_cnn.load_weights('./model/facenet_keras_weights.h5')

embedding = keras.Model(base_cnn.input, base_cnn.output, name="Embedding")
for layer in embedding.layers[:-2]:
    layer.trainable = False

# print(layers.Layer)
class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # super(DistanceLayer, self).__init__()
        pass

    def __call__(self, anchor, positive, negative):

        i = B.placeholder(shape=(2,), name='input')
        ap_distance = B.sum(B.square(anchor - positive), -1)
        an_distance = B.sum(B.square(anchor - negative), -1)
        # f = B.function([i], [i])
        # print(type(((ap_distance, an_distance))),type(ap_distance))
        # out = np.array([ap_distance,ap_distance])
        # print(type(out))
        # print(f([out]))
        # return (ap_distance, an_distance)
        return [anchor, positive, negative]


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))
# print(embedding(negative_input))

distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)

siamese_network = tf.keras.Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

class SiameseModel(tf.keras.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5,**kwargs):
        super(SiameseModel, self).__init__(**kwargs)
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        aanchor, positive, negative = self.siamese_network(data)
        ap_distance = B.sum(B.square(anchor - positive), -1)
        an_distance = B.sum(B.square(anchor - negative), -1)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


siamese_model = SiameseModel(siamese_network, inputs=[
                             anchor_input, positive_input, negative_input], outputs=distances)
siamese_model.compile(optimizer=keras.optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

