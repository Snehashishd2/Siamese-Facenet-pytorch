from keras import layers
from keras.models import load_model
import keras 
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
tf.executing_eagerly()
from datetime import datetime
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
shutil.rmtree('logs/', ignore_errors=True)
from keras import backend as K
tf.executing_eagerly() 
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

epochs = 50
learning_rate = 0.00006
batch_size = 8
target_shape = (160, 160)
base_dir = "./logs"
# print(tf.summary.create_file_writer())

class DataGenerator():
    def __init__(self, dataset_path, batch_size=32, shuffle=True):
        self.dataset = self.create_dataset(dataset_path)
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.no_of_people = len(list(self.dataset.keys()))
        self.min_length = min([len(self.dataset[i]) for i in self.dataset.keys()])
        self.on_epoch_end()

    def __getitem__(self, index):
        people = list(self.dataset.keys())
        P = []
        A = []
        N = []
        for i in range(self.batch_size):
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
        A = np.asarray(A)
        N = np.asarray(N)
        P = np.asarray(P)
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
    
    def on_epoch_end(self):
        if self.shuffle:
            keys = list(self.dataset.keys())
            random.shuffle(keys)
            dataset_ = {}
            for key in keys:
                dataset_[key] = self.dataset[key]
            self.dataset = dataset_
    
    def get_image(self, person, index):
        # print(os.path.join(self.dataset_path, os.path.join( person, self.dataset[person][index])))
        img = cv2.imread(os.path.join(self.dataset_path, os.path.join(
            person, self.dataset[person][index])))        
        img = cv2.resize(img, (160, 160))
        img = np.asarray(img, dtype=np.float64)
        # print(img.shape,np.mean(img, axis=1, keepdims=True).shape)
        mean, std = img.mean(),img.std()
        img = (img - mean)/std 
        # img = preprocess_input(img)
        return img


data = DataGenerator(dataset_path='data', batch_size=8)
# print(data[0], data.dataset.keys(), np.array(data.__getitem__(1)).shape )

embedding = load_model('model/facenet_keras.h5')
embedding.load_weights('./model/facenet_keras_weights.h5')
for layer in embedding.layers:
    layer.trainable = True
for layer in embedding.layers[:-10]:
    layer.trainable = False



class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, anchor, positive, negative):
        return [anchor, positive, negative]
anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))
distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)


class SiameseNetwork(keras.Model):
    def __init__(self, embedding):
        super(SiameseNetwork, self).__init__()
        self.embedding = embedding
        
    # @tf.function
    def __call__(self, inputs):
        image_1, image_2, image_3 =  inputs
        with tf.name_scope("Anchor") as scope:
            feature_1 = self.embedding.predict_on_batch(image_1)
            feature_1 = tf.math.l2_normalize(feature_1, axis=-1)
        with tf.name_scope("Positive") as scope:
            feature_2 = self.embedding.predict_on_batch(image_2)
            feature_2 = tf.math.l2_normalize(feature_2, axis=-1)
        with tf.name_scope("Negative") as scope:
            feature_3 = self.embedding.predict_on_batch(image_3)
            feature_3 = tf.math.l2_normalize(feature_3, axis=-1)
        return [feature_1, feature_2, feature_3]
    
    # @tf.function
    def get_features(self, inputs):
        return tf.math.l2_normalize(self.embedding.predict_on_batch(inputs), axis=-1)


model = SiameseNetwork(embedding)
# model.summary()
print(model.trainable_weights)

def loss_function(x, alpha = 0.2):
    # Triplet Loss function.
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.mean(K.maximum(basic_loss,0.0))
    return loss

optimizer = tf.keras.optimizers.Adam(lr=0.0003)
def train(X):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = loss_function(y_pred)
    # print(model.optimizer.get_gradients(loss,model.trainable_weights))
    grad = tape.gradient(loss, model.trainable_weights)
    print(grad)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    return loss

# checkpoint = tf.train.Checkpoint(model=model)
for i in range(data.__len__()):
    d = data[i]
    print(model(d))

losses = []
accuracy = []

no_of_batches = data.__len__()
for i in range(1, epochs+1, 1):
    loss = 0
    with tqdm(total=no_of_batches) as pbar:
        
        description = "Epoch " + str(i) + "/" + str(epochs)
        pbar.set_description_str(description)
        
        for j in range(no_of_batches):
            d = data[j]
            temp = train(d)
            loss += temp
            
            pbar.update()
            print_statement = "Loss :" + str(temp.numpy())
            pbar.set_postfix_str(print_statement)
        
        loss /= no_of_batches
        losses.append(loss.numpy())
        with file_writer.as_default():
            tf.summary.scalar('Loss', data=loss.numpy(), step=i)
            
        print_statement = "Loss :" + str(loss.numpy())
        
        pbar.set_postfix_str(print_statement)
