import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NP_DIR = os.path.join("dataset", "np_data")
MODEL_PATH = "siamese_model.keras"
EMB_MODEL_PATH = "embedding_model.keras"
IMG_SIZE = 160

def load_data():
    x = np.load(os.path.join(NP_DIR, "img_train.npy"))
    y = np.load(os.path.join(NP_DIR, "label_train.npy")).reshape(-1)
    x = x.astype("float32") / 255.0

    return x, y

def generate_pairs(x, y):
    pairs = []
    labels = []

    num_classes = len(np.unique(y))

    class_idx = {cls: np.where(y == cls)[0] for cls in range(num_classes)}

    for cls in range(num_classes):
        idxs = class_idx[cls]

        for i in range(len(idxs) - 1):
            pairs.append([x[idxs[i]], x[idxs[i + 1]]])
            labels.append(1)

        other_classes = [c for c in range(num_classes) if c != cls]
        for i in range(len(idxs)):
            neg_cls = np.random.choice(other_classes)
            neg_idx = np.random.choice(class_idx[neg_cls])
            pairs.append([x[idxs[i]], x[neg_idx]])
            labels.append(0)

    pairs = np.array(pairs)
    labels = np.array(labels).astype("float32")

    return pairs[:, 0], pairs[:, 1], labels

def build_embedding_network():
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="linear")(x)

    model = keras.Model(inp, x, name="embedding_model")
    return model

def build_siamese_network(embedding_model):
    input_a = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    input_b = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    emb_a = embedding_model(input_a)
    emb_b = embedding_model(input_b)

    distance = layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=1, keepdims=True))([emb_a, emb_b])

    siamese = keras.Model([input_a, input_b], distance, name="siamese_network")
    return siamese

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return tf.reduce_mean(
        y_true * tf.square(y_pred) +
        (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    )

def train():
    x, y = load_data()
    xa, xb, labels = generate_pairs(x, y)

    embedding = build_embedding_network()
    siamese = build_siamese_network(embedding)

    siamese.compile(optimizer="adam", loss=contrastive_loss)

    siamese.fit(
        [xa, xb], labels,
        epochs=30,
        batch_size=50,
        validation_split=0.1,
    )

    siamese.save(MODEL_PATH)
    embedding.save(EMB_MODEL_PATH)

if __name__ == "__main__":
    train()