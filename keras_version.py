import os
import glob
import h5py
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

DATA_DIR = "ModelNet10"
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!R]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 8
folders = glob.glob(os.path.join(DATA_DIR, "[!R]*"))
class_map = {}
for i,folder in enumerate(folders):
    class_map[i] = folder.split("\\")[-1]
# train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
#     NUM_POINTS
# )

# with h5py.File("ModelNet10.h5", "w") as f:
#     f.create_dataset("train_x", data=train_points)
#     f.create_dataset("test_x", data=test_points)
#     f.create_dataset("train_y", data=train_labels)
#     f.create_dataset("test_y", data=test_labels)

with h5py.File("ModelNet10.h5", "r") as f:
    print(f.keys())
    train_points = f["train_x"][:]
    train_labels = f["train_y"][:]
    test_points = f["test_x"][:]
    test_labels = f["test_y"][:]


def augment(points, label):
    '''points:(N,3),label:int'''
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
train_dataset = train_dataset.shuffle(
    len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)


# Each convolution and fully-connected layer (with exception for end layers)
# consits of Convolution / Dense -> Batch Normalization -> ReLU Activation.
def conv_bn(x, filters):
    x = layers.Conv1D(filters, 1)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


# PointNet consists of two core components. 
# The primary MLP network, and the transformer net (T-net). 
# The T-net aims to learn an affine transformation matrix by its own mini network. 
# The T-net is used twice. The first time to transform the input features (n, 3) 
# into a canonical(规范的) representation. 
# The second is an affine transformation for alignment(对准) in feature space (n, 3). 
# As per the original paper we constrain the transformation to be close to 
# an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self,num_features,l2reg=0.001) -> None:
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x,(-1,self.num_features,self.num_features))
        xxt = tf.tensordot(x,x,axes=(2,2))
        xxt = tf.reshape(xxt,(-1,self.num_features,self.num_features))
        return tf.reduce_sum(self.l2reg*tf.square(xxt-self.eye))


# T-net
def TNET(inputs,num_features):
    # 初始化bias为单位矩阵
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs,32)
    x = conv_bn(x,64)
    x = conv_bn(x,512)
    x = layers.GlobalMaxPool1D()(x)
    x = dense_bn(x,256)
    x = dense_bn(x,128)
    x = layers.Dense(
        num_features*num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features,num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2,1))([inputs,feat_T])

inputs = keras.Input(shape=(NUM_POINTS,3))

x = TNET(inputs,3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = TNET(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs,outputs=outputs,name="PointNet")
model.summary()


model.compile(
    loss ="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)
model.fit(train_dataset,epochs=20,validation_data=test_dataset)
