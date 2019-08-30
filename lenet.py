import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras
import keras.layers as layers
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from batch_normalization import CustomizedBatchNorm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def create_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # The shape of X_train is (60000, 28, 28). Each image has 28 x 28 resolution. The shape of X_test is (10000, 28, 28)

    # tensorflow input format: (batch, height, width, channels)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # one-hot coding
    number_of_classes = 10

    Y_train = np_utils.to_categorical(y_train, number_of_classes)
    Y_test = np_utils.to_categorical(y_test, number_of_classes)

    print('# of training images:', X_train.shape[0])
    print('# of test images:', X_test.shape[0])
    return X_train, X_test, Y_train, Y_test


def evaluate_model(X_train, X_test, Y_train, Y_test, BN = False):
    model = keras.Sequential()
    if BN is False:
        model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
        model.add(layers.AveragePooling2D())

        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())

        model.add(layers.Flatten())

        model.add(layers.Dense(units=120, activation='relu'))

        model.add(layers.Dense(units=84, activation='relu'))

        model.add(layers.Dense(units=10, activation = 'softmax'))

    else:
        model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), input_shape=(28,28,1)))
        model.add(CustomizedBatchNorm())
        model.add(layers.Activation("relu"))
        model.add(layers.AveragePooling2D())

        model.add(layers.Conv2D(filters=16, kernel_size=(3, 3)))
        model.add(CustomizedBatchNorm())
        model.add(layers.Activation("relu"))
        model.add(layers.AveragePooling2D())

        model.add(layers.Flatten())

        model.add(layers.Dense(units=120))
        model.add(CustomizedBatchNorm())
        model.add(layers.Activation("relu"))

        model.add(layers.Dense(units=84))
        model.add(CustomizedBatchNorm())
        model.add(layers.Activation("relu"))

        model.add(layers.Dense(units=10, activation='softmax'))

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    EPOCHS = 20
    BATCH_SIZE = 128

    steps_per_epoch = X_train.shape[0]//BATCH_SIZE
    test_steps = X_test.shape[0]//BATCH_SIZE

    # use data augmentation to reduce over-fitting
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                             height_shift_range=0.08, zoom_range=0.08)

    test_gen = ImageDataGenerator()

    train_generator = gen.flow(X_train, Y_train, batch_size=BATCH_SIZE)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=BATCH_SIZE)

    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                        validation_data=test_generator, validation_steps=test_steps, verbose=0)

    score = model.evaluate(X_test, Y_test)
    return history, score


X_train, X_test, Y_train, Y_test = create_dataset()

start = datetime.now()
history, score = evaluate_model(X_train, X_test, Y_train, Y_test)
time_delta = datetime.now() - start
start = datetime.now()
history_nb, score_nb = evaluate_model(X_train, X_test, Y_train, Y_test, BN=True)
time_delta_nb = datetime.now() - start

print('training without Batch Normalization took {}s.'.format(time_delta))
print('Test loss without Batch Normalization:', score[0])
print('Test accuracy without Batch Normalization: ', score[1])

print('training with Batch Normalization took {}s.'.format(time_delta_nb))
print('Test loss with Batch Normalization:', score_nb[0])
print('Test accuracy with Batch Normalization: ', score_nb[1])

train_acc = history.history['acc']
test_acc = history.history['val_acc']
train_acc_nb = history_nb.history['acc']
test_acc_nb = history_nb.history['val_acc']

plt.plot(train_acc, label = 'No Batch Normalization')
plt.plot(train_acc_nb, label = 'With Batch Normalization')
plt.title('Evaluation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.savefig('training accuracy result.png')

plt.clf()
plt.plot(test_acc, label = 'No Batch Normalization')
plt.plot(test_acc_nb, label = 'With Batch Normalization')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.legend()
plt.savefig('accuracy result.png')
plt.show()

