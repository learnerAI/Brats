
import os
import cv2
import numpy as np
import glob
# neural imaging
import nibabel as nib
import tarfile
import utils
import unetmodel
import tensorflow as tf
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.tensorflow
from mlflow import log_metric, log_param, log_artifact
import matplotlib.pyplot as plt

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

VOLUME_SLICES = 100
VOLUME_START_AT = 22  # first slice of volume that we will include


INPUT_SHAPE = 64
IMG_SIZE = 64


# DEFINE seg-areas
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, filepaths, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.filepaths = filepaths
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        Batch_ids = [self.filepaths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim))
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        # Generate data
        for c, case_path in enumerate(Batch_ids):
            base = os.path.basename(case_path)
            data_path = os.path.join(case_path, base + '_flair.nii.gz')
            flair = nib.load(data_path).get_fdata()

            # data_path = os.path.join(case_path, base + '_t1ce.nii.gz');
            # ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, base + '_seg.nii.gz')
            seg = nib.load(data_path).get_fdata()

            for j in range(VOLUME_SLICES):
                X[j + VOLUME_SLICES * c, :, :] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), 0)
                #  X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

                y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START_AT]

        # Generate masks
        y[y == 4] = 3
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
        return X / np.max(X), Y

<<<<<<< HEAD
def run(LEARNING_RATE=0.001, DROPOUT=0.0, LOSS="categorical_crossentropy", EPOCHS=50, DEVICE='GPU'):
=======
def run(LEARNING_RATE=0.001, DROPOUT=0.0, LOSS="categorical_crossentropy", EPOCHS=1, DEVICE='GPU'):
>>>>>>> 985812f1580fd4a838a12c4e77b7af6c62b6248d
    mlflow.set_experiment("unet")
    with mlflow.start_run():
        mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
        mlflow.log_param("DROPOUT", DROPOUT)
        mlflow.log_param("LOSS", LOSS)
        mlflow.log_param("EPOCHS", EPOCHS)
        mlflow.log_param("DEVICE", DEVICE)

        model = unetmodel.get_model(INPUT_SHAPE, DEVICE, LEARNING_RATE, LOSS, DROPOUT)
        files = glob.glob(os.path.join("data", '*'))

        files = [file for file in files if '.gz' not in file]

        train_ids, val_ids = train_test_split(files, test_size=0.2)

        training_generator = DataGenerator(train_ids)
        valid_generator = DataGenerator(val_ids)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                          patience=3, verbose=1, mode='auto'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                              patience=5, min_lr=0.00001, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                            verbose=1, save_best_only=True, save_weights_only=True)
        ]

        tf.keras.backend.clear_session()

        history = model.fit(training_generator,
                            epochs=EPOCHS,
                            steps_per_epoch=len(train_ids),
                            callbacks=callbacks,
                            validation_data=valid_generator
                            )

        hist = history.history

        acc = hist['accuracy']
        val_acc = hist['val_accuracy']

        epoch = range(len(acc))

        loss = hist['loss']
        val_loss = hist['val_loss']

        train_dice = hist['dice_coef']
        val_dice = hist['val_dice_coef']

        f, ax = plt.subplots(1, 3, figsize=(16, 8))

        ax[0].plot(epoch, acc, 'b', label='Training Accuracy')
        ax[0].plot(epoch, val_acc, 'r', label='Validation Accuracy')
        ax[0].legend()

        ax[1].plot(epoch, loss, 'b', label='Training Loss')
        ax[1].plot(epoch, val_loss, 'r', label='Validation Loss')
        ax[1].legend()

        ax[2].plot(epoch, train_dice, 'b', label='Training dice coef')
        ax[2].plot(epoch, val_dice, 'r', label='Validation dice coef')
        ax[2].legend()

        plt.savefig("acc_loss_plot.png", bbox_inches='tight', pad_inches=0)

        # Log metrics
        log_metric("accuracy", acc[-1])
        log_metric("val_accuracy", val_acc[-1])
        log_metric("loss", loss[-1])
        log_metric("val_loss", val_loss[-1])
        log_metric("train_dice_coef", train_dice[-1])
        log_metric("val_dice_coef", val_dice[-1])

        # Log artifact
        log_artifact("acc_loss_plot.png")

        model.load_weights("best_model.h5")
        # Log the model
        mlflow.tensorflow.log_model(model, "model", custom_objects={'dice_coef': unetmodel.dice_coef, 'precision': unetmodel.precision,"sensitivity": unetmodel.sensitivity, "specificity":unetmodel.specificity} )
    return

if __name__ == '__main__':
    if not os.path.exists("data"):
        utils.extract_data_to_folder("./ProjectBrain")

    run()







