import utils
import unetmodel
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate
import shutil
import glob
import os
import gc
from tqdm import tqdm

def launch_model(Input_Data, code, model):
    InData = utils.Data_Concatenate(Input_Data)
    AIO = concatenate(InData, axis=3)
    AIO = np.array(AIO, dtype='float32')
    TR = np.array(AIO[:, :, :, 1], dtype='float32')
    TRL = np.array(AIO[:, :, :, 4], dtype='float32')
    X_train, X_test, Y_train, Y_test = train_test_split(TR, TRL, test_size=0.15, random_state=32)
    AIO = TRL = 0

    # Load the weights from pretrained_ckpt into model.
    model.load_weights("./ProjectBrain/BraTs2020_2.h5")

    # Fitting the model over the data
    print("-- Fitting the model over the data --")
    history = model.fit(X_train, Y_train, batch_size=16, epochs=20, validation_split=0.20, verbose=1, initial_epoch=0)

    # Evaluating the model on the training and testing data
    print("-- Evaluating the model on the training and testing data --")
    model.evaluate(x=X_train, y=Y_train, batch_size=16, verbose=1, sample_weight=None, steps=None)
    model.evaluate(x=X_test, y=Y_test, batch_size=16, verbose=1, sample_weight=None, steps=None)

    # Plotting the Graphs of Accuracy, Dice_coefficient, Loss at each epoch on Training and Testing data
    print("-- Plotting the Graphs of Accuracy, Dice_coefficient, Loss at each epoch on Training and Testing data --")
    utils.Accuracy_Graph(history)
    utils.Dice_coefficient_Graph(history)
    utils.Loss_Graph(history)

    model.save('./final_models/BraTs2021_' + code + '.h5')
    return


def train_model(model, Path, top_limit_number, split_number):
    p = os.listdir(Path)
    Input_Data = []
    init_counter = 0
    inside_split_countert = 1
    total_count_img, partial_count_img = len(p), int(len(p) / split_number)

    for i in tqdm(p):
        if (int(init_counter * inside_split_countert) == total_count_img) or (
                int(top_limit_number * split_number) == int(init_counter * inside_split_countert)):
            print("Launch Final model.")
            launch_model(Input_Data, str(init_counter), model)
            del (Input_Data)
            gc.collect()
            break
        if (init_counter == split_number):
            print("Launch model :" + str(init_counter * inside_split_countert))
            launch_model(Input_Data, str(init_counter * inside_split_countert), model)
            del (Input_Data)
            gc.collect()
            Input_Data = []
            inside_split_countert = inside_split_countert + 1
            init_counter = 0
        create_path_post = Path + '/' + i + '/*'

        for name in sorted(glob.glob(create_path_post.replace('\n', ''))):
            os.system('gunzip ' + name)
        brain_dir = os.path.normpath(Path + '/' + i + '/')
        flair = glob.glob(os.path.join(brain_dir, '*_flair*.nii'))
        t1 = glob.glob(os.path.join(brain_dir, '*_t1*.nii'))
        t1ce = glob.glob(os.path.join(brain_dir, '*_t1ce*.nii'))
        t2 = glob.glob(os.path.join(brain_dir, '*_t2*.nii'))
        gt = glob.glob(os.path.join(brain_dir, '*_seg*.nii'))
        modalities_dir = [flair[0], t1[0], t1ce[0], t2[0], gt[0]]
        P_Data = utils.Data_Preprocessing(modalities_dir)
        Input_Data.append(P_Data)
        shutil.rmtree(Path + '/' + i)
        init_counter = init_counter + 1
    return

if __name__ == '__main__':
    utils.extract_data_to_folder("./ProjectBrain")
    utils.data_to_folders("./data", './post_process_data/', './final_models')
    path = './post_process_data'

    top_limit_number = 1
    split_number = 20
    model = unetmodel.get_model(240, 'gpu', 0.001, 'binary_crossentropy')
    train_model(model, path, top_limit_number, split_number)
