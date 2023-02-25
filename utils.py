import os
import tarfile
import glob
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def extract_tar_data(path):
  if not os.path.exists("./data"):
    os.mkdir("./data")
  tar = tarfile.open(path)
  tar.extractall("./data")
  tar.close()
  return

def extract_data_to_folder(path_to_folder):
  files = glob.glob(os.path.join(path_to_folder, "*.tar"))
  for file in files:
    extract_tar_data(file)


def data_to_folders(path_to_main_folder, path_to_post_process, path_to_final_models):
    # Declaring paths
    Path_brats = path_to_main_folder
    post_data = path_to_post_process
    final_models = path_to_final_models

    # Creating directories
    if not os.path.exists(post_data):
        os.mkdir(post_data)
    if not os.path.exists(final_models):
        os.mkdir(final_models)

    # Creating a text file which will contain list of all data set numbers in sorted order like 0001 etc.
    # This will be used in ordering data
    files = glob.glob(os.path.join(Path_brats, "*"))

    filenames = [os.path.basename(x).split("_")[1] for x in files]

    for linea in filenames:
        path_dest = post_data + linea
        path_dest = path_dest.replace('\n', '')
        if os.path.isdir(path_dest) != True:
            os.mkdir(path_dest)
        create_path_brats = Path_brats + "/BraTS2021_" + linea + '/*'
        for name in sorted(glob.glob(create_path_brats.replace('\n', ''))):
            shutil.move(name, path_dest)
    return

def Data_Concatenate(Input_Data):
    counter=0
    Output= []
    for i in range(5):
        print('$')
        c=0
        counter=0
        for ii in range(len(Input_Data)):
            if (counter != len(Input_Data)):
                a= Input_Data[counter][:,:,:,i]
                #print('a={}'.format(a.shape))
                b= Input_Data[counter+1][:,:,:,i]
                #print('b={}'.format(b.shape))
                if(counter==0):
                    c= np.concatenate((a, b), axis=0)
                    print('c1={}'.format(c.shape))
                    counter= counter+2
                else:
                    c1= np.concatenate((a, b), axis=0)
                    c= np.concatenate((c, c1), axis=0)
                    print('c2={}'.format(c.shape))
                    counter= counter+2
        c= c[:,:,:,np.newaxis]
        Output.append(c)
    return Output

def Data_Preprocessing(modalities_dir):
    all_modalities = []
    for modality in modalities_dir:
        nifti_file   = nib.load(modality)
        brain_numpy  = np.asarray(nifti_file.dataobj)
        all_modalities.append(brain_numpy)
    # brain_affine   = nifti_file.affine
    all_modalities = np.array(all_modalities)
    all_modalities = np.rint(all_modalities).astype(np.int16)
    all_modalities = all_modalities[:, :, :, :]
    all_modalities = np.transpose(all_modalities)
    return all_modalities


# Accuracy vs Epoch
def Accuracy_Graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


# Dice Similarity Coefficient vs Epoch
def Dice_coefficient_Graph(history):
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    # plt.title('Dice_Coefficient')
    plt.ylabel('Dice_Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


# Loss vs Epoch
def Loss_Graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()