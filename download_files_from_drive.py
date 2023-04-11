import gdown

folder = "https://drive.google.com/drive/folders/1hiKIIjghYpu6mad-96qzS2Q6ChYJ7B5j"
gdown.download_folder(folder, quiet=False, fuzzy=True)
