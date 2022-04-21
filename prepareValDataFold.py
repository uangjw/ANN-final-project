import os
import random
import shutil

train_path = r'dataset_skin40/train'
val_path = r'dataset_skin40/val'

fold0 = list(range(0, 12))
fold1 = list(range(12, 24))
fold2 = list(range(24, 36))
fold3 = list(range(36, 48))
fold4 = list(range(48, 60))

if os.path.exists(train_path):
    dirs = os.listdir(train_path)
    for dir in dirs:
        if os.path.isdir(os.path.join(train_path, dir)):
            try:
                os.makedirs(os.path.join(val_path, dir))
            except FileExistsError:
                print('[WARNING] dir ' + dir + ' already exists.')
            cur_train_path = os.path.join(train_path, dir)
            files = os.listdir(cur_train_path)
            if len(files) > 12:
                numlist = fold0
            else:
                numlist = [0]
            cur_val_path = os.path.join(val_path, dir)
            for n in numlist:
                filename = files[n]
                old_path = os.path.join(cur_train_path, filename)
                new_path = os.path.join(cur_val_path, filename)
                shutil.copy(old_path, new_path)
            print('finish copy in class ' + dir + '.')
