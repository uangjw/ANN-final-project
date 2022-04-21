import os
import random
import shutil

train_path = r'dataset_skin40/train'
val_path = r'dataset_skin40/val'

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
            numlist = random.sample(range(0, len(files)), int(len(files) / 5))
            cur_val_path = os.path.join(val_path, dir)
            for n in numlist:
                filename = files[n]
                old_path = os.path.join(cur_train_path, filename)
                new_path = os.path.join(cur_val_path, filename)
                shutil.copy(old_path, new_path)
            print('finish copy in class ' + dir + '.')
