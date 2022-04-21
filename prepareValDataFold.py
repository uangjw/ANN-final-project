import os
import shutil

data_path = r'dataset_skin40/train'
train_path = r'dataset_skin40/fold/train'
val_path = r'dataset_skin40/fold/val'

val_range = list(range(0, 12))

if os.path.exists(data_path):
    dirs = os.listdir(data_path)
    for dir in dirs:
        if os.path.isdir(os.path.join(data_path, dir)):
            try:
                os.makedirs(os.path.join(val_path, dir))
                os.makedirs(os.path.join(train_path, dir))
            except FileExistsError:
                print('[WARNING] dir ' + dir + ' already exists.')
            cur_data_path = os.path.join(data_path, dir)
            files = os.listdir(cur_data_path)
            if len(files) > 12:
                numlist = val_range
            else:
                numlist = [0]
            cur_val_path = os.path.join(val_path, dir)
            cur_train_path = os.path.join(train_path, dir)
            for i in range(0, len(files)):
                filename = files[i]
                old_path = os.path.join(cur_data_path, filename)
                if i in numlist:
                    new_path = os.path.join(cur_val_path, filename)
                else:
                    new_path = os.path.join(cur_train_path, filename)
                shutil.copy(old_path, new_path)
                
            print('finish copy in class ' + dir + '.')
