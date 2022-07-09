''' vitb-R50 no trick '''
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import timm
from timm.loss import LabelSmoothingCrossEntropy


def train_model(model, criterion, optimizer, scheduler, result, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    class_lowest_acc = {}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        result.write(f'Epoch {epoch}/{num_epochs - 1}\n')
        result.write('-' * 10)
        result.write('\n')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            class_corrects = {}

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                i = 0
                for label in labels.data:
                    if preds[i] == label:
                        if label.item() in class_corrects.keys():
                            class_corrects[label.item()] += 1
                        else :
                            class_corrects[label.item()] = 1
                    i += 1
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            epoch_class_acc = {}
            lowest_class = 0
            lowest_class_acc = 1
            for key in class_corrects.keys():
                if phase == 'train':
                    epoch_class_acc[key] = class_corrects[key] / 48
                    # print("class:" + str(key) + ", acc:" + str(epoch_class_acc[key]))
                    if epoch_class_acc[key] < lowest_class_acc:
                        lowest_class = key
                        lowest_class_acc = epoch_class_acc[key]
                else:
                    epoch_class_acc[key] = class_corrects[key] / 12
                    # print("class:" + str(key) + ", acc:" + str(epoch_class_acc[key]))
                    if epoch_class_acc[key] < lowest_class_acc:
                        lowest_class = key
                        lowest_class_acc = epoch_class_acc[key]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            result.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
            
            epoch_class_acc_sorted = sorted(epoch_class_acc.items(), key=lambda d:d[1], reverse=False)
            print(epoch_class_acc_sorted)
            print("class with lowest acc:" + str(lowest_class))
            print("lowest acc:" + str(lowest_class_acc))
            
            class_lowest_acc [lowest_class] = lowest_class_acc

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    result.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    result.write(f'Best val Acc: {best_acc:4f}\n')
    
    print(class_lowest_acc)

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    result = open("vit-result/no-trick-result.txt", "x")
    cudnn.benchmark = True
    plt.ion()

    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
        ]),
        'val' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'dataset_skin40/fold1'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model_ft = timm.create_model('vit_base_resnet50_224_in21k', pretrained=True, num_classes=40)
    model_ft = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=40)
    
    model_ft = model_ft.to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()

    optimizer_ft = optim.SGD(model_ft.parameters(), momentum=0.9, nesterov=True, lr=0.01)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, result, num_epochs=200)

    # torch.save(model_ft.state_dict(), 'method1/model.pth')
    
    result.close()