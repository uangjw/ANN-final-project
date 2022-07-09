''' vitb-R50 / vitb with trick '''
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

from timm.scheduler import StepLRScheduler


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def train_model(model, criterion, optimizer, scheduler, result, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    class_total_cnt = {}
    class_correct_cnt = {}

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
                    if label.item() in class_total_cnt.keys():
                        class_total_cnt[label.item()] += 1
                    else:
                        class_total_cnt[label.item()] = 1
                    if preds[i] == label:
                        if label.item() in class_correct_cnt.keys():
                            class_correct_cnt[label.item()] += 1
                        else :
                            class_correct_cnt[label.item()] = 1
                    i += 1
                
            if phase == 'train':
                scheduler.step(epoch+1)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            result.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
            

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    result.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    result.write(f'Best val Acc: {best_acc:4f}\n')
    
    print(f'Class Acc:')
    result.write('Class Acc:')
    for i in range(len(class_total_cnt)):
        acc = class_correct_cnt[i] / class_total_cnt[i]
        print('class id: '+str(i)+"; acc: "+str(acc))
        result.write('class id: '+str(i)+"; acc: "+str(acc))
    


    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    file_name = "vit-result/vitb-200epoch-trick-fold3.txt"
    if (os.path.isfile(file_name)):
        os.remove(file_name)
    result = open(file_name, "x")
    cudnn.benchmark = True
    plt.ion()

    data_transforms = {
        'train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomApply(transforms=[transforms.RandomAffine(degrees=(5, 15), translate=(0.1, 0.2), scale=(0.8, 0.95))], p=0.5),
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

    data_dir = 'dataset_skin40/fold3'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=40)
    #model_ft = timm.create_model('vit_base_resnet50_224_in21k', pretrained=True, num_classes=40)
    
    model_ft = model_ft.to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()

    # weight_decay
    skip = {}
    if hasattr(model_ft, 'no_weight_decay'):
        skip = model_ft.no_weight_decay()
    parameters = add_weight_decay(model_ft, 0.0001, skip)
    weight_decay = 0.1

    optimizer_ft = optim.SGD(parameters, momentum=0.9, nesterov=True, lr=0.01, weight_decay=weight_decay)    

    noise_range = None
    lr_scheduler = StepLRScheduler(optimizer_ft, decay_t=30, decay_rate=0.1,
                                   warmup_lr_init=0.0001, warmup_t=3, noise_range_t=None, noise_pct=0.67,
                                   noise_std=1., noise_seed=42)

    model_ft = train_model(model_ft, criterion, optimizer_ft, lr_scheduler, result, num_epochs=200)

    # torch.save(model_ft.state_dict(), 'method1/model.pth')
    
    result.close()