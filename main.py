# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader, random_split
from models.resnet import wide_resnet50_2, wide_resnet101_2
from models.de_resnet import de_wide_resnet50_2, de_wide_resnet101_2
from models.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
from models.convnext import de_convnext_tiny, de_convnext_small, de_convnext_base, de_convnext_large
from models.stage_attn import AdaptiveStages, adap_loss_function
from dataset import MVTecDataset, GFCDataset, train_collate
import torch.backends.cudnn as cudnn
from test import evaluation
from torch.nn import functional as F
import sys
import pickle as pkl
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_parameters(model):
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)/(1024**2)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_function(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def validation(encoder, bn, decoder, layer_attn, val_dataloader, device):
    bn.eval()
    decoder.eval()
    layer_attn.eval()
    with torch.no_grad():
        loss_list = []
        for img, _ in val_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            # loss = loss_function(inputs, outputs)
            loss = adap_loss_function(inputs, outputs, layer_attn(), w_entropy=0.01)
            loss_list.append(loss.item())
    return np.mean(loss_list)

def train(dataset, _class_, filter=None, filter_name=None):
    if dataset == 'mvtec':
        if 'kaggle' in os.getcwd():
            data_path = f'/kaggle/input/mvtec-ad/{_class_}'
        else:
            data_path = f'./dataset/mvtec/{_class_}'
        ckp_path = './checkpoints/mvtec/'
        if not os.path.isdir(ckp_path):
            os.makedirs(ckp_path)
        ckp_path += f'{backbone}_{_class_}.pth'

        train_log_path = './train_logs/mvtec/'
        if not os.path.isdir(train_log_path):
            os.makedirs(train_log_path)
        train_log_path += f'{backbone}_{_class_}.txt'

        train_data = MVTecDataset(root=data_path, image_size=image_size, phase="train", filter=filter)
        # mean_std = train_data.get_meta_data()
        # test_data = MVTecDataset(root=data_path, image_size=image_size, phase="test",
        #                          transform=get_data_transforms(image_size, image_size, filter=filter))

        test_data = MVTecDataset(root=data_path, image_size=image_size, phase="test", filter=filter)
    else:
        if 'kaggle' in os.getcwd():
            data_path = f'/kaggle/input/gfc-ad'
        else:
            data_path = './dataset/gfc'
        ckp_path = f'./checkpoints/gfc/'
        if filter:
            ckp_path += filter_name + '/'
        if not os.path.isdir(ckp_path):
            os.makedirs(ckp_path)
        ckp_path += f'{backbone}_{_class_}.pth'

        train_log_path = f'./train_logs/gfc/'
        if filter:
            train_log_path += filter_name + '/'
        if not os.path.isdir(train_log_path):
            os.makedirs(train_log_path)
        train_log_path += f'{backbone}_{_class_}.txt'

        train_data = GFCDataset(root=data_path, image_size=image_size, phase="train", filter=filter)
        # mean_std = train_data.get_meta_data()
        # test_data = GFCDataset(root=data_path, image_size=image_size, phase="test",
        #                        transform=get_data_transforms(image_size, image_size, filter=filter))

        test_data = GFCDataset(root=data_path, image_size=image_size, phase="test", filter=filter)
    print(_class_)
    start_time = time.time()
    data_path = './dataset'

    if filter:
        loss_path = f'./train_logs/{dataset}/{filter_name}/{backbone}_{_class_}.pkl'
    else:
        loss_path = f'./train_logs/{dataset}/{backbone}_{_class_}.pkl'

    # Clean train_log
    open(train_log_path, 'w').close()

    # # Store preprocess data
    # with open(f'./checkpoints/gfc/{_class_}_metadata.pkl', 'wb') as f:
    #     pkl.dump(mean_std, f)

    # Split train val
    train_len = int(len(train_data) * 0.8)
    train_data, val_data = random_split(train_data, [train_len, len(train_data) - train_len],
                                        generator=torch.Generator().manual_seed(SEED))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   collate_fn=train_collate, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                                 collate_fn=train_collate, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder_fn, decoder_fn = backbone_module[backbone]
    encoder, bn = encoder_fn(pretrained=True)
    decoder = decoder_fn(pretrained=False)
    layer_attn = AdaptiveStages(num_stages=3)
    encoder = encoder.to(device)
    encoder.eval()
    bn = bn.to(device)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters())+list(layer_attn.parameters()),
                                 lr=learning_rate, betas=optimizer_momentum)

    print(count_parameters(decoder) + count_parameters(bn), 'params')
    print(measure_parameters(decoder) + measure_parameters(bn), 'Mb')

    loss_dict = {'train': {}, 'val': {}}
    eva = None
    best_val_loss = float('inf')
    patience_counter = 0
    train_time = time.time()
    for epoch in range(epochs):
        epoch_time = time.time()
        bn.train()
        decoder.train()
        layer_attn.train()
        loss_list = []
        for img, _ in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))#bn(inputs))
            # loss = loss_function(inputs, outputs)
            loss = adap_loss_function(inputs, outputs, layer_attn(), w_entropy=0.01)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        val_time = time.time()
        loss_dict['train'][epoch] = np.mean(loss_list)
        loss_dict['val'][epoch] = validation(encoder, bn, decoder, layer_attn, val_dataloader, device)

        if loss_dict['val'][epoch] < best_val_loss:
            best_val_loss = loss_dict['val'][epoch]
            patience_counter = 0
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping!')
                break

        print('epoch [{}/{}]: loss:{:.4f}, Train time: {:.5f}s, Epoch time: {:.5f}s'.format(epoch + 1,
                                                                                            epochs,
                                                                                            loss_dict['train'][epoch],
                                                                                            val_time - epoch_time,
                                                                                            time.time()-epoch_time))
        if (epoch + 1) % 20 == 0:
            eva = evaluation(encoder, bn, decoder, test_dataloader, device, layer_attn)
            print('AUROC_AL: {}, AUROC_AD: {}, PRO: {}'.format(*eva[:3]))

        #
        # with open(train_log_path, 'a') as f:
        #         f.write('Evaluation: auroc_px, auroc_sp, aupro_px, ap_px, ap_sp\n')
        #         f.write('epoch [{} / {}]'.format(epoch + 1, epochs) + ' '.join([str(i) for i in eva[:-2]]) + '\n')

    with open(train_log_path, 'a') as f:
        f.write('Loading time: {}\nTraining time: {}\n'.format(round(train_time-start_time, 5), round(time.time()-train_time, 5)))
    with open(loss_path, 'wb') as f:
        pkl.dump(loss_dict, f)
    return eva

backbone_module ={
    'wres50': (wide_resnet50_2, de_wide_resnet50_2),
    'wres101': (wide_resnet101_2, de_wide_resnet101_2),
    'resnet50': (wide_resnet50_2, de_wide_resnet50_2),
    'resnet101': (wide_resnet101_2, de_wide_resnet101_2),
    'convnext-t': (convnext_tiny, de_convnext_tiny),
    'convnext-s': (convnext_small, de_convnext_small),
    'convnext-b': (convnext_base, de_convnext_base),
    'convnext-l': (convnext_large, de_convnext_large)
}

if __name__ == '__main__':
    backbones = ['resnet50', 'resnet101', 'wres50', 'wres101',
                 'convnext-t', 'convnext-s', 'convnext-b', 'convnext-l']

    SEED = 111
    setup_seed(SEED)

    image_size = 224
    epochs = 200
    # epochs = 40
    learning_rate = 5e-3
    optimizer_momentum = (0.5, 0.999)
    batch_size = 16
    # batch_size = 8
    backbone = 'wres50'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    patience = 20

    item_list = []
    if sys.argv[1] == 'mvtec':
        if len(sys.argv) > 2:
            item_list = [sys.argv[2]]
        else:
            item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                         'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    else:
        item_list = ['gfc']

    # filter_list = [BilateralFilter(d=5), WaveletFilter()]
    # filter_name_list = ['bilateral', 'wavelet']
    for i in item_list:
        train(sys.argv[1], i)
        # for j in range(len(filter_list)):
        #     train(sys.argv[1], i, filter_list[j], filter_name_list[j])


