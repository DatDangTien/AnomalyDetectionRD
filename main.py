import torch
from torch.nn import DataParallel as DP
import numpy as np
import random
import os
from torch.utils.data import DataLoader, random_split
import models.resnet as resnet
import models.de_resnet as de_resnet
import models.convnext as convnext
import models.mambavision as mambavision
from models.stage_attn import AdaptiveStagesFusion, adap_loss_function
from dataset import MVTecDataset, GFCDataset, train_collate
import torch.backends.cudnn as cudnn
from test import evaluation
from torch.nn import functional as F
import pickle as pkl
import time
from argparse import ArgumentParser
from test import test

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
        # print(a[item].shape)
        # print(b[item].shape)
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
    if layer_attn:
        layer_attn.eval()
    with torch.no_grad():
        loss_list = []
        for img, _ in val_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            # loss = loss_function(inputs, outputs)
            loss = adap_loss_function(inputs, outputs, layer_attn, w_entropy=layer_entropy, device=device)
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

        train_data = GFCDataset(root=data_path, image_size=image_size, phase="train", filter=filter, cropped=crop)
        # mean_std = train_data.get_meta_data()
        # test_data = GFCDataset(root=data_path, image_size=image_size, phase="test",
        #                        transform=get_data_transforms(image_size, image_size, filter=filter))

        test_data = GFCDataset(root=data_path, image_size=image_size, phase="test", filter=filter, cropped=crop)
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
    if patience > 0:
        train_len = int(len(train_data) * 0.8)
        train_data, val_data = random_split(train_data, [train_len, len(train_data) - train_len],
                                            generator=torch.Generator().manual_seed(SEED))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   collate_fn=train_collate, num_workers=4)
    if patience > 0:
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                                     collate_fn=train_collate, num_workers=4)
    else:
        val_dataloader = None
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)


    encoder_fn, decoder_fn = backbone_module[backbone]
    encoder, bn = encoder_fn(pretrained=True)
    decoder = decoder_fn(pretrained=False)
    layer_attn = AdaptiveStagesFusion(num_stages=3, trainable=use_layer_attn, w_alpha=w_alpha, inverse=weight_inverse, device=device)
    encoder = encoder.to(device)
    encoder.eval()
    bn = bn.to(device)
    decoder = decoder.to(device)
    layer_attn.to(device)
    layer_attn.freeze()
    freeze_layer_attn = True

    # Layer_attn init:
    dummy_img = torch.randn(1, 3, image_size, image_size, device=device)
    dummy_inputs = encoder(dummy_img)
    layer_attn(dummy_inputs)

    if torch.cuda.device_count() > 1:
        encoder = DP(encoder)
        bn = DP(bn)
        decoder = DP(decoder)
        # layer_attn = DP(layer_attn)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters())+list(layer_attn.parameters()),
                                 lr=learning_rate, betas=optimizer_momentum)


    print(count_parameters(decoder), 'decoder params')
    print(count_parameters(decoder) + count_parameters(bn), 'params')
    print(measure_parameters(decoder) + measure_parameters(bn), 'Mb')

    loss_dict = {'train': {}, 'val': {}}
    eva = None
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_delay = 0

    #  For fusion last epochs:
    #  Init flag: freeze layer_attn
    #  Unfreeze if epoch == 180
    #  Unfreeze if early stop, delay counter = 20

    train_time = time.time()
    for epoch in range(epochs):
        # Unfreeze layer attn
        if use_layer_attn:
            if epoch == epochs - fusion_epochs:
                freeze_layer_attn = False
                print('Unfreeze layer_attn')
                layer_attn.module.unfreeze() if isinstance(layer_attn,DP) else layer_attn.unfreeze()

        epoch_time = time.time()
        bn.train()
        decoder.train()
        if freeze_layer_attn:
            layer_attn.eval()
        else:
            layer_attn.train()
        loss_list = []
        for img, _ in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            embed = bn(inputs)
            outputs = decoder(embed)
            # loss = loss_function(inputs, outputs)
            loss = adap_loss_function(inputs, outputs, layer_attn, w_entropy=layer_entropy, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        val_time = time.time()
        loss_dict['train'][epoch] = np.mean(loss_list)
        if val_dataloader:
            loss_dict['val'][epoch] = validation(encoder, bn, decoder, layer_attn, val_dataloader, device)

        # print
        if epoch == 0 and print_shape:
            print('Encoder: ', [f'{t.shape}, ' for t in inputs])
            print('Bottleneck: ', embed.shape)
            print('Decoder: ', [f'{t.shape}, ' for t in outputs])
        print('epoch [{}/{}]: loss:{:.4f}, Train time: {:.5f}s, Epoch time: {:.5f}s'.format(epoch+1,
                                                                                            epochs,
                                                                                            loss_dict['train'][epoch],
                                                                                            val_time - epoch_time,
                                                                                            time.time()-epoch_time))

        # Debug: Layer attn save miss param.
        if use_layer_attn:
            for name, value in layer_attn.named_parameters():
                if name.replace('module','').startswith('weight'):
                    print(f'{name}: {value.data}\n')

        if (epoch + 1) % 10 == 0:
            # Inverse adap weight for evaluation
            # layer_attn.module.set_inverse() if isinstance(layer_attn, DP) else layer_attn.set_inverse()
            eva = evaluation(encoder, bn, decoder, test_dataloader, device, layer_attn)
            # Inverse back for training
            # layer_attn.module.set_inverse() if isinstance(layer_attn, DP) else layer_attn.set_inverse()
            print('AUROC_AD: {}, AUROC_AL: {},  PRO: {}'.format(eva[0], eva[2], eva[3]))

            if patience == 0:
                torch.save({'bn': bn.state_dict(),
                            'decoder': decoder.state_dict(),
                            'layer_attn': layer_attn.state_dict()}, ckp_path)

        if patience > 0:
            if loss_dict['val'][epoch] < best_val_loss:
                print(f'Best epoch: {epoch + 1}')
                best_val_loss = loss_dict['val'][epoch]
                patience_counter = 0
                torch.save({'bn': bn.state_dict(),
                            'decoder': decoder.state_dict(),
                            'layer_attn': layer_attn.state_dict()}, ckp_path)
            else:
                patience_counter += 1
                # Out of patience -> start layer_attn unfreeze
                if patience_counter >= patience:
                    if use_layer_attn:
                        if freeze_layer_attn:
                            freeze_layer_attn = False
                            early_stop_delay = fusion_epochs
                            print('Unfreeze layer_attn')
                            layer_attn.module.unfreeze() if isinstance(layer_attn, DP) else layer_attn.unfreeze()
                        else:
                            # Layer Attn 20 epochs fixed.
                            continue
                    else:
                        print('Early stop!')
                        break

            if early_stop_delay == 1:
                print('Early stop!')
                break

            if early_stop_delay > 0:
                early_stop_delay -= 1



        #
        # with open(train_log_path, 'a') as f:
        #         f.write('Evaluation: auroc_px, auroc_sp, aupro_px, ap_px, ap_sp\n')
        #         f.write('epoch [{} / {}]'.format(epoch + 1, epochs) + ' '.join([str(i) for i in eva[:-2]]) + '\n')

    with open(train_log_path, 'a') as f:
        f.write('Loading time: {}\nTraining time: {}\n'.format(round(train_time-start_time, 5), round(time.time()-train_time, 5)))
    with open(loss_path, 'wb') as f:
        pkl.dump(loss_dict, f)
    return eva

def Parser():
    parser = ArgumentParser(description="Train RD4AD model")
    parser.add_argument('-is', '--image_size', type=int, default=224, help='Image resolution')
    parser.add_argument('-be', '--backbone', type=str, default='wres50', help='Backbone model name')
    parser.add_argument('-d', '--dataset', type=str, default='mvtec', help='Dataset name')
    parser.add_argument('-c', '--dclass', type=str, default='', help='Data class.')
    parser.add_argument('-w', '--layer_weights', type=int, default=0,
                        choices=[0,1,2], help='Layer weights flag, 0: no weights, 1: adaptive weight, 2: inverse adaptive weight')
    parser.add_argument('-wa', '--w_alpha', type=float, default=1, help='Adaptive weight alpha, alpha<1: weight sharper.')
    parser.add_argument('-s', '--seed', type=int, default=111, help='Seed number')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of train epochs')
    parser.add_argument('-fe', '--fusion_epochs', type=int, default=40, help='Number of fusion epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('-pa', '--patience', type=int, default=0, help='Early stop patience')
    parser.add_argument('-er', '--entropy_rate', type=float, default=0.05, help='Entropy rate in loss')
    parser.add_argument('-p', '--print_shape', type=bool, default=False, help='Print shape of each module')
    parser.add_argument('-cr', '--crop', type=bool, default=False, help='Crop GFC images')
    return parser.parse_args()


backbone_module ={
    'wres50': (resnet.wide_resnet50_2, de_resnet.de_wide_resnet50_2),
    'wres101': (resnet.wide_resnet101_2, de_resnet.de_wide_resnet101_2),
    'resnet50': (resnet.wide_resnet50_2, de_resnet.de_wide_resnet50_2),
    'resnet101': (resnet.wide_resnet101_2, de_resnet.de_wide_resnet101_2),
    'convnext-t': (convnext.convnext_tiny, convnext.de_convnext_tiny),
    'convnext-s': (convnext.convnext_small, convnext.de_convnext_small),
    'convnext-b': (convnext.convnext_base, convnext.de_convnext_base),
    'convnext-l': (convnext.convnext_large, convnext.de_convnext_large),
    'mambavision-t': (mambavision.mambavision_t, mambavision.demambavision_t),
    'mambavision-s': (mambavision.mambavision_s, mambavision.demambavision_s),
    'mambavision-b': (mambavision.mambavision_b21k, mambavision.demambavision_b),
    'mambavision-l': (mambavision.mambavision_l, mambavision.demambavision_l),
}

if __name__ == '__main__':
    backbones = ['resnet50', 'resnet101', 'wres50', 'wres101',
                 'convnext-t', 'convnext-s', 'convnext-b', 'convnext-l',
                 'mambavision-t', 'mambavision-s', 'mambavision-b', 'mambavision-l']

    args = Parser()

    SEED = args.seed
    setup_seed(SEED)

    kwargs = {}
    image_size = args.image_size
    epochs = args.epochs
    fusion_epochs = args.fusion_epochs
    use_layer_attn = (args.layer_weights > 0)
    weight_inverse = (args.layer_weights == 2)
    layer_entropy = args.entropy_rate
    learning_rate = args.learning_rate
    optimizer_momentum = (0.5, 0.999)
    batch_size = args.batch_size
    backbone = args.backbone
    patience = args.patience
    print_shape = args.print_shape
    crop = args.crop
    w_alpha = args.w_alpha

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)


    item_list = []
    res_path = ''

    if args.dataset == 'mvtec':
        if args.dclass != '':
            item_list = [args.dclass]
        else:
            item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                         'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        res_path = f'./result/mvtec/'
    else:
        # gfc dataset only 1 class
        item_list = ['gfc']
        res_path = f'./result/gfc/'

    for i in item_list:
        train(args.dataset, i)

    # if not os.path.isdir(res_path):
    #     os.makedirs(res_path)
    # res_path += 'benchmark.txt'
    # res_list = []
    # with open(res_path, 'a') as f:
    #     f.write('----------------------------\n')
    #     f.write(backbone + '\n')
    #     f.write(str(image_size) + '\n')
    #     f.write('\tAUROC_AD, AP_AD, AUROC_AL, PRO, AP_AL, Overkill, Underkill\n')
    #     for i in item_list:
    #         res_class = test(args.dataset, i)
    #         res_list.append(res_class)
    #         f.write(i.capitalize() + ' ' + ' '.join([str(me_num) for me_num in res_class]) + '\n')
    #     res_avr = [0] * len(res_class)
    #     for cl in res_list:
    #         for ind, me in enumerate(cl):
    #             if me:
    #                 res_avr[ind] += me
    #     res_avr = [str(round(res_me / len(item_list), 3)) for res_me in res_avr]
    #     if len(item_list) > 1:
    #         f.write('Avr {}\n'.format(' '.join(res_avr)))


    # filter_list = [BilateralFilter(d=5), WaveletFilter()]
    # filter_name_list = ['bilateral', 'wavelet']
    # for i in item_list:
    #     train(sys.argv[3], i)
        # for j in range(len(filter_list)):
        #     train(sys.argv[1], i, filter_list[j], filter_name_list[j])


