import torch
from dataset import load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2, wide_resnet101_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2, de_wide_resnet101_2
from dataset import MVTecDataset, GFCDataset, get_data_transforms
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, auc, roc_curve
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import os
import shutil


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    # return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def visualize_loss(dataset, _class_):
    train_log_path = f'./train_logs/{dataset}/' + 'wres50_' +_class_+'.pkl'
    loss = pickle.load(open(train_log_path, 'rb'))
    epochs = range(1, len(loss['train'].keys()) + 1)
    plt.plot(epochs, loss['train'].values(), label='train loss')
    plt.plot(epochs, loss['val'].values(), label='val loss')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(loss['train'].keys()) + 1, 1))
    plt.legend()
    plt.show()

def evaluation(encoder, bn, decoder, dataloader, device, _class_=None, predict=None):
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    auroc_px = None
    auroc_sp = None
    aupro_sp = None
    ap_px = None
    ap_sp = None
    overkill = None
    underkill = None


    with torch.no_grad():
        for img, gt, label, _ in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            if gt.isnan().any():
                gt_list_sp.append(label.cpu().numpy().astype(int))
            else:
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                if label.item() != 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))
                gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
                gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_px.extend(anomaly_map.ravel())
            pr_list_sp.append(np.max(anomaly_map))

            # ano_score = (pr_list_sp - np.min(pr_list_sp)) / (np.max(pr_list_sp) - np.min(pr_list_sp))
            # vis_data = {}
            # vis_data['Anomaly Score'] = ano_score
            # vis_data['Ground Truth'] = np.array(gt_list_sp)
            # print(type(vis_data))
            # np.save('vis.npy',vis_data)
            # with open('{}_vis.pkl'.format(_class_), 'wb') as f:
            #    pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)

        if not gt.isnan().any():
            aupro_sp = round(np.mean(aupro_list), 3)
            ap_px = round(average_precision_score(gt_list_px, pr_list_px), 3)

        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 3)

        if predict:
            fpr, tpr, thres = roc_curve(gt_list_sp, pr_list_sp)
            optimal_ind = np.argmax(np.sqrt(tpr * (1 - fpr)))
            overkill = round(fpr[optimal_ind], 3)
            underkill = round(1 - tpr[optimal_ind], 3)

            pr_list = [0 if i < thres[optimal_ind] else 1 for i in pr_list_sp]
            with open(predict, 'w') as f:
                f.write('Threshold: {}\n'.format(round(thres[optimal_ind], 3)))
                for i in range(len(pr_list)):
                    f.write('{}, {} {}\n'.format(i + 1, pr_list[i], round(pr_list_sp[i], 3)))


    return auroc_px, auroc_sp, aupro_sp, ap_px, ap_sp, overkill, underkill

def test(dataset, _class_):
    print(_class_)

    if 'kaggle' in os.getcwd():
        test_path = f'/kaggle/input/{dataset}-ad'
    else:
        test_path = f'./dataset/{dataset}'
    test_path += '/' + _class_ if dataset == 'mvtec' else ''
    ckp_path = f'./checkpoints/{dataset}/{backbone}_' + _class_ + '.pth'
    predict_path = f'./result/{dataset}/{backbone}/predict/'
    if not os.path.isdir(predict_path):
        os.makedirs(predict_path)
    predict_path += _class_ + '_predict.txt'

    # Load preprocess metadata
    try:
        with open(f'./checkpoints/{dataset}/{_class_}_metadata.pkl', 'rb') as f:
            mean_std = pickle.load(f)
    except:
        mean_std = None
    print(mean_std)
    transform = get_data_transforms(image_size, image_size, mean_std)

    if dataset == 'mvtec':
        test_data = MVTecDataset(root=test_path, image_size=image_size, phase="test", transform=transform)
    else:
        test_data = GFCDataset(root=test_path, image_size=image_size, phase="test", transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    print(backbone)
    if backbone == 'wres50':
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False)
    elif backbone == 'wres101':
        encoder, bn = wide_resnet101_2(pretrained=True)
        decoder = de_wide_resnet101_2(pretrained=False)
    encoder = encoder.to(device)
    encoder.eval()
    bn = bn.to(device)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    result_metrics = evaluation(encoder, bn, decoder, test_dataloader, device,_class_,predict_path)
    print(f'{_class_}: ' + ' '.join([str(me_num) for me_num in result_metrics]))
    return result_metrics

def visualize(dataset, _class_):
    print(_class_)

    if 'kaggle' in os.getcwd():
        test_path = f'/kaggle/input/{dataset}-ad'
    else:
        test_path = f'./dataset/{dataset}'
    test_path += '/' + _class_ if dataset == 'mvtec' else ''

    ckp_path = f'./checkpoints/{dataset}/{backbone}_{_class_}.pth'

    result_path = f'./result/{dataset}/{backbone}/images/'
    result_path += f'{_class_}/' if dataset == 'mvtec' else ''
    if os.path.isdir(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
    result_heat = result_path + 'heat/'
    result_ori = result_path + 'ori/'
    result_gt = result_path + 'gt/'
    if not os.path.isdir(result_heat):
        os.mkdir(result_heat)
    if not os.path.isdir(result_ori):
        os.mkdir(result_ori)
    if not os.path.isdir(result_gt):
        os.mkdir(result_gt)

    # Load preprocess metadata
    try:
        with open(f'./checkpoints/{dataset}/{_class_}_metadata.pkl', 'rb') as f:
            mean_std = pickle.load(f)
    except:
        mean_std = None
    print(mean_std)
    transform = get_data_transforms(image_size, image_size, mean_std)

    if dataset == 'mvtec':
        test_data = MVTecDataset(root=test_path, image_size=image_size, phase="test", transform=transform)
    else:
        test_data = GFCDataset(root=test_path, image_size=image_size, phase="test", transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    print(backbone)
    if backbone == 'wres50':
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False)
    elif backbone == 'wres101':
        encoder, bn = wide_resnet101_2(pretrained=True)
        decoder = de_wide_resnet101_2(pretrained=False)
    encoder = encoder.to(device)
    encoder.eval()
    bn = bn.to(device)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    count = 0
    with torch.no_grad():
        for img, gt, label, typ in test_dataloader:

            # if (label.item() == 0):
            #     continue
            #if count <= 10:
            #    count += 1
            #    continue

            decoder.eval()
            bn.eval()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            #inputs.append(feature)
            #inputs.append(outputs)
            #t_sne(inputs)


            # anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map, amp_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_RGB2BGR)
            # img = img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255
            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)

            cv2.imwrite('{}{}_{}.png'.format(result_ori, count, typ[0]), img)
            cv2.imwrite('{}{}_{}.png'.format(result_heat, count, typ[0]), ano_map)
            if not gt.isnan().any():
                gt = gt.cpu().numpy().astype(int).squeeze((0,1))*255
                cv2.imwrite('{}{}_{}.png'.format(result_gt, count, typ[0]), gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
            #ano_map = show_cam_on_image(img, ano_map)
            #cv2.imwrite(str(name) + '.png', ano_map)
            #plt.imshow(ano_map)
            #plt.axis('off')
            #plt.savefig(str(name) + '.png')
            #plt.show()
            #    name+=1
            count += 1
            #if count>20:
            #    return 0
            #assert 1==2
        print(count)


def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            #if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            #print(inputs[-1].shape)
            outputs = decoder(bn(inputs))


            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            #anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
            #plt.imshow(ano_map)
            #plt.axis('off')
            #plt.savefig('ad.png')
            #plt.show()

            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            #count += 1
            #if count>40:
            #    return 0
                #assert 1==2
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        #print(type(vis_data))
        #np.save('vis.npy',vis_data)
        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def detection(encoder, bn, decoder, dataloader,device,_class_):
    #_, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)


            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1


        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean

import sys
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    backbone = 'wres101'
    image_size = 256
    
    item_list = []
    res_path = ''
    if sys.argv[2] == 'mvtec':
        if len(sys.argv) > 3:
            item_list = [sys.argv[3]]
        else:
            item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                         'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        res_path = f'./result/mvtec/'
    elif sys.argv[2] == 'gfc':
        item_list = ['gfc']
        res_path = f'./result/gfc/'

    if sys.argv[1] == 'test':
        if not os.path.isdir(res_path):
            os.makedirs(res_path)
        res_path += 'benchmark.txt'
        res_list = []
        with open(res_path, 'a') as f:
            f.write('----------------------------\n')
            f.write(backbone + '\n')
            f.write('\tauroc_px, auroc_sp, aupro_px, ap_px, ap_sp, overkill, underkill\n')
            for i in item_list:
                res_class = globals()[sys.argv[1]](sys.argv[2], i)
                res_list.append(res_class)
                f.write(i + ' ' + ' '.join([str(me_num) for me_num in res_class]) + '\n')
            res_avr = [0] * len(res_class)
            for cl in res_list:
                for ind, me in enumerate(cl):
                    if me:
                        res_avr[ind] += me
            res_avr = [str(round(res_me / len(item_list), 3)) for res_me in res_avr]
            if len(item_list) > 1:
                f.write('Avr {}\n'.format(' '.join(res_avr)))
    # elif sys.argv[1] == 'visualize':
    else:
        for i in item_list:
            globals()[sys.argv[1]](sys.argv[2], i)






