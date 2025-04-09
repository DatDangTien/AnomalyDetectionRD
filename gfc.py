import os
import glob
import random
import shutil
from bs4 import BeautifulSoup

def read_xml(xml_path) -> list:
    """
    Read xml file and return all labels
    :param xml_path:
    :return: List of labels.
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'xml')
    labels = soup.find_all('object')
    return [label.find('name').string.lower() for label in labels]

def GFC_format():
    data_path = 'dataset/gfc'
    test_path = os.path.join(data_path, 'test')
    train_path = os.path.join(data_path, 'train/good')

    if os.path.isdir(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)
    if os.path.isdir(train_path):
        shutil.rmtree(train_path)
    os.makedirs(train_path)

    im_non_defect_tot_paths = glob.glob(os.path.join(data_path, 'defect-free images') + "/*.bmp")
    sub_defect_paths = glob.glob(os.path.join(data_path, 'defect images') + "/*")
    im_defect_tot_paths = []
    label_defect_tot_paths = []
    for path in sub_defect_paths:
        im_defect_tot_paths.extend(glob.glob(os.path.join(path, 'test/image') + "/*.bmp"))
        label_defect_tot_paths.extend(glob.glob(os.path.join(path, 'test/label') + "/*.xml"))

    # Train_set
    # for _ in range(len(im_non_defect_tot_paths) // 4):
    for i in range(int(len(im_non_defect_tot_paths) * 0.7)):
        rand_idx = random.randint(0, len(im_non_defect_tot_paths)-1)
        # dst_path = os.path.join(train_path, im_non_defect_tot_paths[rand_idx].split('\\')[-1])
        dst_path = os.path.join(train_path, f'{i}.bmp')
        # print(dst_path)
        shutil.copy(im_non_defect_tot_paths[rand_idx], dst_path)
        del im_non_defect_tot_paths[rand_idx]

    # Test_set
    test_good_path = os.path.join(test_path, 'good')
    if not os.path.isdir(test_good_path):
        os.mkdir(test_good_path)
    test_non_defect_tot_paths = random.sample(im_non_defect_tot_paths, len(im_non_defect_tot_paths))
    for i, path in enumerate(test_non_defect_tot_paths):
        dst_path = os.path.join(test_good_path, f'{i}.bmp')
        # print(dst_path)
        shutil.copy(path, dst_path)

    test_defect_path = os.path.join(test_path, 'defect')
    # test_label_path = os.path.join(test_path, 'label')
    if not os.path.isdir(test_defect_path):
        os.mkdir(test_defect_path)
    # if not os.path.isdir(test_label_path):
    #     os.mkdir(test_label_path)
    
    for i in range(len(im_defect_tot_paths)):
        rand_idx = random.randint(0, len(im_defect_tot_paths)-1)
        # print(label_defect_tot_paths[rand_idx])
        labels = read_xml(label_defect_tot_paths[rand_idx])
        dst_path = os.path.join(test_defect_path, '{}_{}.bmp'.format(i, '_'.join(labels)))
        # print(dst_path)
        shutil.copy(im_defect_tot_paths[rand_idx], dst_path)
        del im_defect_tot_paths[rand_idx], label_defect_tot_paths[rand_idx]
        

if __name__ == '__main__':
    random.seed(111)
    GFC_format()
