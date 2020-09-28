import numpy as np
from PIL import Image
import h5py
from os import listdir
from os.path import isfile, join
from datetime import datetime

JPG_EXT = 'jpg'
H5_EXT = '.h5'


def jpg2nparr(fname, size=None):
    image = Image.open(fname)
    # image.show()

    if size is not None:
        image = image.resize(size, Image.ANTIALIAS)

    arr = np.array(image)
    return arr


def h5write(datasets, writepath):
    dsets = []

    with h5py.File(writepath, 'w') as hf:
        for k, v in datasets.items():
            dset = hf.create_dataset(k, v['data'].shape, dtype=v['dt'])
            print(v)
            hf[k][:] = v['data']
            dsets.append(dset)

    return dsets


def jpg2h5(set_name, filepath, size, classes,
               write_name=None, writefilepath=""):
    datasets = {}

    img_files = [f for f in listdir(filepath)
                  if isfile(join(filepath, f)) and f.split(".")[-1] == JPG_EXT]

    set_x = np.zeros([0, size[0], size[1], 3])
    set_y = np.zeros([1, 0])
    for img_file in img_files:
        new_arr = jpg2nparr(filepath + img_file, size=size)
        class_find = [(len(c), i) for i, c in enumerate(classes)
                      if c in img_file]
        if class_find == []:
            class_idx = -1
        else:
            class_idx = sorted(class_find, reverse=True).pop(0)[1]

        set_x = np.r_[set_x, [new_arr]]
        set_y = np.c_[set_y, class_idx]

    datasets[set_name + '_set_x'] = {'data': set_x, 'dt': np.uint8}
    datasets[set_name + '_set_y'] = {'data': set_y, 'dt': np.uint8}
    datasets['list_classes'] = {'data': np.array(classes),
                                'dt': h5py.special_dtype(vlen=np.unicode)}

    if write_name is None:
        write_name = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")

    writefilename = set_name + "_" + write_name + H5_EXT

    dsets = h5write(datasets, writefilepath + writefilename)

    conv_result = {"datasets": datasets,
                   "dsets": dsets}

    return conv_result


def jpg2h5set(train_filepath=None, test_filepath=None,\
              train_setname="train", test_setname="test",\
               size=None, classes=None,\
               write_name=None, writefilepath=""):

    conv_train_result = None
    conv_test_result =  None

    if train_filepath is not None:
        conv_train_result = jpg2h5(train_setname, train_filepath, size, classes,
                                   write_name=write_name, writefilepath=writefilepath)

    if test_filepath is not None:
        conv_test_result = jpg2h5(test_setname, test_filepath, size, classes,
                                  write_name=write_name, writefilepath=writefilepath)

    return conv_train_result, conv_test_result
