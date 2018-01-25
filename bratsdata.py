import numpy as np
import nibabel as nib
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
import copy
from skimage.transform import rotate
from scipy.ndimage.morphology import binary_dilation
import tensorflow as tf
import deep_api
import time
from tqdm import tqdm
ext = [
    '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz','mask.nii.gz', '_seg.nii.gz'
]
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def one_hot_encoding(x, labels=[0, 1, 2, 4]):
    return np.stack([1 * (x == l) for l in labels], axis=-1)


def read_instance(filedir, ext=[
    '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz', '_seg.nii.gz'
]):
    filename = os.path.split(filedir)[-1]
    imgdata = []
    for e in ext:
        imgdata.append(nib.load(glob.glob(os.path.join(filedir, '*' + e))[0]))
    return {
            'data':[x.get_data() for x in imgdata],
            'affine':imgdata[0].get_affine()
           }


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


class BratsData(object):
    def __init__(self,
                 parent_dir='data/',
                 test_split=0.2,
                 validation_split=0.25,
                 hgg=True,
                 lgg=False,
                 tumour=False,
                 batch_size=1,
                 channel_axis=0):
        self._parent_dir = parent_dir
        self.tumour = tumour
        self.TrainData = []
        self.TestData = []
        self.ValidationData = []
        self.channel_axis = channel_axis
        if hgg:
            hgg_dir = os.path.join(self._parent_dir, 'HGG')
            hgg_files = glob.glob(os.path.join(hgg_dir, '*'))
            self.splitter(hgg_files, test_split, validation_split)

        if lgg:
            lgg_dir = os.path.join(self._parent_dir, 'LGG')
            lgg_files = glob.glob(os.path.join(lgg_dir, '*'))
            self.splitter(lgg_files, test_split, validation_split)
        self.batch_size = batch_size

    def splitter(self, filelist, test_split, validation_split):
        Data, Test_set = train_test_split(filelist,
                                          test_size=test_split,
                                          random_state=2017)
        Train_set, Validation_set = train_test_split(Data,
                                                     test_size=validation_split,
                                                     random_state=2017)
        self.TrainData.extend(Train_set)
        self.ValidationData.extend(Validation_set)
        self.TestData.extend(Test_set)

    def get_data(self, filelist, randomize=True, skip=True, augmentation=False, modes=[0]):
        temp_filelist = copy.copy(filelist)
        if randomize:
            random.shuffle(temp_filelist)
        for filename in temp_filelist:
            data_dict= read_instance(filename)
            imgdata = data_dict['data']
            affine = data_dict['affine']
            name=os.path.split(filename)[-1]
            normalized_img = [
                normalize(x) for (x, i) in zip(imgdata, range(4))
            ]

            for i in range(155):
                Xtrain = np.array([x[:, :, i]
                                   for x in normalized_img]).transpose(2,1,0)

                Ytrain = imgdata[-1][:, :, i]
                mask = imgdata[-2][:, :, i]
                count = np.count_nonzero(Ytrain)
                if skip and count == 0 :
                    continue
                air = 1*(mask==0)
                tumour  = 1*(Ytrain>0)
                brain = 1*np.all([mask!=0, tumour==0], axis=0)
                if self.tumour:
                    Ytrain = np.stack([air,brain,tumour], axis=-1).transpose(1,0,2)
                else:
                    Ytrain[Ytrain>0]+=1
                    Ytrain[brain==1]=1
                    Ytrain = one_hot_encoding(Ytrain, labels=[0,1,2,3,5]).transpose(1,0,2)

                if not augmentation:
                    yield {'name': name,
                           'affine':affine,
                           'X':Xtrain,
                           'Y':Ytrain}
                    continue
                if np.random.rand()>0.5:
                    axis = np.random.choice([1])
                    Xtrain = np.flip(Xtrain, axis=axis)
                    Ytrain = np.flip(Ytrain, axis=axis)
                # else:
                #     time = np.random.choice([1,2,3])
                #     axes=np.random.choice([0,1], 2, replace=False)
                #     Xtrain = np.rot90(Xtrain, k=time, axes=axes)
                #     Ytrain = np.rot90(Ytrain, k=time, axes=axes)
                yield {'name':name,
                       'affine':affine,
                       'X':Xtrain,
                       'Y':Ytrain}

    def get_minibatches(self, data, batch_size=1):
        iterator = data
        while (True):
            batch = {}

            try:
                for i in range(batch_size):
                    new_slice = next(iterator)
                    if len(batch.keys())==0:
                        batch = {key:[item] for key, item in new_slice.items()}
                    else:
                        for key in batch.keys():
                            batch[key].append(new_slice[key])
                if 'X' in batch.keys():
                    batch['X'] = np.array(batch['X'])
                if 'Y' in batch.keys():
                    batch['Y'] = np.array(batch['Y'])

                yield batch
            except StopIteration:
                raise StopIteration

    def get_train_data(self, augmentation=True, skip=True, randomize=True):
        return self.get_minibatches(
            self.get_data(self.TrainData, skip=skip, randomize=randomize, augmentation=augmentation),
            batch_size=self.batch_size)

    def get_test_data(self, skip=True, randomize=True):
        return self.get_minibatches(
            self.get_data(self.TestData, skip=skip, randomize=randomize),
            batch_size=self.batch_size)

    def get_validation_data(self, skip=True, randomize=True):
        return self.get_minibatches(
            self.get_data(self.ValidationData, skip=skip, randomize=randomize),
            batch_size=self.batch_size)

class BratsValidationData(object):
    def __init__(self,
                 parent_dir='/media/brats/0d4a2225-d6b1-4b80-94fd-7c8ae0b1fa102/MGG/brats-segmentation/Brats17ValidationData/',
                 batch_size=1):
        self._parent_dir = parent_dir
        self._batch_size = batch_size
        self.dirs = glob.glob(os.path.join(self._parent_dir,'*'))

    def read_instance(self, filedir, ext=[
        '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz']):
        filename = os.path.split(filedir)[-1]
        imgdata = []
        for e in ext:
            imgdata.append(nib.load(os.path.join(filedir, filename + e)))
        return [x.get_data() for x in imgdata],imgdata[0].get_affine()

    def normalize(self, x):
        return (x - np.mean(x)) / np.std(x)

    def get_slices(self, filelist):
        for filename in filelist:
            imgdata, affine = self.read_instance(filename)
            name=os.path.split(filename)[-1]
            normalized_img = [
                normalize(x) for (x, i) in zip(imgdata, range(4))
            ]
            for i in range(155):
                Xtrain = np.array([x[:, :, i]
                                   for x in normalized_img]).transpose(
                                       2, 1, 0)
                yield {'X':Xtrain,
                       'affine':affine,
                       'name':name,
                       'index':i}

    def get_minibatches(self, data, batch_size=1):
        iterator = data
        while (True):
            batch = {}
            try:
                for i in range(batch_size):
                    new_slice = next(iterator)
                    if len(batch.keys())==0:
                        batch = {key:[item] for key, item in new_slice.items()}
                    else:
                        for key in batch.keys():
                            batch[key].append(new_slice[key])
                if 'X' in batch.keys():
                    batch['X'] = np.array(batch['X'])
                if 'Y' in batch.keys():
                    batch['Y'] = np.array(batch['Y'])
                yield batch
            except StopIteration:
                raise StopIteration

    def get_data(self):
        return self.get_minibatches(self.get_slices(self.dirs), batch_size=self._batch_size)

class BratsEnsembleData(object):
    def __init__(self,
                 parent_dir='results/ensemble/',
                 batch_size=1):
        self.models = glob.glob(os.path.join(parent_dir,'brats_*'))
        self.patients = set()
        for model in self.models:
            patients = [os.path.split(x)[-1] for x in glob.glob(os.path.join(model,'*.nii.gz'))]
            self.patients.update(set(patients))

    def read_instance(self,
                      patient):
        pred_data = {'model':[], 'pred':[], 'affine':[]}
        for model in self.models:
            prediction=nib.load(os.path.join(model,patient))
            pred_data['model'].append(model)
            pred_data['pred'].append(prediction.get_data())
            pred_data['affine'].append(prediction.get_affine())

    def get_data(self):
        pass

class BratsTriAxialData(object):
    def __init__(self,
                 parent_dir='../Brats17TrainingData',
                 test_split=0.2,
                 validation_split=0.25,
                 hgg=True,
                 lgg=False,
                 patch_size=(33,33,33),
                 batch_size=1):
        self._parent_dir = parent_dir
        self._patch_size = patch_size
        self.TrainData = []
        self.TestData = []
        self.ValidationData = []
        if hgg:
            hgg_dir = os.path.join(self._parent_dir, 'HGG')
            hgg_files = glob.glob(os.path.join(hgg_dir, '*'))
            self.splitter(hgg_files, test_split, validation_split)

        if lgg:
            lgg_dir = os.path.join(self._parent_dir, 'LGG')
            lgg_files = glob.glob(os.path.join(lgg_dir, '*'))
            self.splitter(lgg_files, test_split, validation_split)
        self.batch_size = batch_size

    def splitter(self, filelist, test_split, validation_split):
        Data, Test_set = train_test_split(filelist,
                                          test_size=test_split,
                                          random_state=2017)
        Train_set, Validation_set = train_test_split(Data,
                                                     test_size=validation_split,
                                                     random_state=2017)
        self.TrainData.extend(Train_set)
        self.ValidationData.extend(Validation_set)
        self.TestData.extend(Test_set)

    def window(self, data, center, patch_size):
        corner1 = [center[i] - patch_size[i]//2 for i in range(3)]
        corner2 = [corner1[i] + patch_size[i] for i in range(3)]
        patch = np.concatenate([
                          data[corner1[0]:corner2[0], corner1[1]:corner2[1],             center[2], :],
                          data[            center[0], corner1[1]:corner2[1], corner1[2]:corner2[2], :],
                          data[corner1[0]:corner2[0],             center[1], corner1[2]:corner2[2], :]
                          ], axis=-1)
        return {
         'patch':patch,
         'corner1':corner1,
         'corner2':corner2
         }
    def read_roi(self, filename):
        ext='_roi.nii.gz'
        path = glob.glob(os.path.join(filename, '*' + ext))
        if len(path)>0:
            roi, = read_instance(filename, ext=[ext])
            return roi, True
        else:
            return None, False

    def write_roi(self, roi, filename, affine):
        nii_img = nib.Nifti1Image(1*roi, affine)
        nii_img.set_data_dtype(np.uint8)
        name = os.path.split(filename)[-1]
        nib.save(nii_img,os.path.join(filename, name+'_roi.nii.gz'))



    def get_data(self, filelist, randomize=True, skip=True, modes=[0]):
        temp_filelist = copy.copy(filelist)
        if randomize:
            random.shuffle(temp_filelist)
        for filename in temp_filelist:
            data_dict = read_instance(filename)
            imgdata = data_dict['data']
            affine = data_dict['affine']
            name=os.path.split(filename)[-1]
            X = np.stack([
                normalize(x) for (x, i) in zip(imgdata, range(4))
            ], axis=-1)
            Y = imgdata[-1]
            print(Y.shape)
            ROI, flag = self.read_roi(filename)
            if not flag:
                start = time.time()
                ROI = binary_dilation(Y, structure=np.ones((20,20,20)), iterations=1)
                end = time.time()
                print('Created ROI in', end - start)
                self.write_roi(ROI, filename, affine)
            centers = np.array(np.where(ROI)).T
            Y_one_hot = one_hot_encoding(Y)
            print(Y_one_hot.shape)
            for center in centers:
                cx,cy,cz = center
                Ytrain = Y_one_hot[cx,cy,cz]
                xdict = self.window(X, center, self._patch_size)
                Xtrain = xdict['patch']
                print(Xtrain.shape, Ytrain.shape)
                yield {'name': name,
                       'center': center,
                       'X':Xtrain,
                       'Y':Ytrain}

    def get_minibatches(self, data, batch_size=1):
        iterator = data
        while (True):
            batch = {}

            try:
                for i in range(batch_size):
                    new_slice = next(iterator)
                    if len(batch.keys())==0:
                        batch = {key:[item] for key, item in new_slice.items()}
                    else:
                        for key in batch.keys():
                            batch[key].append(new_slice[key])
                if 'X' in batch.keys():
                    batch['X'] = np.array(batch['X'])
                if 'Y' in batch.keys():
                    batch['Y'] = np.array(batch['Y'])

                yield batch
            except StopIteration:
                raise StopIteration

    def get_train_data(self, skip=True, randomize=True):
        return self.get_minibatches(
            self.get_data(self.TrainData, skip=skip, randomize=randomize),
            batch_size=self.batch_size)

    def get_test_data(self, skip=True, randomize=True):
        return self.get_minibatches(
            self.get_data(self.TestData, skip=skip, randomize=randomize),
            batch_size=self.batch_size)

    def get_validation_data(self, skip=True, randomize=True):
        return self.get_minibatches(
            self.get_data(self.ValidationData, skip=skip, randomize=randomize),
            batch_size=self.batch_size)


if __name__ == '__main__':
    lgg_data = BratsData(
        parent_dir='../Brats17TrainingData/',
        batch_size=40,
        tumour=False,
        hgg=True,
        lgg=True)
    freq = np.array([0,0,0,0,0])
    for batch in tqdm(lgg_data.get_train_data()):
        # print(batch['X'].shape, batch['Y'].shape)
        freq += np.sum(batch['Y'], axis=(0,1,2))
    print(freq)
