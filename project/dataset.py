import os
import math
import json
import rasterio
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
matplotlib.use('Agg')

# labels normalization values       
label_norm = {0:["_vv.tif", -17.54, 5.15],
                1:["_vh.tif",-10.68, 4.62],
                2:["_nasadem.tif",166.47, 178.47],
                3:["_jrc-gsw-change.tif", 238.76, 5.15],
                4:["_jrc-gsw-extent.tif", 2.15, 22.71],
                5:["_jrc-gsw-occurrence.tif", 6.50, 29.06],
                6:["_jrc-gsw-recurrence.tif", 10.04, 33.21],
                7:["_jrc-gsw-seasonality.tif", 2.60, 22.79],
                8:["_jrc-gsw-transitions.tif", 0.55, 1.94]}


def transform_data(label, num_classes):
    
    """
    Summary:
        transform label/mask into one hot matrix and return
    Arguments:
        label (arr): label/mask
        num_classes (int): number of class in label/mask
    Return:
        one hot label matrix
    """
    
    return to_categorical(label, num_classes = num_classes) # return the label as one hot encoded



def read_img(directory, in_channels=None, label=False, patch_idx=None, height=512, width=512):
    
    """
    Summary:
        read image with rasterio and normalize the feature
    Arguments:
        directory (str): image path to read
        in_channels (bool): number of channels to read
        label (bool): TRUE if the given directory is mask directory otherwise False
        patch_idx (list): patch indices to read
    Return:
        numpy.array
    """
    
    # for mask images
    if label:
        with rasterio.open(directory) as fmask: # opening the directory
            mask = fmask.read(1)    # read the image (Data from a raster band can be accessed by the band’s index number. Following the GDAL convention, bands are indexed from 1. [int or list, optional] – If indexes is a list, the result is a 3D array, but is a 2D array if it is a band index number.)
            mask[mask == 255] = 0   # convert unlabeled to non-water
            
            if patch_idx:   # if patch is true then returning the extracted patch from mask else returning the whole mask
                return mask[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3]] # extract patch from original mask
            else:
                return mask
    # for features images
    else:
        X = np.zeros((height,width, in_channels)) # get a numpy array of image size
        
        for i in range(in_channels):    # read N number of channels
            tmp_ext = label_norm[i][0]  # get the name of channel (label_norm is dictonary containing {channel name, norm_val_1, norm_val_2})
            with rasterio.open((directory+tmp_ext)) as f:   # opening the image from the directory with channel name
                fea = f.read(1)
            
            X[:,:,i] = (fea - label_norm[i][1]) / label_norm[i][2]  # normalizing the data
            
        if patch_idx:   # if patch is true then returning the extracted patch from image else returning the whole image
            return X[patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3],:] # extract patch from original image
        else:
            return X


def data_split(images, masks, config):
    
    """
    Summary:
        split dataset into train, valid and test
    Arguments:
        images (list): all image directory list
        masks (list): all mask directory
        config (dict): Configuration directory
    Return:
        return the split data.
    """
    
    # spliting dataset where training 80%, validation 10% and test 10%.
    x_train, x_rem, y_train, y_rem = train_test_split(images, masks, train_size = config['train_size'], random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5, random_state=42)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def save_csv(dictionary, config, name):
    
    """
    Summary:
        save csv file
    Arguments:
        dictionary (dict): data as a dictionary object
        config (dict): Configuration directory
        name (str): file name to save
    Return:
        save file
    """
    
    df = pd.DataFrame.from_dict(dictionary) # converting dictionary to pandas dataframe
    df.to_csv((config['dataset_dir']+name), index=False, header=True)   # saving csv from dataframe


def data_path_split(config):
    
    """
    Summary:
        spliting data into train, test, valid
    Arguments:
        config (dict): Configuration directory
    Return:
        save file
    """
    
    paths = pd.read_csv((config['dataset_dir']+"flood-training-metadata.csv"))  # get all the features and corresponding masks directory
    paths = paths.drop_duplicates('chip_id').reset_index(drop=True) # droping duplicate row based on same chip_id
    ids = list(paths.chip_id.values)    # get only the list of chip_ids

    masks = []
    images = []
    
    for i in range(len(ids)):   # seperating all the images and masks
        masks.append(config['dataset_dir']+"train_labels/"+ids[i]+".tif")
        images.append(config['dataset_dir']+"train_features/"+ids[i])
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(images, masks, config)  # spliting dataset where training 80%, validation 10% and test 10%.
    
    # creating dictionary for train, test and validation
    train = {'feature_ids': x_train, 'masks': y_train}
    valid = {'feature_ids': x_valid, 'masks': y_valid}
    test = {'feature_ids': x_test, 'masks': y_test}
    
    # saving dictionary as csv files
    save_csv(train, config, "train.csv")
    save_csv(valid, config, "valid.csv")
    save_csv(test, config, "test.csv")


def class_percentage_check(label):
    
    """
    Summary:
        check class percentage of a single mask image
    Arguments:
        label (numpy.ndarray): mask image array
    Return:
        dict object holding percentage of each class
    """
        
    total_pix = label.shape[0]*label.shape[0]   # calculating total pixels
    class_one = np.sum(label)   # get the total number of pixel labeled as 1
    class_zero_p = total_pix-class_one  # get the total number of pixel labeled as 0
    
    # returning a dictionary containing pixel percent of each class
    return {"zero_class":((class_zero_p/total_pix)*100), 
            "one_class":((class_one/total_pix)*100)
    }



def save_patch_idx(path, patch_size=256, stride=8, test=None, patch_class_balance=None):
    
    """
    Summary:
        finding patch image indices for single image based on class percentage. work like convolutional layer
    Arguments:
        path (str): image path
        patch_size (int): size of the patch image
        stride (int): how many stride to take for each patch image
    Return:
        list holding all the patch image indices for a image
    """
    
    with rasterio.open(path) as t:  # opening the image directory 
        img = t.read(1)
        img[img == 255] = 0 # convert unlabeled to non-water/background
    
    # calculating number patch for given image. Total patch images = patch_height * patch_weight
    patch_height = int((img.shape[0]-patch_size)/stride)+1 # [{(image height-patch_size)/stride}+1]
    patch_weight = int((img.shape[1]-patch_size)/stride)+1 # [{(image weight-patch_size)/stride}+1]
    
    patch_idx = []
    
    for i in range(patch_height):   # image column traverse
        # get the start and end row index
        s_row = i * stride
        e_row = s_row + patch_size
        if e_row <= img.shape[0]:   # check if the taken row index is less then image width 
            for j in range(patch_weight):   # image row traverse
                # get the start and end column index
                start = (j*stride)
                end = start+patch_size
                if end <= img.shape[1]:
                    tmp = img[s_row:e_row, start:end]   # slicing the image
                    percen = class_percentage_check(tmp) # get class percentage
                    
                    if patch_class_balance or test=='test': # for without patch_class_balance or test take all patch images
                        patch_idx.append([s_row, e_row, start, end])
                        
                    else:   # for patch_class_balance take patch image indices based on class percentage
                        if percen["one_class"]>19.0: # take 19% as the threshold for class percentage
                            patch_idx.append([s_row, e_row, start, end])
    return  patch_idx


def write_json(target_path, target_file, data):
    
    """
    Summary:
        save dict object into json file
    Arguments:
        target_path (str): path to save json file
        target_file (str): file name to save
        data (dict): dictionary object holding data
    Returns:
        save json file
    """
    
    if not os.path.exists(target_path): # check the existence of  target directory
        try:
            os.makedirs(target_path)    # making target directory
        except Exception as e:
            print(e)
            raise
    
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)  # writing the jason file


def patch_images(data, config, name):
    
    """
    Summary:
        save all patch indices of all images
    Arguments:
        data: data file contain image paths
        config (dict): configuration directory
        name (str): file name to save patch indices
    Returns:
        save patch indices into file
    """
    
    img_dirs = []
    masks_dirs = []
    all_patch = []
    
    for i in range(len(data)):  # loop through all images
        
        patches = save_patch_idx(data.masks.values[i], patch_size=config['patch_size'], stride=config['stride'], test=name.split("_")[0], patch_class_balance=config['patch_class_balance'])    # fetching patch indices
        
        for patch in patches:   # append data for each patch image
            img_dirs.append(data.feature_ids.values[i])
            masks_dirs.append(data.masks.values[i])
            all_patch.append(patch)
            
    temp = {'feature_ids': img_dirs, 'masks': masks_dirs, 'patch_idx':all_patch}    # dictionary for patch images
    
    write_json((config['dataset_dir']+"json/"), (name+str(config['patch_size'])+'.json'), temp) # save dictionary as json file

# Data Augment class
# ----------------------------------------------------------------------------------------------
class Augment:
    def __init__(self, batch_size, channels, ratio=0.3, seed=42):
        
        """
        Summary:
            initialize class variables
        Arguments:
            batch_size (int): how many data to pass in a single step
            ratio (float): percentage of augment data in a single batch
            seed (int): both use the same seed, so they'll make the same random changes.
        Return:
            class object
        """

        super().__init__()
        self.ratio=ratio
        self.channels= channels
        self.aug_img_batch = math.ceil(batch_size*ratio)
        self.aug = A.Compose([
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Blur(p=0.5),])

    def call(self, feature_dir, label_dir, patch_idx=None):
        
        """
        Summary:
            randomly select a directory and augment data 
            from that specific image and mask
        Arguments:
            feature_dir (list): all train image directory list
            label_dir (list): all train mask directory list
        Return:
            augmented image and mask
        """

        aug_idx = np.random.randint(0, len(feature_dir), self.aug_img_batch) # choose random image from dataset to augment
        features = []
        labels = []
        
        for i in aug_idx:   # get the augmented features and masks
            
            if patch_idx:   # get the patch image and mask if the patch_idx is true
                img = read_img(feature_dir[i], in_channels = self.channels, patch_idx=patch_idx[i])
                mask = read_img(label_dir[i], label=True,patch_idx=patch_idx[i])
                
            else:   # else get the image and mask
                img = read_img(feature_dir[i], in_channels = self.channels)
                mask = read_img(label_dir[i], label=True)
                
            augmented = self.aug(image=img, mask=mask)  # get the augmented the image and mask
            features.append(augmented['image']) # get the list of augmented image
            labels.append(augmented['mask'])    # get the list of augmented mask
        return features, labels



# Dataloader class
# ----------------------------------------------------------------------------------------------

class MyDataset(Sequence):

    def __init__(self, img_dir, tgt_dir, in_channels, 
                 batch_size, num_class, patchify,
                 transform_fn=None, augment=None, weights=None, patch_idx=None):
        
        """
        Summary:
            initialize class variables
        Arguments:
            img_dir (list): all image directory
            tgt_dir (list): all mask/ label directory
            in_channels (int): number of input channels
            batch_size (int): how many data to pass in a single step
            patchify (bool): set TRUE if patchify experiment
            transform_fn (function): function to transform mask images for training
            num_class (int): number of class in mask image
            augment (object): Augment class object
            weight (list): class weight for imblance class
            patch_idx (list): list of patch indices
        Return:
            class object
        """

        self.img_dir = img_dir
        self.tgt_dir = tgt_dir
        self.patch_idx = patch_idx
        self.patchify = patchify
        self.in_channels = in_channels
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.num_class = num_class
        self.augment = augment
        self.weights = weights



    def __len__(self):
        
        """
        return total number of batch to travel full dataset
        """
        
        return math.ceil(len(self.img_dir) // self.batch_size)  # get the total number of batch

    def __getitem__(self, idx):
        
        """
        Summary:
            create a single batch for training
        Arguments:
            idx (int): sequential batch number
        Return:
            images and masks as numpy array for a single batch
        """
        
        # get a single batch for given idx. Ex: for idx=0, batch[0:batch_size] again for idx=1, batch[batch_size:2*batch_size]
        batch_x = self.img_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_y = self.tgt_dir[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        if self.patchify:   # for patchify is true, get patch index for single batch
            batch_patch = self.patch_idx[idx * self.batch_size:(idx + 1) *self.batch_size]
        
        imgs = []
        tgts = []
        
        for i in range(len(batch_x)):   # get all image and target for single batch
            if self.patchify:   # for patchify is true get imgs and targets
                imgs.append(read_img(batch_x[i], in_channels = self.in_channels, patch_idx = batch_patch[i])) # get image from the directory
                
                if self.transform_fn:   # for transform is true, get transform mask for model
                    tgts.append(self.transform_fn(read_img(batch_y[i], label=True, patch_idx = batch_patch[i]), self.num_class))
                else:   # get the mask without transform
                    tgts.append(read_img(batch_y[i], label=True,patch_idx=batch_patch[i]))
            else:   # for patchify is false get imgs and targets
                imgs.append(read_img(batch_x[i], in_channels = self.in_channels))
                
                if self.transform_fn:   # transform mask for model (categorically)
                    tgts.append(self.transform_fn(read_img(batch_y[i], label=True), self.num_class))    # get image from the directory
                else:   # get the mask without transform
                    tgts.append(read_img(batch_y[i], label=True))
        
        if self.augment:    # augment data using Augment class if augment is true
            if self.patchify: # for patchify images
                aug_imgs, aug_masks = self.augment.call(self.img_dir, self.tgt_dir, self.patch_idx) # get augment images and mask randomly
                imgs = imgs+aug_imgs    # adding augmented images
            else:   # without patchify images
                aug_imgs, aug_masks = self.augment.call(self.img_dir, self.tgt_dir) # get augment images and mask randomly
                imgs = imgs+aug_imgs    # adding augmented images

            if self.transform_fn:   # for transform_fn is true, get transform mask (categorically)
                for i in range(len(aug_masks)):
                    tgts.append(self.transform_fn(aug_masks[i], self.num_class))
            else:
                tgts = tgts+aug_masks   # adding augmented masks

        # converting list to numpy array
        tgts = np.array(tgts)
        imgs = np.array(imgs)
        
        
        if self.weights != None:    # for weights is true, return weighted features and lables
            class_weights = tf.constant(self.weights)   # creating a constant tensor
            class_weights = class_weights/tf.reduce_sum(class_weights)  # normalizing the weights
            
            y_weights = tf.gather(class_weights, indices=tf.cast(tgts, tf.int32))   # get weighted mask according to the class weights and targests ([self.paths[i] for i in indexes])

            return tf.convert_to_tensor(imgs), y_weights    # return weighted images and targets
        
        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts)   # return non-weighted images and targets
    

    def get_random_data(self, idx=-1):
        
        """
        Summary:
            randomly chose an image and mask or the given index image and mask
        Arguments:
            idx (int): specific image index default -1 for random
        Return:
            image and mask as numpy array
        """
        
        if idx!=-1:
            idx = idx
        else:
            idx = np.random.randint(0, len(self.img_dir))
        
        imgs = []
        tgts = []
        if self.patchify:   # for patchify is true, get patch images and targets
            imgs.append(read_img(self.img_dir[idx], in_channels=self.in_channels, patch_idx=self.patch_idx[idx]))
            
            if self.transform_fn:   # for transform_fn is true, get transform mask
                tgts.append(self.transform_fn(read_img(self.tgt_dir[idx], label=True,patch_idx=self.patch_idx[idx]), self.num_class))
            else:   # for transform_fn is false, get mask
                tgts.append(read_img(self.tgt_dir[idx], label=True,patch_idx=self.patch_idx[idx]))
        else:   # for patchify is false, get images and targets
            imgs.append(read_img(self.img_dir[idx], in_channels=self.in_channels))
            
            if self.transform_fn:   # for transform_fn is true, get transform mask
                tgts.append(self.transform_fn(read_img(self.tgt_dir[idx], label=True), self.num_class))
            else:   # for transform_fn is false, get mask
                tgts.append(read_img(self.tgt_dir[idx], label=True))

        return tf.convert_to_tensor(imgs), tf.convert_to_tensor(tgts), idx


def get_train_val_dataloader(config):
    
    """
    Summary:
        read train and valid image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        train and valid dataloader
    """

    if not (os.path.exists(config['train_dir'])):   # creating csv files for train, test and validation
        data_path_split(config)
    
    if not (os.path.exists(config["p_train_dir"])) and config['patchify']:  # creating json files for train, test and validation
        print("Saving patchify indices for train and test.....")
        
        # for training
        data = pd.read_csv(config['train_dir'])
        if config["patch_class_balance"]:
            patch_images(data, config, "train_patch_WOC_")
        else:
            patch_images(data, config, "train_patch_")
        
        # for validation
        data = pd.read_csv(config['valid_dir'])
        if config["patch_class_balance"]:
            patch_images(data, config, "valid_patch_WOC_")
        else:
            patch_images(data, config, "valid_patch_")        
    
    if config['patchify']:  # initializing train, test and validatinn for patch images
        print("Loading Patchified features and masks directories.....")
        with open(config['p_train_dir'], 'r') as j:
            train_dir = json.loads(j.read())
        with open(config['p_valid_dir'], 'r') as j:
            valid_dir = json.loads(j.read())
        train_features = train_dir['feature_ids']
        train_masks = train_dir['masks']
        valid_features = valid_dir['feature_ids']
        valid_masks = valid_dir['masks']
        train_idx = train_dir['patch_idx']
        valid_idx = valid_dir['patch_idx']
    
    else:   # initializing train, test and validatinn for images
        print("Loading features and masks directories.....")
        train_dir = pd.read_csv(config['train_dir'])
        valid_dir = pd.read_csv(config['valid_dir'])
        train_features = train_dir.feature_ids.values
        train_masks = train_dir.masks.values
        valid_features = valid_dir.feature_ids.values
        valid_masks = valid_dir.masks.values
        train_idx = None
        valid_idx = None

    print("train Example : {}".format(len(train_features)))
    print("valid Example : {}".format(len(valid_features)))
    
    if config['augment'] and config['batch_size']>1:    # create Augment object and get new batch size if augment is true
        augment_obj = Augment(config['batch_size'], config['in_channels'])
        n_batch_size = config['batch_size']-augment_obj.aug_img_batch # new batch size after augment data for train
    else:
        n_batch_size = config['batch_size']
        augment_obj = None

    if config['weights']:   # get the class weight if weights is true
        weights=tf.constant(config['balance_weights'])
    else:
        weights = None
    
    # create dataloader object for train dataset
    train_dataset = MyDataset(train_features,
                              train_masks,
                              in_channels = config['in_channels'],
                              patchify = config['patchify'],
                              batch_size = n_batch_size,
                              transform_fn = transform_data,
                              num_class = config['num_classes'],
                              augment = augment_obj,
                              weights = weights,
                              patch_idx = train_idx)

    # create dataloader object for validation dataset
    val_dataset = MyDataset(valid_features, valid_masks,
                            in_channels = config['in_channels'],
                            patchify = config['patchify'],
                            batch_size = config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=valid_idx)
    
    return train_dataset, val_dataset


def get_test_dataloader(config):
    
    """
    Summary:
        read test image and mask directory and return dataloader
    Arguments:
        config (dict): Configuration directory
    Return:
        test dataloader
    """

    if not (os.path.exists(config['test_dir'])): # split the dataset if the test directory does not exist
        data_path_split(config)
    
    if not (os.path.exists(config["p_test_dir"])) and config['patchify']:   # creating json files for test dataset
        print("Saving patchify indices for test.....")
        data = pd.read_csv(config['test_dir'])
        patch_images(data, config, "test_patch_")
        
    
    if config['patchify']:  # get the patch images and masks for test dataset
        print("Loading Patchified features and masks directories.....")
        with open(config['p_test_dir'], 'r') as j:
            test_dir = json.loads(j.read())
        test_features = test_dir['feature_ids']
        test_masks = test_dir['masks']
        test_idx = test_dir['patch_idx']
    
    else:   # get images and masks for test dataset
        print("Loading features and masks directories.....")
        test_dir = pd.read_csv(config['test_dir'])
        test_features = test_dir.feature_ids.values
        test_masks = test_dir.masks.values
        test_idx = None

    print("test Example : {}".format(len(test_features)))   # print number of test dataset

    # create dataloader object for test dataset
    test_dataset = MyDataset(test_features, test_masks,
                            in_channels=config['in_channels'],patchify=config['patchify'],
                            batch_size=config['batch_size'], transform_fn=transform_data, 
                            num_class=config['num_classes'],patch_idx=test_idx)
    
    return test_dataset


if __name__ == '__main__':
    train_dir = pd.read_csv("/home/mdsamiul/github_project/flood_water_mapping_segmentation/data/train.csv")
    train_features = train_dir.feature_ids.values
    train_masks = train_dir.masks.values


    train_dataset = MyDataset(train_features, train_masks,
                              in_channels=3, patchify=False,
                              batch_size=64, transform_fn=transform_data,
                              num_class=2, augment=None,
                              weights=None, patch_idx=None)


    #x, y = train_dataset.__getitem__(1)
    for batch in train_dataset:
        x, y = batch
        break
    print(type(x))
    print(tf.shape(x))
    print(type(y))
    print(tf.shape(y))
    print(train_dataset.__len__())