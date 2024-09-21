import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color

import math
import imageio
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.spatial.distance
import util

# sampled_responses_path = '../data/sampled_response'

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----

    # handling grayscale images by duplicating them into 3 channels
    if len(img.shape)< 3:
        img = np.stack([img]*3, axis=-1)

    # convert image to lab color space
    img_lab = skimage.color.rgb2lab(img)

    filter_responses = []

    for scale in filter_scales:
        # applying filter to each channel

        # Guassian Filter
        for c in range(3):
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:,:,c],sigma=scale))
        # print(len(filter_responses[0]))  # 310 which is = size of image - SO CALCULATE ACCORDINGLY FOR NUMBER OF SCALES

        # LoG
        for c in range(3):
            filter_responses.append(scipy.ndimage.gaussian_laplace(img_lab[:,:,c],sigma=scale))

        # DoG in x
        for c in range(3):
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:,:,c],sigma=scale,order=[1,0]))
        
        # DoG in y
        for c in range(3):
            filter_responses.append(scipy.ndimage.gaussian_filter(img_lab[:,:,c],sigma=scale,order=[0,1]))

    filter_responses = np.dstack(filter_responses)  # 4 filters * 3 channels * 3 scales
    # print(len(filter_responses))
    return filter_responses

    

# if multi-processing is used
def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # DOUBT: alpha responses is computed per channel right? - YES. The size of the response is - alpha*T x 3F

    # ----- TODO -----

    opts, alpha, img_dir = args # TODO: change img_dir in compute_dictionary

    img_path = join(opts.data_dir, img_dir)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255

    # get image features
    filter_responses = extract_filter_responses(opts, img)

    h, w, d = np.shape(filter_responses) # height, width, no. of channels
    # flattened_img = np.reshape(filter_responses,(h*w,d)) # combining all three channels
    # # print(flattened_img)
    # num_pixels = h * w
    # sampled_indices = np.random.choice(num_pixels, size = alpha, replace = False)
    # sampled_response = flattened_img[sampled_indices, :]

    pixel_row = np.random.randint(0, filter_responses.shape[0], size=alpha, dtype=int)
    pixel_col = np.random.randint(0, filter_responses.shape[1], size=alpha, dtype=int)

    pixel_array_shape = alpha, filter_responses.shape[2]
    pixel_array = np.empty(pixel_array_shape, np.float32)

    for pixel in range(0, alpha):
        for channels in range(0, filter_responses.shape[2]):
            pixel_array[pixel,channels] = filter_responses[pixel_row[pixel], pixel_col[pixel], channels]

    # Remove the args to save files
    mod_args= img_dir.replace('/', '_')
    mod_args= mod_args.replace('.jpg', '.npy')
    fil_resp_file = join(opts.data_dir, 'trained_responses', mod_args)
    np.save(fil_resp_file, pixel_array)
    pass
    # return sampled_response

    # save sampled_responses to a file
    # np.save('%s%d'%(sampled_responses_path,  i), np.asarray(sampled_response))

def compute_dictionary(opts, n_worker):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()

    # ----- TODO -----

    # Make new folder if it does not exist
    new_folder_path = os.path.join(opts.data_dir, 'trained_responses')
    try:
        os.makedirs(new_folder_path)
    except FileExistsError:
        pass

    # get the number of rows of training images
    n_train = len(train_files)
    # arguments for the above function
    args = [(opts, alpha, train_files[id]) for id in range (n_train)]

    pool = multiprocessing.Pool(processes=n_worker)
    pool.map(compute_dictionary_one_image, args)
    pool.close()
    pool.join()

    shape = alpha*n_train, 12*len(opts.filter_scales)
    filter_response = np.empty(shape, np.float32)

    for i in range(0, n_train):
        mod_args= train_files[i].replace('/', '_')
        mod_args= mod_args.replace('.jpg', '.npy')
        file_name_ = join(opts.data_dir, 'trained_responses', mod_args)
        #Load data
        loaded_data = np.load(file_name_, allow_pickle=True)
        #print("Loading data in arr:", opts.alpha*i, opts.alpha*i + opts.alpha,)
        filter_response[opts.alpha*i : opts.alpha*i + opts.alpha, :] = loaded_data[0]

    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_response)
    dictionary = kmeans.cluster_centers_ 

    print("K-means Clustering Completed Successfully")
    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    pass

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    # print(np.shape(dictionary))
    filter_responses = extract_filter_responses(opts, img)

    shape = filter_responses.shape[0], filter_responses.shape[1]
    wordmap = np.empty(shape, np.float32)
    pixel_shape_ = 1, filter_responses.shape[2]
    pixel_arr = np.empty(pixel_shape_, np.float32)
    
    for i in range(0, filter_responses.shape[0]):
        for j in range(0, filter_responses.shape[1]):
            pixel_arr = filter_responses[i, j]
            min_dist_arr_ = scipy.spatial.distance.cdist(pixel_arr.reshape(1,filter_responses.shape[2]), dictionary)    
            # Get Index in the array
            min_index = np.argmin(min_dist_arr_)
            wordmap[i, j] = min_index

    return wordmap