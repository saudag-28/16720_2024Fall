import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import imageio

import visual_words
from matplotlib import pyplot as plt


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K

    # flatten the wordmap array
    wordmap_1d = wordmap.flatten()

    hist, bins = np.histogram(wordmap_1d, bins = K, range=(0,K))
    sum = np.sum(hist)

    if sum!=0:
        hist = hist/sum

    # print(hist)
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----

    weights = []
    hist_all = []
    h, w = np.shape(wordmap)

    # calculating weights for each layer
    for l in range(L+1):
        if l == 0 or l == 1:
            weights.append(2.0**(-L))
        else:
            weights.append(2.0**(l-L-1))

    for l in range(L+1):
        num_cells = 2 ** l
        cell_h = h // num_cells
        cell_w = h // num_cells

        for i in range(num_cells):
            for j in range(num_cells):
                cell_wordmap = wordmap[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
                hist = get_feature_from_wordmap(opts, cell_wordmap)
                weighted_hist = hist * weights[l]
                hist_all.append(weighted_hist)

    hist_all = np.concatenate(hist_all)
    hist_all /= np.sum(hist_all)

    # print(np.shape(hist_all))
    return hist_all

def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    # ----- TODO -----

    # opts, img_path, dictionary = args

    img = Image.open(join(opts.data_dir, img_path))
    img = np.array(img).astype(np.float32)/255

    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    return feature


def build_recognition_system(opts, n_worker):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----

    N = len(train_files)
    K = dictionary.shape[0]

    args = [(opts, img_path, dictionary) for img_path in train_files]
    pool = multiprocessing.Pool(processes=n_worker)
    results = pool.starmap(get_image_feature, args)

    features = np.array(results)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    pass

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    intersections = np.minimum(word_hist, histograms)
    similarity_scores = np.sum(intersections, axis = 1) # summing all the elementes for each histogram
    distances = 1 - similarity_scores

    # print(distances.len())

    return distances
        
    
def evaluate_recognition_system(opts, n_worker):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----

    train_labels = trained_system['labels']
    train_features = trained_system['features']
    
    N = len(test_files)
    confusion = np.zeros(shape=(8,8), dtype=float)

    args = [(test_opts, img_path, dictionary) for img_path in test_files]
    pool = multiprocessing.Pool(processes=n_worker)
    test_features = pool.starmap(get_image_feature, args)

    for test_feature, img_spm in enumerate(test_features):
        distance = distance_to_set(img_spm, train_features)
        closest_id = np.argmin(distance)
        confusion[test_labels[test_feature], train_labels[closest_id]] +=1
    
    accuracy = np.trace(confusion)/np.sum(confusion)

    return confusion, accuracy
    