import sys
import pickle
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import SVC


# Spatial binning
def bin_spatial(img, size=(32,32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


# Color histogram
def color_hist(img, nbins=32):
    ch1_hist = np.histogram(img[:,:,0], bins=nbins, range=(0,256))
    ch2_hist = np.histogram(img[:,:,1], bins=nbins, range=(0,256))
    ch3_hist = np.histogram(img[:,:,2], bins=nbins, range=(0,256))
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
    return hist_features


# HOG features
def get_hog_features(img, feature_vec=True, orient=11, 
                     pix_per_cell=14, cell_per_block=2):   
    features = hog(img, orientations=orient, 
                   pixels_per_cell=(pix_per_cell, pix_per_cell), 
                   cells_per_block=(cell_per_block, cell_per_block), 
                   block_norm='L2-Hys', transform_sqrt=True, 
                   visualize=False, feature_vector=feature_vec)
    return features


def extract_features(imgs):
    features = []

    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb) 
        
        # Spatial binning
        spatial_features = bin_spatial(feature_image)
        file_features.append(spatial_features)
        
        # Color histograms
        hist_features = color_hist(feature_image)
        file_features.append(hist_features)
        
        # Hog features for all channels
        hog_features = []
        
        for ch in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,ch]))
            
        hog_features = np.ravel(hog_features)        
        file_features.append(hog_features)

        features.append(np.concatenate(file_features))
        
    return features


if __name__ == "__main__":
    cars = glob.glob('vehicles/*/*.png')
    not_cars = glob.glob('non-vehicles/*/*.png')
    car_features = extract_features(cars)
    not_car_features = extract_features(not_cars)

    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=rand_state,
                                                        shuffle=True)
    # Normalize
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    # Train classifier
    clf = SVC(C=100, verbose=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    joblib.dump(clf, 'clf.joblib')
    print('SVC saved')

    with open('scaler.pkl', 'wb') as file:
        pickle.dump(X_scaler, file)