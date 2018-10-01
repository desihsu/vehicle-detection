import sys
import pickle
import cv2
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
from train_svc import get_hog_features, color_hist, bin_spatial


svc = joblib.load('clf.joblib')
with open('scaler.pkl', 'rb') as file:
    X_scaler = pickle.load(file)
    

def find_cars(img, ystart, ystop, scale, 
              orient=11, pix_per_cell=14, cell_per_block=2): 
    img = img.astype(np.float32) / 255
    img = img[ystart:ystop,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img = cv2.resize(img, (np.int(img.shape[1]/scale), 
                           np.int(img.shape[0]/scale)))
        
    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for entire image
    hog1 = get_hog_features(ch1, feature_vec=False)
    hog2 = get_hog_features(ch2, feature_vec=False)
    hog3 = get_hog_features(ch3, feature_vec=False)
    
    boxes = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, 
                             xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, 
                             xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, 
                             xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img[ytop:ytop+window, xleft:xleft+window], 
                                (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg)
            hist_features = color_hist(subimg)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(
                                            (spatial_features, hist_features, 
                                             hog_features)).reshape(1,-1)) 
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return boxes


def add_heat(heatmap, boxes):
    for box in boxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
    return heatmap

    
def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_boxes(img, labels):
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        box = ((np.min(nonzerox), np.min(nonzeroy)), 
               (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, box[0], box[1], (0,0,255), 6)
        
    return img


def process_image(img):
    ystarts = [388, 400, 416, 400, 416, 432, 400, 432, 400, 464]
    ystops = [452, 464, 480, 496, 512, 528, 528, 560, 596, 660]
    scales = [1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 3.5, 3.5]
    boxes = []
    
    # Get all detected boxes from all searches
    for ystart, ystop, scale in zip(ystarts, ystops, scales):
        boxes += find_cars(img, ystart, ystop, scale)
    
    # Create heatmap
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, boxes)
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    
    # Identify and draw blobs from heatmap
    labels = label(heatmap)
    draw_img = draw_boxes(np.copy(img), labels)
    return draw_img


if __name__ == "__main__":
    if (sys.argv[1].split(".")[-1] == "mp4"):
        clip = VideoFileClip(sys.argv[1])
        output = clip.fl_image(process_image)
        output.write_videofile("output.mp4", audio=False)
    else:
        img = mpimg.imread(sys.argv[1])
        img = process_image(img)
        mpimg.imsave("output.jpg", img)