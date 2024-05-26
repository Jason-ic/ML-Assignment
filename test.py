import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import  confusion_matrix
import numpy as np
from scipy.fftpack import fft2, fftshift


# Define transformations: convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
])

# Load the FashionMNIST training dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Convert the datasets to DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

train_images, train_targets = next(iter(train_loader))
test_images, test_targets = next(iter(test_loader))

train_images = train_images.numpy().squeeze() # train_images.shape: (60000, 28, 28)
test_images = test_images.numpy().squeeze() # test_images.shape: (10000, 28, 28)

def lpq(image, win_size=3):
    # Create a 2D Hanning window
    hanning_window = np.outer(np.hanning(win_size), np.hanning(win_size))
    
    # Fourier transform of the 2D Hanning window
    f = fft2(hanning_window)
    
    # Create frequency response filters
    freqresp = np.zeros((win_size, win_size, 4), dtype=complex)
    freqresp[:, :, 0] = f
    freqresp[:, :, 1] = f.T
    freqresp[:, :, 2] = f * np.exp(-2j * np.pi / win_size)
    freqresp[:, :, 3] = f.T * np.exp(-2j * np.pi / win_size)

    # Convolution of the image with the filters
    conv_real = np.zeros((image.shape[0], image.shape[1], 4))
    for i in range(4):
        conv_real[:, :, i] = np.abs(cv2.filter2D(image, cv2.CV_64F, np.real(freqresp[:, :, i]), borderType=cv2.BORDER_REFLECT))
    
    # Trim the result to match the expected output size
    trim_size = win_size - 1
    conv_real = conv_real[trim_size // 2: -trim_size // 2, trim_size // 2: -trim_size // 2, :]
    
    # LPQ descriptor calculation
    LPQdesc = (conv_real > 0).astype(np.uint8)
    LPQdesc = LPQdesc[:, :, 0] + (LPQdesc[:, :, 1] << 1) + (LPQdesc[:, :, 2] << 2) + (LPQdesc[:, :, 3] << 3)
    
    # Compute histogram
    hist, _ = np.histogram(LPQdesc.flatten(), bins=256, range=(0, 255))
    
    return hist

def extract_feature(images, feature_type):
    if feature_type == 'hog':
        hog_features = []
        for image in images:
            feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
            hog_features.append(feature)
        return np.array(hog_features)
    if feature_type == 'lbp':
        lbp_features = []
        for image in images:
            image = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            feature = local_binary_pattern(image, P=8, R=2, method='uniform')
            lbp_features.append(feature)
        return np.array(lbp_features)
    if feature_type == 'sift':
        sift = cv2.SIFT_create()
        sift_features = []
        for image in images:
            # image = image.astype(np.uint8)
            image = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            # print(image)
            keypoints, descriptors = sift.detectAndCompute(image, None)
            if descriptors is None:
                descriptors = np.zeros((1, sift.descriptorSize()), dtype=object)
            # print(descriptors)
            sift_features.append(descriptors)

            # # Plot the original image
            # plt.figure(figsize=(8, 4))
            # plt.subplot(1, 2, 1)
            # plt.title('Original Image')
            # plt.imshow(image, cmap='gray')
            # plt.axis('off')
            # # Plot the LBP feature image
            # plt.subplot(1, 2, 2)
            # plt.title('sift Feature Image')
            # plt.imshow(descriptors, cmap='gray')
            # plt.axis('off')
            # plt.show()

            # break
         # Filter out None values
        sift_features = [sublist for sublist in sift_features if sublist is not None]

        if not sift_features:
            return np.array([])  # Return an empty array if there are no valid features

        # Determine the number of columns (assuming all feature vectors have the same number of columns)
        num_columns = sift_features[0].shape[1]

        # Determine the maximum number of rows among the sublists
        max_length = max(len(sublist) for sublist in sift_features)

        # Pad sublists with rows of zeros to ensure uniform length
        padded_sift_features = [np.vstack((sublist, np.zeros((max_length - len(sublist), num_columns)))) for sublist in sift_features]

        # Create a NumPy array from the padded lists
        sift_features_array = np.array(padded_sift_features)

        return np.array(sift_features, dtype=object)
    
    if feature_type == 'lpq':
        lpq_features = []
        for image in images:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            feature = lpq(image)
            lpq_features.append(feature)
        return np.array(lpq_features)

feature_type = 'hog'

if feature_type != 'pixel' :
    print(feature_type + ' feature: ')
    train_images = extract_feature(train_images, feature_type)
    test_images = extract_feature(test_images, feature_type)

train_images = np.reshape(train_images, (train_images.shape[0], -1))  
test_images = np.reshape(test_images, (test_images.shape[0], -1))  


print(train_images.shape, test_images.shape) # output feature demension

# Class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# # choose a kernel function
# clf = SVC(kernel=kernels[2], verbose=True)
# clf.fit(train_images, train_targets)
# predictions = clf.predict(test_images)
# accuracy = accuracy_score(test_targets, predictions)
# print(f'Accuracy : {accuracy * 100:.2f}%')

# SVM_report = classification_report(test_targets, predictions, target_names = Class_names)
# print("Classification report for Support Vector Machine:\n", SVM_report)

# conf_mat2 = confusion_matrix(test_targets, predictions)

# plt.figure(figsize = (12,8))
# sns.heatmap(conf_mat2, annot = True, cmap = 'Blues',fmt = 'd', xticklabels = Class_names, yticklabels = Class_names)
# plt.title('Confusion Matrix for Support Vector Machine')
# plt.xlabel('Predicted Class', fontsize = 10)
# plt.ylabel('True Class', fontsize = 10)
# plt.show()