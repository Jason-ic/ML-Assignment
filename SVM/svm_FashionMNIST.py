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

# 将加入的数据集转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
])

# 下载FashionMNIST数据集
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 加载FashionMNIST数据集
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

train_images, train_targets = next(iter(train_loader))
test_images, test_targets = next(iter(test_loader))

train_images = train_images.numpy().squeeze() # train_images.shape: (60000, 28, 28)
test_images = test_images.numpy().squeeze() # test_images.shape: (10000, 28, 28)

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

feature_type = 'hog'

if feature_type != 'pixel' :
    print(feature_type + ' feature: ')
    train_images = extract_feature(train_images, feature_type)
    test_images = extract_feature(test_images, feature_type)

train_images = np.reshape(train_images, (train_images.shape[0], -1))  
test_images = np.reshape(test_images, (test_images.shape[0], -1))  


print(train_images.shape, test_images.shape) # 输出特征图维度

Class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# 选择SVM对应的核函数
clf = SVC(kernel=kernels[2], verbose=True)
clf.fit(train_images, train_targets)
predictions = clf.predict(test_images)
accuracy = accuracy_score(test_targets, predictions)
print(f'Accuracy : {accuracy * 100:.2f}%')

SVM_report = classification_report(test_targets, predictions, target_names = Class_names)
print("Classification report for Support Vector Machine:\n", SVM_report)

conf_mat2 = confusion_matrix(test_targets, predictions)

plt.figure(figsize = (12,8))
sns.heatmap(conf_mat2, annot = True, cmap = 'Blues',fmt = 'd', xticklabels = Class_names, yticklabels = Class_names)
plt.title('Confusion Matrix for Support Vector Machine')
plt.xlabel('Predicted Class', fontsize = 10)
plt.ylabel('True Class', fontsize = 10)
plt.show()
