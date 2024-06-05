import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import time
import os

from torch.utils.data import DataLoader

# Hyperparameters
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
DROPOUT = float(os.getenv("DROPOUT", 0))

VGG16_PATH = f'./weight/FashionMNIST_vgg16_{LEARNING_RATE}_{DROPOUT}.pth'

print(f'LEARNING_RATE={LEARNING_RATE}, DROPOUT={DROPOUT}')

def append_to_file(file_path, string_to_append):
    with open(file_path, "a") as file:
        file.write(string_to_append)

def main():
    device = torch.device("cuda")
    vgg16 = models.vgg16(dropout=DROPOUT)
    vgg16.to(device)

    trainset_fashion = torchvision.datasets.FashionMNIST(
        root='./data/pytorch/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    testset_fashion = torchvision.datasets.FashionMNIST(
        root='./data/pytorch/FashionMNIST',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    trainloader_fashion = DataLoader(trainset_fashion, batch_size=4,
                                                      shuffle=True, num_workers=2)
    testloader_fashion = DataLoader(testset_fashion, batch_size=4,
                                                     shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.parameters(), lr=LEARNING_RATE, momentum=0.9)

    accuracy_list = []
    epoch_list = []
    start_time = time.time()
    print("Start Training >>>")
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader_fashion, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.repeat(1, 3, 2, 2)
            optimizer.zero_grad()
            outputs = vgg16(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 2000}')
                running_loss = 0.0

        start_test = time.time()
        print("\nStart Testing >>>")
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader_fashion, 0):
                images, labels = data[0].to(device), data[1].to(device)
                images = images.repeat(1, 3, 2, 2)
                outputs = vgg16(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if i % 2000 == 1999:
                    print(f'Testing Batch: {i + 1}')
        test_time = (time.time() - start_test) / 60
        print('>>> Finished Testing')
        print(f'Testing time: {test_time} mins.')
        print(f'Accuracy: {100 * correct / total}')
        accuracy_list.append(100 * correct / total)
        epoch_list.append(epoch + 1)

    append_to_file("./result.txt", f'VGG16_{LEARNING_RATE}_{DROPOUT}: Acc List: {accuracy_list}\tEpoch List: {epoch_list}')
    train_time = (time.time() - start_time) / 60
    torch.save(vgg16.state_dict(), VGG16_PATH)
    print('>>> Finished Training')
    print(f'Training time: {train_time} mins.')

    plt.plot(epoch_list, accuracy_list, 'b--', label='Custom CNN Accuracy')
    plt.title('Custom CNN Accuracy vs epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    axes = plt.gca()
    axes.set_ylim(0, 100)
    plt.legend()
    plt.savefig(f'./visualization/CustomCNNvsEpoch_{LEARNING_RATE}_{DROPOUT}.png')
    plt.show()


if __name__ == '__main__':
    main()
