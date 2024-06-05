import matplotlib.pyplot as plt

data = {
    'VGG16 lr=0.001 dropout=0.0': {'Acc List': [87.27, 89.2, 90.17, 91.33, 91.72, 91.73, 91.77, 92.26, 92.6, 92.08, 92.39, 92.28, 92.0, 92.09, 92.54, 92.61, 92.14, 92.41, 92.43, 92.01], 'Epoch List': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    'VGG16 lr=0.001 dropout=0.5': {'Acc List': [86.58, 90.18, 90.26, 90.48, 91.18, 91.6, 92.26, 92.66, 92.17, 92.25, 92.51, 92.22, 92.49, 92.98, 92.47, 92.3, 92.3, 92.32, 92.33, 92.3], 'Epoch List': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    'VGG16 lr=0.01 dropout=0.0': {'Acc List': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 'Epoch List': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    'VGG16 lr=0.01 dropout=0.5': {'Acc List': [10.03, 10.05, 10.07, 10.06, 9.96, 10.28, 10.43, 10.13, 9.82, 10.05, 10.16, 9.7, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 'Epoch List': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    'CNN lr=0.001 dropout=0.0': {'Acc List': [80.65, 85.49, 85.75, 87.56, 87.34, 87.83, 88.84, 87.96, 89.45, 88.93, 89.05, 88.65, 89.25, 88.5, 88.54, 89.47, 89.56, 89.22, 88.91, 89.42], 'Epoch List': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    'CNN lr=0.001 dropout=0.5': {'Acc List': [75.08, 79.75, 82.92, 83.95, 84.84, 85.97, 84.16, 84.98, 86.44, 85.87, 86.36, 86.69, 86.17, 86.6, 85.89, 86.53, 87.21, 87.14, 87.1, 86.69], 'Epoch List': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    'CNN lr=0.01 dropout=0.0': {'Acc List': [81.1, 81.36, 81.03, 81.4, 82.1, 79.34, 72.57, 76.71, 78.75, 77.91, 53.05, 74.73, 76.13, 76.47, 77.73, 71.65, 72.93, 10.0, 10.0, 10.0], 'Epoch List': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
    'CNN lr=0.01 dropout=0.5': {'Acc List': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 'Epoch List': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
}

def compare_datasets(*model_identifiers):

    metrics = [data[dataset_name] for dataset_name in model_identifiers if dataset_name in data]

    for i, metric in enumerate(metrics, 1):
        plt.plot(metric['Epoch List'], metric['Acc List'], label=model_identifiers[i-1])

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Models')
    plt.legend()
    plt.grid(True)

    filename = f'comparison {model_identifiers}.svg'
    plt.savefig(filename)
    plt.show()

compare_datasets('VGG16 lr=0.01 dropout=0.0', 'VGG16 lr=0.01 dropout=0.5', 'VGG16 lr=0.001 dropout=0.5', 'VGG16 lr=0.001 dropout=0.0')
compare_datasets('CNN lr=0.01 dropout=0.0', 'CNN lr=0.01 dropout=0.5', 'CNN lr=0.001 dropout=0.5', 'CNN lr=0.001 dropout=0.0')
compare_datasets('CNN lr=0.001 dropout=0.0', 'VGG16 lr=0.001 dropout=0.0')
