import pandas as pd
import matplotlib.pyplot as plt

path = './experiments'

files = [
    path + '/2025-06-04-15-45-26.csv',             # valores reales: DeiT-Tiny
    path + '/efficientnetb0_iid_prediction.csv',
    path + '/convnext_iid_prediction.csv',
    path + '/visionmamba_iid_prediction.csv'
]

colors = ['blue', 'green', 'orange', 'red']
labels = ['DeiT-Tiny', 'EfficientNetB0', 'ConvNeXt-Tiny', 'Vision Mamba']

plt.figure(figsize=(10, 6))

for i, file in enumerate(files):
    df = pd.read_csv(file)
    plt.plot(df['Step'], df['Value'], label=labels[i], color=colors[i])

# Título más específico
plt.title('Accuracy of Pretrained Vision Models in Federated Learning with IID Brain MRI Data (4 Clients)')

plt.xlabel('Round')
plt.xticks(range(1, 21))
plt.ylabel('Accuracy')
plt.legend(loc='lower right')  # leyenda en la esquina inferior derecha
plt.grid(True)
plt.tight_layout()
plt.show()
