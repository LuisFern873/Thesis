import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})  # Tamaño de fuente por defecto

path = './experiments'

files = [
    path + '/cifar10_deit.csv',
    # path + '/cifar10_mamba.csv',
    path + '/cifar10_convnext.csv',
    path + '/cifar10_efficient.csv',
]

colors = ['blue', 'olive', 'purple']  # Colores por modelo
labels = ['DeiT-tiny', 'ConvNeXt-tiny', 'EfficientNetB0']  # Etiquetas

plt.figure(figsize=(10, 6))
shadow_margin = 0.5  # Margen para sombra decorativa

for i, file in enumerate(files):
    df = pd.read_csv(file)
    steps = df['Step']
    values = df['Value']
    
    # Línea principal
    plt.plot(steps, values, label=labels[i], color=colors[i])
    
    # Sombra alrededor de la línea
    plt.fill_between(
        steps,
        values - shadow_margin,
        values + shadow_margin,
        color=colors[i],
        alpha=0.15
    )

# Título más específico
plt.title('Accuracy of Pretrained Vision Models on CIFAR-10 dataset in Federated Learning (10 Clients)')

plt.xlabel('Communication round')
plt.xticks(range(1, 21))
plt.ylabel('Testing accuracy (%)')

plt.legend(loc='lower right')  # Leyenda
plt.grid(True)
plt.tight_layout()
plt.show()
