import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})  # Tamaño de fuente por defecto

path = './experiments'

files = [
    path + '/deit_domain.csv',       
    path + '/mamba_domain.csv',
    path + '/convnext_domain.csv',
    path + '/efficient_domain.csv',
]


colors = ['blue', 'green', 'olive', 'purple']
labels = ['DeiT-tiny', 'Vim-tiny', 'ConvNeXt-tiny', 'EfficientNetB0']

plt.figure(figsize=(10, 6))
shadow_margin = 1.0  # Puedes ajustar este valor si quieres más o menos sombra

for i, file in enumerate(files):
    df = pd.read_csv(file)
    steps = df['Step']
    values = df['Value']

    plt.plot(steps, values, label=labels[i], color=colors[i])
    plt.fill_between(
        steps,
        values - shadow_margin,
        values + shadow_margin,
        color=colors[i],
        alpha=0.15
    )

plt.title('Accuracy of Pretrained Vision Models on DomainNet dataset in Federated Learning (6 Clients)')
plt.xlabel('Communication Round')
plt.xticks(range(1, 21))
plt.ylabel('Testing accuracy (%)')
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
