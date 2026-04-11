import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})  # Tamaño de fuente por defecto

path = './data/tumor/exp2/'

files = [
    path + '2025-06-08-17-21-14_Accuracy-tumor-4clients-0%IID-Dir(0.5)-seed42_testset-afterLocalTraining_clients_client0.csv',
    path + '2025-06-08-17-21-14_Accuracy-tumor-4clients-0%IID-Dir(0.5)-seed42_testset-afterLocalTraining_clients_client1.csv',
    path + '2025-06-08-17-21-14_Accuracy-tumor-4clients-0%IID-Dir(0.5)-seed42_testset-afterLocalTraining_clients_client2.csv',
    path + '2025-06-08-17-21-14_Accuracy-tumor-4clients-0%IID-Dir(0.5)-seed42_testset-afterLocalTraining_clients_client3.csv'
]

colors = ['blue', 'green', 'red', 'purple']
labels = ['Client 0', 'Client 1', 'Client 2', 'Client 3']

plt.figure(figsize=(10, 6))
shadow_margin = 1.5  # Ajusta este valor para aumentar/disminuir la sombra

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

plt.title('Testing accuracy for each client, DeiT non-pretrained')
plt.xlabel('Communication round')
plt.ylabel('Testing accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
