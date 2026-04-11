# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator

# log_dir = "./out/fedavg/tumor/2025-06-04-18-27-21/events.out.tfevents.1749079643.LAPTOP-HVF502VA.19452.0"
# ea = event_accumulator.EventAccumulator(log_dir)
# ea.Reload()

# # Nombre de la métrica que sí está en los TENSORS
# tensor_name = 'Accuracy-tumor-20clients-0%IID-4classes-seed42/testset-CentralizedEvaluation'
# tensors = ea.Tensors(tensor_name)

# # Extraer steps y valores flotantes
# steps = [e.step for e in tensors]
# values = [e.tensor_proto.float_val[0] for e in tensors]  # Porque es un tensor escalar

# # Visualización
# plt.figure(figsize=(10, 5))
# plt.plot(steps, values)
# plt.xlabel("Rounds")
# plt.ylabel("Accuracy")
# plt.ylim((0, 100))
# plt.title("Accuracy over rounds")
# plt.grid(True)
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})  # Tamaño de fuente por defecto

# Leer CSV
df = pd.read_csv("data/tumor/exp1/2025-06-04-18-27-21.csv")
df.columns = df.columns.str.strip()

# Datos
x = df['Step']
y = df['Value']

# Crear sombra (±1.5 puntos artificiales como ejemplo)
shadow_range = 1.5
y_lower = y - shadow_range
y_upper = y + shadow_range

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Accuracy', color='blue')
plt.fill_between(x, y_lower, y_upper, color='blue', alpha=0.2)  # Sin label

plt.xlabel("Communication Round")
plt.ylabel("Testing Accuracy (%)")
plt.ylim((0, 100))
plt.title("Accuracy of non-pretrained DeiT (4 clients)")
plt.grid(True)
# plt.legend()
plt.tight_layout()
plt.show()




