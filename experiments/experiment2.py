import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


log_dir = "./out/fedavg/tumor/2025-06-08-17-21-14/events.out.tfevents.1749421276.LAPTOP-HVF502VA.14684.0"

ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

print(ea.Tags())

print(ea.Tags()['scalars']) # ['accuracy']

print(ea.Tags()['scalars'])


# Accuracy-tumor-4clients-0%IID-Dir(0.5)-seed42/testset-afterLocalTraining


scalars = ea.Scalars('Accuracy-tumor-4clients-0%IID-Dir(0.5)-seed42/testset-afterLocalTraining')

steps = [s.step for s in scalars]
values = [s.value for s in scalars]

plt.figure(figsize=(10, 5))
plt.plot(steps, values) # label='Accuracy'
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.ylim((0,100))
plt.title("Accuracy over rounds")
# plt.legend()
plt.grid(True)
plt.show()
