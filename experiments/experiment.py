# Modifica el codigo para 

# Plot multiple graphs in one plot using Tensorboard
# https://stackoverflow.com/questions/48951136/plot-multiple-graphs-in-one-plot-using-tensorboard

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

clients = 6

for i in range(100):
    writer.add_scalars(
        'accuracy', 
        {f'clients/client{client}': client / (i + 1) for client in range(clients)},
        i
    )

writer.close()



"""
python experiment.py
tensorboard --logdir=runs
"""

"""
- 
- 4 clients

First experiment
- No data augmentation
- No pretrained model

- Label and quantity skew ()

"""