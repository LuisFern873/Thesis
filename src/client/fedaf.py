import torch
from src.client.fedavg import FedAvgClient

"""
    The overall mechanism of our proposed FedAF is illustrated in Figure 2. In each learning round, clients first engage 
    in collaborative data condensation, updating their local condensed data. They then share this condensed data and soft 
    labels with the server. Subsequently, the server uses this information to update the global model.
"""

class FedAFClient(FedAvgClient):
    def __init__(self, **commons):
        super(FedAFClient, self).__init__(**commons)
        """
            Initialize condensed dataset:
            The client first initializes each class of condensed data by sampling from the local original data D_k or Gaussian noise.
            We initialize each class of condensed data using the average of randomly sampled local original data.
        """
        num_classes = self.args.dataset.class_num
        ipc = self.args.fedaf.ipc  # image-per-class
        channel, height, width = self.dataset[0][0].shape

        self.image_syn = torch.randn(
            size=(num_classes * ipc, channel, height, width), 
            dtype=torch.float, 
            requires_grad=True, 
            device=self.device
        )

        self.label_syn = torch.tensor(
            [i for i in range(num_classes) for _ in range(ipc)], 
            dtype=torch.long, 
            device=self.device
        )

    # class-wise mean logits
    def compute_class_means(self, features, labels):
        class_means = []
        for c in range(self.args.dataset.class_num):
            idx = labels == c
            if idx.sum() > 0:
                mean_feat = features[idx].mean(dim=0)
            else:
                mean_feat = torch.zeros_like(features[0])
            class_means.append(mean_feat)
        return torch.stack(class_means)


    """
        Package condensed data.
        Soft Labels: local class-wise soft labels about its original data
    """
    def package(self):
        with torch.no_grad():
            features = self.model.base(self.image_syn)
            logits = self.model.classifier(features)
            soft_labels = torch.softmax(logits, dim=1)
        
        client_package = dict(
            image_syn=self.image_syn.detach().cpu(),
            label_syn=self.label_syn.detach().cpu(),
            soft_labels=soft_labels.detach().cpu(),
            model_params=self.model.state_dict()
        )
        return client_package
    
    """
        Data condensation with distribution matching.
        client k is tasked with learning a set of local condensed data denoted as S_k.
    """
    def fit(self):

        self.model.train()
        self.dataset.train()

        for _ in range(self.local_epoch):

            for x, y in self.trainloader: # 

                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                syn_x, syn_y = x.to(self.device), y.to(self.device) # replace with condensed data.

                real_features = self.model.base(x)
                syn_features = self.model.base(syn_x)

                # Distribution Matching (DM) loss
                mu_real = self.compute_class_means(real_features, y)
                mu_syn  = self.compute_class_means(syn_features, syn_y)
                loss_DM = torch.nn.functional.mse_loss(mu_syn, mu_real)

                # Collaborative Data Condensation (CDC) loss
                real_logits = self.model.classifier(real_features)
                syn_logits = self.model.classifier(syn_features)
                loss_CDC = torch.nn.functional.mse_loss(syn_logits, real_logits)

                loss = loss_DM + self.args.fedaf.lambda_loc * loss_CDC


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        

    