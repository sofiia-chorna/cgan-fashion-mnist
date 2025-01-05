import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim=100, n_classes=10, img_dim=28):
        super().__init__()

        # params
        self.z_dim = z_dim  # noise vector size
        self.n_classes = n_classes
        self.img_dim = img_dim  # output image size

        # fully connected layers for noise and labels
        self.fc_noise = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.ReLU(),
            nn.Dropout(0)
        )
        self.fc_label = nn.Sequential(
            nn.Linear(n_classes, 1000),
            nn.ReLU(),
            nn.Dropout(0)
        )

        # combined layers
        self.fc_combined = nn.Sequential(
            nn.Linear(200 + 1000, 1200),
            nn.ReLU(),
            nn.Dropout(0)
        )

        # output layer to map to image dims
        self.fc_output = nn.Sequential(
            nn.Linear(1200, img_dim * img_dim),
            nn.Sigmoid()  # output in [0, 1]
        )

    def forward(self, noise, labels):
        # one-hot encode labels
        labels_one_hot = F.one_hot(labels, num_classes=self.n_classes).float()

        # process noise and embedded labels separately
        noise_vector = self.fc_noise(noise)
        label_vector = self.fc_label(labels_one_hot)

        # pass through combined layer
        combined_vector = self.fc_combined(torch.cat([noise_vector, label_vector], dim=1))

        # generate image
        img = self.fc_output(combined_vector)
        img = img.view(-1, 1, self.img_dim, self.img_dim)  # reshape to (batch_size, 1, 28, 28)

        return img
