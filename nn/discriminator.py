import torch
import torch.nn as nn
import torch.nn.functional as F
from maxout import Maxout


class Discriminator(nn.Module):
    def __init__(self, n_classes=10, img_dim=28):
        super().__init__()

        # params
        self.n_classes = n_classes
        self.img_dim = img_dim

        # layers for images and labels
        self.maxout_img = Maxout(d_in=img_dim * img_dim, num_piece=5, num_units=240)
        self.maxout_label = Maxout(d_in=n_classes, num_piece=5, num_units=50)

        # combined layer
        self.maxout_combined = Maxout(d_in=240 + 50, num_piece=4, num_units=240)

        # output layer
        self.linear_output = nn.Linear(240, 1)
        self.dropout = nn.Dropout(0)

        # output in binary value
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, labels):
        # pass image
        x_img = img.view(-1, self.img_dim * self.img_dim)  # flatten
        x_img = self.maxout_img(x_img)
        x_img = self.dropout(x_img)

        # pass labels
        labels_one_hot = F.one_hot(labels, num_classes=self.n_classes).float()
        x_label = self.maxout_label(labels_one_hot)
        x_label = self.dropout(x_label)

        # combined layer
        combined = torch.cat((x_img, x_label), dim=1)
        combined = self.maxout_combined(combined)
        combined = self.dropout(combined)

        # result: fake or not
        output = self.sigmoid(self.linear_output(combined))

        return output
