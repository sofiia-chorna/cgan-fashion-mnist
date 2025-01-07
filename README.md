# WIP
## Task
You will work with the Fashion MNIST dataset, aiming to implement a Conditional Generative Adversarial Network (Conditional GAN) to generate images from this dataset. Start by splitting the dataset into training and test sets. While maintaining simplicity, focus on devising an innovative approach to analyze the test dataset using the Conditional GAN model. Employ data science techniques to explore and gain valuable insights from the generated data.

## Implementation
The architecture and implementation follow the CGAN paper [1], including the choice of optimizers, schedulers, hyperparameters etc. I still experimented with the values, and ended up with a mode collapse 🤪, so I remained the publication values. As mentioned in the paper, we use the Maxout activation [2] for the discriminator.

For logging, I used Weight & Biases, it does suprisingly great plots with all possible statistics (see below).

## Feedback
The results aren't as promising as I had hoped 😕 : quality of the generated images is currently not that high. Upon further reflection, I suppose that using convolutional layers instead of fully connected layers would have likely improved performance significantly... Anyways, it was still a nice task for me 😇

## References
[1] Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. https://arxiv.org/abs/1411.1784

[2] Goodfellow, I. J., et al. (2013). Maxout Networks. https://arxiv.org/abs/1302.4389
