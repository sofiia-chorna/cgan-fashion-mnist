# WIP
## Task
You will work with the Fashion MNIST dataset, aiming to implement a Conditional Generative Adversarial Network (Conditional GAN) to generate images from this dataset. Start by splitting the dataset into training and test sets. While maintaining simplicity, focus on devising an innovative approach to analyze the test dataset using the Conditional GAN model. Employ data science techniques to explore and gain valuable insights from the generated data.

## Implementation
### Architecture
The architecture and implementation wholy follow the CGAN paper [1], including the choice of optimizers, schedulers, hyperparameters etc. As mentioned in the paper, the Maxout activation [2] is used for the discriminator.

For logging, I tried to use [Weight & Biases](https://wandb.ai/site/), and indeed it does suprisingly great plots with all possible statistics.

#### Analysis
To analyse the generated data sets, I used dimentionality reduction analysis with the classical PCA and FID metric.  

### Results with the architecture proposed in [1] 
TODO
- generation examples
- loss plot
- PCA
- FID

### Further experiments
Trying to improve the quality of the generated images, I used some of the proposed techniques from the [3] paper.
- Normalisation of the inputs: (normalise images between -1 and 1 and used of tanh as the last layer of the generator output)
- Use of gaussian noise instead of one with uniform distribution
- One-sided label smoothing (replace the 0 and 1 targets for a classifier with smoothed values, like .9 or .1)

### Results with the some techniques improvements proposed in [3]
TODO
- generation examples
- loss plot
- PCA
- FID

## Feedback
The results aren't as promising as I had hoped 😕 : quality of the generated images is currently not that high. Upon further reflection, I suppose that using convolutional layers instead of fully connected layers would have likely improved performance significantly... Anyways, it was still a nice task for me 😇

## References
[1] Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. https://arxiv.org/abs/1411.1784

[2] Goodfellow, I. J., et al. (2013). Maxout Networks. https://arxiv.org/abs/1302.4389

[3] Salimans, T., et al. (2016). Improved techniques for training GANs. https://arxiv.org/abs/1606.03498
