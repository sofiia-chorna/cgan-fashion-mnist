import os
import torch
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
from utils import DEVICE


def get_feature_extractor():
    """Use pretrained inception_v3 model for getting features"""
    pretrained_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(DEVICE)
    pretrained_model.eval()

    return pretrained_model


def get_pca(features, labels, output_dir, prefix):
    """Apply PCA on the samples"""

    # run dim reduction with pca
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(features)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels.cpu().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f"PCA of {prefix} samples")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    pca_plot_path = os.path.join(output_dir, f"{prefix}_pca_plot.png")
    plt.savefig(pca_plot_path)
    plt.close()

    wandb.log({"PCA Plot": wandb.Image(pca_plot_path)})
    print(f"PCA plot saved at {pca_plot_path}")


def get_tsne(features, labels, output_dir, prefix):
    """Apply t-SNE on the samples"""

    # Run dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_reduced = tsne.fit_transform(features)

    # Plot the t-SNE results
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels.cpu().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f"t-SNE of {prefix} samples")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    tsne_plot_path = os.path.join(output_dir, f"{prefix}_tsne_plot.png")
    plt.savefig(tsne_plot_path)
    plt.close()

    # Log the plot to WandB
    wandb.log({"t-SNE Plot": wandb.Image(tsne_plot_path)})
    print(f"t-SNE plot saved at {tsne_plot_path}")


def get_embeddings(feature_extractor, samples):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # required for inception_v3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalize
    ])

    # transform the entire batch
    samples = samples.to(DEVICE)
    feature_extractor = feature_extractor.to(DEVICE)
    transformed_samples = torch.stack([
        transform(transforms.ToPILImage()(img).convert("RGB")) for img in samples
    ]).to(DEVICE)

    with torch.no_grad():
        features = feature_extractor(transformed_samples)

    # flatten to (num_samples, num_features)
    embeddings = features.view(features.size(0), -1).cpu().detach().numpy()

    return embeddings


def run_dim_reduction(feature_extractor, samples, labels, output_dir, real):
    # get features for real and generated data
    features = get_embeddings(feature_extractor, samples)

    # run pca and tsne on both real and generated embeddings
    prefix = 'real' if real else 'generated'
    get_pca(features, labels, output_dir, prefix)
    get_tsne(features, labels, output_dir, prefix)
