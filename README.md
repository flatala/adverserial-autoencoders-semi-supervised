# Adverserial Autoencoders (Semi-Supervised)

A PyTorch implementation of a **semi-supervised Adversarial Autoencoder (AAE)**, with example experiments on MNIST.  
The AAE learns a low-dimensional latent representation split into  
- a **categorical** part for class labels, and  
- a **continuous** part for “style.”  
---

## 📦 Requirements

Install with:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `torch >= 2.6.0`  
- `torchvision >= 0.21.0`  
- `numpy >= 2.2.5`  
- `pandas >= 2.2.3`  
- `matplotlib >= 3.10.1`  
- `tensorboard >= 2.19.0`  
- `scikit-learn == 1.6.1`  
- `notebook >= 7.4.1, < 8.0.0`  
- `jupyter >= 1.1.1, < 2.0.0`  
- `ipykernel >= 6.29.5, < 7.0.0`  
- `tqdm`

---

## 🚀 Installation

```bash
git clone https://github.com/flatala/adverserial-autoencoders-semi-supervised.git
cd adverserial-autoencoders-semi-supervised
pip install -r requirements.txt
```

---

## 📝 Usage

### 1. Define model options

```python
from aae import SemiSupervisedAutoEncoderOptions, SemiSupervisedAdversarialAutoencoder

opts = SemiSupervisedAutoEncoderOptions(
    input_dim=784,
    ae_hidden_dim=1024,
    disc_hidden_dim=512,
    latent_dim_categorical=10,    # 10 classes
    latent_dim_style=16,          # style vector size
    recon_loss_fn=torch.nn.MSELoss(),
    init_recon_lr=1e-3,
    semi_supervised_loss_fn=torch.nn.CrossEntropyLoss(),
    init_semi_sup_lr=1e-3,
    init_gen_lr=1e-4,
    init_disc_categorical_lr=1e-4,
    init_disc_style_lr=1e-4,
    use_decoder_sigmoid=True
)
model = SemiSupervisedAdversarialAutoencoder(opts)
```

### 2. Prepare data loaders

Use standard PyTorch `DataLoader`s for MNIST (or your own dataset), splitting into labeled and unlabeled sets.

### 3. Train

```python
model.train_mbgd(
    train_labeled_loader,
    val_loader,
    epochs=2000,
    result_folder="results_2000_epochs_2000_samples",
    prior_std=5.0,
    add_gaussian_noise=False,
    train_unlabeled_loader=train_unlabeled_loader,
    save_interval=100
)
