import torch
import torch.nn as nn
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
from itertools import cycle
import csv
import os

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Encoder(nn.Module):
    def __init__(self, input_dim: int, ae_hidden: int, output_dim: int) -> None:
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, input_dim: int, ae_hidden: int, output_dim: int, use_sigmoid: bool) -> None:
        super(Decoder, self).__init__()
        layers = [
            nn.Linear(input_dim, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, ae_hidden),
            nn.ReLU(),
            nn.Linear(ae_hidden, output_dim)
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, z):
        return self.fc(z)

class Discriminator(nn.Module):
    def __init__(self, dc_hidden, latent_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, dc_hidden),
            nn.ReLU(),
            nn.Linear(dc_hidden, dc_hidden),
            nn.ReLU(),
            nn.Linear(dc_hidden, 1),
        )

    def forward(self, z):
        return self.fc(z).squeeze()

class SemiSupervisedAutoEncoderOptions(object):
    def __init__(
            self, 
            input_dim, 
            ae_hidden_dim, 
            disc_hidden_dim, 
            latent_dim_categorical, 
            latent_dim_style, 
            recon_loss_fn, 
            init_recon_lr,
            semi_supervised_loss_fn,
            init_semi_sup_lr,
            init_gen_lr, 
            init_disc_categorical_lr,
            init_disc_style_lr,
            use_decoder_sigmoid=True
        ):
        
        # Model dimensions
        self.input_dim = input_dim
        self.ae_hidden_dim = ae_hidden_dim
        self.disc_hidden_dim = disc_hidden_dim
        self.latent_dim_categorical = latent_dim_categorical
        self.latent_dim_style = latent_dim_style

        # Reconstruction loss fn and lr
        self.recon_loss_fn = recon_loss_fn
        self.init_recon_lr = init_recon_lr

        # Classification loss fn and lr
        self.semi_supervised_loss_fn = semi_supervised_loss_fn
        self.init_semi_sup_lr = init_semi_sup_lr

        # Generator / encoder lr
        self.init_gen_lr = init_gen_lr

        # Discriminator lrs
        self.init_disc_categorical_lr = init_disc_categorical_lr
        self.init_disc_style_lr = init_disc_style_lr

        # Decoder options
        self.use_decoder_sigmoid = use_decoder_sigmoid

        # Device for computation
        self.device = self._get_device()

    def _get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'


    def __repr__(self):
        return (
            f"<SemiSupervisedAutoEncoderOptions("  \
            f"input_dim={self.input_dim}, "  \
            f"encoder_hidden_dim={self.ae_hidden_dim}, "  \
            f"decoder_hidden_dim={self.disc_hidden_dim}, "  \
            f"latent_dim_categorical={self.latent_dim_categorical}, "  \
            f"latent_dim_style={self.latent_dim_style}, "  \
            f"use_decoder_sigmoid={self.use_decoder_sigmoid}, "  \
            f"device={self.device})>"
        )


class SemiSupervisedAdversarialAutoencoder(nn.Module):

    def __init__(self, options: SemiSupervisedAutoEncoderOptions):
        super(SemiSupervisedAdversarialAutoencoder, self).__init__()
        
        options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.options = options

        self.device = options.device
        self.encoder = Encoder(options.input_dim, options.ae_hidden_dim, options.latent_dim_categorical + options.latent_dim_style).to(options.device)
        self.decoder = Decoder(options.latent_dim_categorical + options.latent_dim_style, options.ae_hidden_dim, options.input_dim, options.use_decoder_sigmoid).to(options.device)
        self.cat_softmax = nn.Softmax(dim=1)

        self.discriminator_categorical = Discriminator(options.disc_hidden_dim, options.latent_dim_categorical).to(options.device)
        self.discriminator_style = Discriminator(options.disc_hidden_dim, options.latent_dim_style).to(options.device)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator_categorical.apply(weights_init)
        self.discriminator_style.apply(weights_init)


        # reconstruction phase optmizer
        self.recon_opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=options.init_recon_lr
        )

        # optimizer for semi-supervised phase
        self.semi_supervised_opt = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=options.init_semi_sup_lr
        )

        # optimizer for categorical discriminator 
        self.disc_cat_opt = torch.optim.AdamW(
            self.discriminator_categorical.parameters(),
            lr=options.init_disc_categorical_lr
        )

        # optimizer for style discriminator
        self.disc_style_opt = torch.optim.AdamW(
            self.discriminator_style.parameters(),
            lr=options.init_disc_style_lr
        )

        # optimizer for generator (encoder)
        self.gen_opt = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=options.init_gen_lr,
        )

        self.recon_loss = options.recon_loss_fn
        self.semi_supervised_loss = options.semi_supervised_loss_fn
        self.adv_loss = nn.BCEWithLogitsLoss()


    def forward_reconstruction(self, x):
        z_cat, z_style = self.forward_encoder(x)
        x_hat = self.decoder(torch.cat((z_cat, z_style), dim=1))
        return x_hat
    
    def forward_encoder(self, x):
        z = self.encoder(x)
        z_cat = self.cat_softmax(z[:, :self.options.latent_dim_categorical])
        z_style = z[:, self.options.latent_dim_categorical:]
        return z_cat, z_style
    
    def forward_no_softmax(self, x):
        z = self.encoder(x)
        z_cat = z[:, :self.options.latent_dim_categorical]
        z_style = z[:, self.options.latent_dim_categorical:]
        return z_cat, z_style
        
    def sample_latent_prior_gaussian(self, n: int , prior_std: float = 5.0) -> torch.Tensor:
        return torch.randn(n, self.options.latent_dim_style).to(self.device) * prior_std
    
    def sample_latent_prior_categorical(self, n: int) -> torch.Tensor:
        latent_dim = self.options.latent_dim_categorical
        labels = torch.randint(0, latent_dim, (n,), device=self.device)
        return F.one_hot(labels, num_classes=latent_dim).float().to(self.device)

    def train_mbgd(
        self,
        train_labeled_loader,
        val_loader,
        epochs,
        result_folder: str,
        prior_std=5.0,
        add_gaussian_noise=False,
        train_unlabeled_loader=None,
    ):
        os.makedirs(result_folder, exist_ok=True)
        with open(f'{result_folder}/train_log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'Recon_U', 'Recon_L', 'SemiSup',
                'Disc_Cat', 'Gen_Cat', 'Disc_Sty', 'Gen_Sty']
            )

        if train_unlabeled_loader is None:
            train_unlabeled_loader = train_labeled_loader

        unlabeled_iter = cycle(train_unlabeled_loader)   # endless

        for epoch in range(epochs):

            if epoch == 50:
                self.recon_opt.param_groups[0]['lr'] = 0.0001
                self.semi_supervised_opt.param_groups[0]['lr'] = 0.0001
                self.gen_opt.param_groups[0]['lr'] = 0.0001
                self.disc_style_opt.param_groups[0]['lr'] = 0.0001
                self.disc_cat_opt.param_groups[0]['lr'] = 0.0001
                
            elif epoch == 1000:
                self.recon_opt.param_groups[0]['lr'] = 0.00001
                self.semi_supervised_opt.param_groups[0]['lr'] = 0.00001
                self.gen_opt.param_groups[0]['lr'] = 0.00001
                self.disc_style_opt.param_groups[0]['lr'] = 0.00001
                self.disc_cat_opt.param_groups[0]['lr'] = 0.00001

            sums = dict(recon_u=0, recon_l=0, cls=0,
                        d_cat=0, g_cat=0, d_sty=0, g_sty=0)

            self.train()
            loop = tqdm(train_labeled_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

            for batch_idx, (x_l, y_l) in enumerate(loop, 1):

                # fetch one unlabeled batch
                x_u, _ = next(unlabeled_iter)
                x_l, y_l = x_l.to(self.device), y_l.to(self.device)
                x_u      = x_u.to(self.device)

                # optional Gaussian noise added to input
                x_l_target = x_l.clone()
                x_u_target = x_u.clone()
                if add_gaussian_noise:
                    x_l = x_l + torch.randn_like(x_l) * 0.3
                    x_u = x_u + torch.randn_like(x_u) * 0.3

                ##################################################
                # 1) Reconstruction phase  - LABELED + UNLABELED #
                ##################################################

                x_hat_u = self.forward_reconstruction(x_u)
                loss_rec_u = self.recon_loss(x_hat_u, x_u_target)
                loss_rec_l = self.recon_loss(self.forward_reconstruction(x_l), x_l_target)
                loss_rec = loss_rec_u + loss_rec_l

                self.recon_opt.zero_grad()
                loss_rec.backward()
                self.recon_opt.step()

                ##############################################################################
                # 2) Latent regularisation phase (GAN)  – BOTH labeled and unlabeled batches #
                ##############################################################################

                # concatenate labeled and unlabeled samples
                x_adv = torch.cat([x_l, x_u], dim=0) 
                batch_adv = x_adv.size(0)

              
                # categorical discriminator update 
                # real prior
                z_real_cat = self.sample_latent_prior_categorical(batch_adv)
                d_real_cat = self.adv_loss(
                    self.discriminator_categorical(z_real_cat),
                    torch.ones(batch_adv, device=self.device)
                )

                # fake data from encoder
                z_fake_cat, _ = self.forward_encoder(x_adv)
                d_fake_cat = self.adv_loss(
                    self.discriminator_categorical(z_fake_cat.detach()),
                    torch.zeros(batch_adv, device=self.device)
                )

                self.disc_cat_opt.zero_grad()
                (d_real_cat + d_fake_cat).backward()
                self.disc_cat_opt.step()

                # style discriminator update 
                # real prior
                z_real_sty = self.sample_latent_prior_gaussian(batch_adv, prior_std)
                d_real_sty = self.adv_loss(
                    self.discriminator_style(z_real_sty),
                    torch.ones(batch_adv, device=self.device)
                )

                # fake data from encoder
                _, z_fake_sty = self.forward_encoder(x_adv)
                d_fake_sty = self.adv_loss(
                    self.discriminator_style(z_fake_sty.detach()),
                    torch.zeros(batch_adv, device=self.device)
                )

                self.disc_style_opt.zero_grad()
                (d_real_sty + d_fake_sty).backward()
                self.disc_style_opt.step()

                # generator (encoder) adversarial loss
                g_cat = self.adv_loss(
                    self.discriminator_categorical(z_fake_cat),
                    torch.ones(batch_adv, device=self.device)
                )
                g_sty = self.adv_loss(
                    self.discriminator_style(z_fake_sty),
                    torch.ones(batch_adv, device=self.device)
                )

                self.gen_opt.zero_grad()
                (g_cat + g_sty).backward()
                self.gen_opt.step()

                ############################################################
                # 3) Semi-supervised classification - only labeled samples #
                ############################################################

                logits_cat, _ = self.forward_no_softmax(x_l)
                loss_cls = self.semi_supervised_loss(logits_cat, y_l)

                self.semi_supervised_opt.zero_grad()
                loss_cls.backward()
                self.semi_supervised_opt.step()

                sums['recon_u'] += loss_rec_u.item()
                sums['recon_l'] += loss_rec_l.item()
                sums['cls']     += loss_cls.item()
                sums['d_cat']   += (d_real_cat + d_fake_cat).item()
                sums['g_cat']   += g_cat.item()
                sums['d_sty']   += (d_real_sty + d_fake_sty).item()
                sums['g_sty']   += g_sty.item()

                loop.set_postfix({
                    'Recon_U': sums['recon_u']/batch_idx,
                    'Recon_L': sums['recon_l']/batch_idx,
                    'SemiSup': sums['cls']    /batch_idx,
                    'Disc_Cat':sums['d_cat']  /batch_idx,
                    'Gen_Cat': sums['g_cat']  /batch_idx,
                    'Disc_Sty':sums['d_sty']  /batch_idx,
                    'Gen_Sty': sums['g_sty']  /batch_idx
                })

            n = len(train_labeled_loader)
            avg = {k: v/n for k,v in sums.items()}
            print(
                f"Epoch {epoch+1}/{epochs} — "
                f"Recon_U: {avg['recon_u']:.4f}, Recon_L: {avg['recon_l']:.4f}, "
                f"SemiSup: {avg['cls']:.4f}, "
                f"Disc_Cat: {avg['d_cat']:.4f}, Gen_Cat: {avg['g_cat']:.4f}, "
                f"Disc_Sty: {avg['d_sty']:.4f}, Gen_Sty: {avg['g_sty']:.4f}"
            )

            # logging
            with open(f'{result_folder}/train_log.csv', 'a', newline='') as f:
                csv.writer(f).writerow(
                    [epoch+1, avg['recon_u'], avg['recon_l'], avg['cls'],
                    avg['d_cat'], avg['g_cat'], avg['d_sty'], avg['g_sty']]
                )

            # evaluation on validation set
            self.eval()
            correct = total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.view(vx.size(0), -1).to(self.device)
                    vy = vy.to(self.device)
                    preds = self.forward_no_softmax(vx)[0].argmax(1)
                    correct += (preds == vy).sum().item()
                    total   += vy.size(0)
            print(f"Validation Accuracy: {100*correct/total:.2f}%\n")

            if (epoch+1) % 50 == 0:
                ckpt_dir = f'{result_folder}/weights_epoch_{epoch+1}'
                os.makedirs(ckpt_dir, exist_ok=True)
                self.save_weights(f'{ckpt_dir}/weights')
            self.save_weights(f'{result_folder}/weights')


    def save_weights(self, path_prefix="aae_weights"):
        """
        Saves the weights of the encoder, decoder, and both discriminators.

        Args:
            path_prefix (str): Prefix for the saved file paths. Files will be saved as:
                - <path_prefix>_encoder.pth
                - <path_prefix>_decoder.pth
                - <path_prefix>_disc_categorical.pth
                - <path_prefix>_disc_style.pth
        """
        torch.save(self.encoder.state_dict(), f"{path_prefix}_encoder.pth")
        torch.save(self.decoder.state_dict(), f"{path_prefix}_decoder.pth")
        torch.save(self.discriminator_categorical.state_dict(), f"{path_prefix}_disc_categorical.pth")
        torch.save(self.discriminator_style.state_dict(), f"{path_prefix}_disc_style.pth")
        print(f"Weights saved to {path_prefix}_*.pth")

    def load_weights(self, path_prefix="aae_weights"):
        """
        Loads the weights of the encoder, decoder, and both discriminators.

        Args:
            path_prefix (str): Prefix for the saved file paths. Expected files are:
                - <path_prefix>_encoder.pth
                - <path_prefix>_decoder.pth
                - <path_prefix>_disc_categorical.pth
                - <path_prefix>_disc_style.pth
        """
        self.encoder.load_state_dict(torch.load(f"{path_prefix}_encoder.pth", map_location=self.device))
        self.decoder.load_state_dict(torch.load(f"{path_prefix}_decoder.pth", map_location=self.device))
        self.discriminator_categorical.load_state_dict(
            torch.load(f"{path_prefix}_disc_categorical.pth", map_location=self.device)
        )
        self.discriminator_style.load_state_dict(
            torch.load(f"{path_prefix}_disc_style.pth", map_location=self.device)
        )
        print(f"Weights loaded from {path_prefix}_*.pth")

    def predict(self, x):
        """
        Predict class labels for input batch x.

        Args:
            x (torch.Tensor): shape (N, 1, 28, 28) or (N, 784)

        Returns:
            probs (torch.Tensor): shape (N, latent_dim_categorical), class probabilities
            preds (torch.Tensor): shape (N,), predicted class indices
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 4:
                x = x.view(x.size(0), -1)

            z = self.encoder(x.to(self.device))
            logits_cat = z[:, : self.options.latent_dim_categorical]
            probs = F.softmax(logits_cat, dim=1)
            preds = probs.argmax(dim=1)

        return probs, preds
    

    def generate_images(self, labels, style_z=None, prior_std=5.0):
        """
        Generate images conditioned on digit labels and style codes.

        Args:
            labels (int or torch.Tensor):
                - If int: a single digit in [0, latent_dim_categorical-1]
                - If 1D Tensor of ints: batch of digit labels
            style_z (torch.Tensor, optional):
                - If provided: Tensor of shape (batch_size, latent_dim_style)
                - If None: sampled from Gaussian prior with std=prior_std
            prior_std (float): std deviation for style prior sampling

        Returns:
            images (torch.Tensor): shape (batch_size, 1, 28, 28), values in [0,1]
        """
        self.eval()
        with torch.no_grad():
            
            if isinstance(labels, int):
                labels = torch.tensor([labels], device=self.device)
            else:
                labels = labels.to(self.device)
            batch_size = labels.size(0)

            
            z_cat = F.one_hot(labels, num_classes=self.options.latent_dim_categorical) \
                     .float().to(self.device)

            
            if style_z is None:
                style_z = self.sample_latent_prior_gaussian(batch_size, prior_std)
            else:
                style_z = style_z.to(self.device)
                assert style_z.shape == (batch_size, self.options.latent_dim_style), \
                       f"Expected style_z of shape {(batch_size, self.options.latent_dim_style)}"

            
            z = torch.cat([z_cat, style_z], dim=1)
            x_hat = self.decoder(z)
            images = x_hat.view(batch_size, 1, 28, 28)

        return images

    def generate_images_style_match(self, labels, image):
        _, z_style = self.forward_encoder(image)
        return self.generate_images(labels, z_style=z_style)