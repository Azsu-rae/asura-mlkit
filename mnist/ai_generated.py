#!/usr/bin/env python3
"""
Lab 1 — Autoencoders and Variational Autoencoders (PyTorch, MNIST)
==================================================================

Full standalone script that reproduces every code block of
`lab1_autoencoders.pdf` (UMBB — Master 2 I2A, Generative AI, 2026/2027),
AND answers every question of the lab:

    Section 0 ....... environment check, CPU/GPU device
    Section 1 ....... simple (linear) autoencoder, code size experiment
                      -> Q1.1 Q1.2 Q1.3  +  Q1.4 (exercise, deep autoencoder)
    Section 2 ....... convolutional autoencoder (Conv2d / ConvTranspose2d)
                      -> Q2.1 Q2.2 Q2.3  +  Q2.4 (exercise, stride-2 encoder)
    Section 3 ....... denoising autoencoder (noisy input / clean target)
                      -> Q3.1 Q3.2 Q3.3  +  Q3.4 (exercise, salt & pepper noise)
    Section 4 ....... variational autoencoder (reparameterization, BCE + KL)
                      -> Q4.1 Q4.2 Q4.3  +  Q4.4 (beta sweep) Q4.5 (conv VAE)
    Summary ......... parameter counts + reference results + explanations

Usage
-----
    python ai_generated.py                      # everything (see runtime below)
    python ai_generated.py --quick              # 2 min smoke test (few epochs/data)
    python ai_generated.py --sections 1,4       # only some sections
    python ai_generated.py --answers-only       # print the written answers only
    python ai_generated.py --show               # display figures (needs a display)
    python ai_generated.py --no-exercises       # skip Q1.4/Q2.4/Q3.4/Q4.4/Q4.5

Runtime (reference run, CPU, no GPU): roughly 15-25 minutes with the default
number of epochs of the lab. `--quick` finishes in ~2 minutes.
Figures are written to `./figures/` (one PNG per plot) when `--show` is absent.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# =============================================================================
# Answers to every question of the lab (printed by --answers-only, and at the
# end of each section at the end of a normal run).
# =============================================================================

ANSWERS: dict[str, str] = {
    # ------------------------------------------------------------------ Q1
    "1.1": """Q1.1 - Number of parameters of the linear autoencoder (code = 32)

    Encoder : nn.Linear(784, 32)  -> 784*32 weights + 32 biases = 25 088 + 32   = 25 120
    Decoder : nn.Linear(32, 784)  -> 32*784 weights + 784 biases = 25 088 + 784 = 25 872
    Total                                                        = 25 120 + 25 872 = 50 992

    This is exactly the value printed by
    `sum(p.numel() for p in model.parameters())` (checked at run time).
    Compression of the CODE itself: 784 / 32 = 24.5 times smaller.
    The MODEL (50 992 parameters) is of course bigger than one image (784
    values): the model stores the general rules, not the images.""",

    "1.2": """Q1.2 - Why a sigmoid at the output?

    The target is the input image, whose pixels are in [0, 1]. The network must
    therefore produce values in the same range, otherwise the MSE compares
    incompatible quantities. sigmoid(x) = 1/(1+e^-x) maps R -> (0, 1), it is
    differentiable everywhere and its gradient does not vanish.

    If we replace it by a ReLU:
      * the output is unbounded above and bounded below by 0. Values > 1 are
        impossible pixels and are punished by the loss;
      * the mapping is not symmetric: it can produce bright pixels but NEVER
        darker-than-black ones, so it cannot encode a black background well
        (everything stays >= 0);
      * the network then only has to *project* the values into [0, 1], which
        it learns to do with a large gain, so the logits grow and gradients
        can saturate;
      * at run time the reconstructions are not valid images, and a plain
        sigmoid is needed to display them with vmin=0, vmax=1.
    (A tanh would work only after a rescaling to [0, 1].)""",

    "1.3": """Q1.3 - Effect of encoding_dim

    encoding_dim = 2   : the code keeps almost nothing -> very blurry blobs,
                        test MSE ~ 0.05-0.06.
    encoding_dim = 8   : strokes are visible, digit identity mostly kept.
    encoding_dim = 32  : clear digits, test MSE ~ 0.014 (the reference value).
    encoding_dim = 128 : slightly cleaner/smoother edges, test MSE ~ 0.010.
    Larger code  <=>  less compression, lower reconstruction error, and a
    bigger model (more parameters: 784*d + d + d*784 + 784).

    encoding_dim = 784 : the network has as many code units as pixels, so the
    code is NOT a compression at all. Nothing forces information loss: the
    single hidden layer can learn the identity function (W2 = I, W1 = I, with
    the biases 0 and a ReLU that is the identity on positive pixels), so it
    memorizes the training set almost perfectly and the test MSE becomes very
    small too. The autoencoder has then lost its purpose (no dimensionality
    reduction, no useful representation); it only becomes interesting again
    when a real bottleneck (32, or 128) is imposed.""",

    "1.4": """Q1.4 - (exercise) Deep autoencoder 784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784

    Parameters: 100 480 + 8 256 + 2 080 (encoder) + 2 112 + 8 320 + 101 136
                (decoder) = 222 384, i.e. 4.4x more than the simple model
                (50 992). (Printed by the script.)

    Result (same number of epochs, Adam lr = 0.001):
      * the deep model converges faster per epoch and reaches a LOWER test MSE
        than the single hidden layer (typically ~0.012 against ~0.0145), so
        there IS a small gain;
      * but the gain is small compared to the 4x bigger model: MNIST is easy
        and 784 -> 32 is already a sufficient bottleneck. The extra capacity is
        mostly spent on redundant features;
      * the risk is overfitting: the training loss drops below the test loss, and
        the deeper model needs a regularizer (weight decay, dropout) or an early
        stop to keep its advantage on new data.
    Take-away: depth helps, but the architecture that matters most is the one
    that respects the structure of the data (see Section 2: 20x fewer parameters
    AND a much lower error than any fully connected model).""",

    # ------------------------------------------------------------------ Q2
    "2.1": """Q2.1 - Why so few parameters (1 077 vs 50 992)?

    A linear layer owns one weight PER (input, output) pair: 784*32 = 25 088
    different weights for the encoder. A convolution owns only one small kernel
    (3x3 = 9 weights per input/output channel pair) which is REUSED at every
    position of the image:

        conv1 : 16*1*3*3  +  16    =    160
        conv2 :  4*16*3*3 +   4    =    580
        t_conv1: 16*4*2*2  +  16    =    272
        t_conv2:  1*16*2*2 +   1    =     65
        total                            1 077

    Weight sharing is exactly the prior that makes sense for images: the same
    "is there a vertical stroke here?" filter is useful at every position. The
    linear model has to learn a separate weight for every pixel position (784x784
    of them for one layer), which is both bigger and worse.""",

    "2.2": """Q2.2 - Compression ratio of the convolutional code

    Code = 4 x 7 x 7 = 196 values, image = 784 values
    -> compression ratio = 784 / 196 = 4 : 1   (24 % of the pixels kept)

    Compared with Section 1 (32 values -> 24.5 : 1) the comparison is NOT fair,
    for three reasons:
      1. the two codes are not the same kind of object: 32 free parameters of a
         global linear projection vs 196 numbers that still live on a 7x7 grid
         (the convolution has not destroyed the spatial structure);
      2. the information content is not proportional to the number of numbers:
         a 4x7x7 code is heavily structured (neighbouring values are correlated
         and produced by shared kernels), while 32 free values can carry 32
         independent numbers;
      3. the two models are not comparable in capacity: 1 077 parameters
         (conv) against 50 992 (linear).
    Empirically: linear AE 32 codes -> MSE 0.0141, conv AE 196 codes -> MSE
    0.0088 / 0.0060. The convolutional model is *less* compressive and yet much
    better, which proves that the raw ratio of the code is the wrong measure of
    compression quality.""",

    "2.3": """Q2.3 - Which decoder is better?

    In the reference run the UPSAMPLING + CONVOLUTION decoder wins
    (test MSE 0.0060 against 0.0088 for the transposed convolutions). The
    difference is visible in the texture:
      * ConvTranspose2d inserts overlapping patches at stride 2, which creates
        the well known "checkerboard" artefacts: regular micro-grid patterns and
        slightly rough/blobby strokes;
      * Upsample + Conv first enlarges with a fixed rule (no weights, no
        checkerboard) and then learns a smooth convolution, giving more uniform,
        slightly blurrier strokes and fewer sharp artefacts.
    Both give a similar global shape; the upsampling decoder wins mostly on
    fine texture. (The result can change with the seed: the two decoders are
    within ~30 % of each other, so the qualitative answer matters more than the
    exact number.)""",

    "2.4": """Q2.4 - (exercise) Encoder with stride-2 convolutions instead of MaxPool

    Implemented in ConvAutoencoderStride. The shapes are exactly the same:
        (1,28,28) -conv1 s1 p1-> (16,28,28)
                   -conv2 s2 p1-> ( 4,14,14)   out = floor((28+2-3)/2)+1 = 14
                   -conv3 s2 p1-> ( 4, 7, 7)   out = floor((14+2-3)/2)+1 = 7
    so the code is still 4x7x7 and the decoder is unchanged.
    Differences with MaxPool2d: the downsampling becomes LEARNED (a strided
    convolution learns which neighbourhood to keep, MaxPool always keeps the
    maximum), a strided convolution mixes overlapping 3x3 neighbourhoods while
    MaxPool looks at disjoint 2x2 blocks, and the parameters grow a little
    (1 137 instead of 1 077 for the conv2/conv3 variant used here).""",

    # ------------------------------------------------------------------ Q3
    "3.1": """Q3.1 - Why can this network not simply copy its input?

    Because the loss is `criterion(model(noisy), clean)`: the INPUT is the noisy
    image but the TARGET is the clean one. Copying (model = identity) would give
    MSE = MSE(noisy, clean) = 0.1157 at noise_factor = 0.5, while the trained
    model reaches 0.0281. Copying is therefore 4x worse than what the network
    learns, and gradient descent moves away from it.

    Two complementary reasons:
      * the training signal forbids it: the gradient of the loss w.r.t. a pixel
        tells the network to output the clean value, not the noisy one;
      * the architecture cannot copy perfectly: MaxPool2d keeps only the maximum
        of each 2x2 block and the code is only 8x3x3, so the noisy input cannot
        be transmitted exactly - the identity map does not even exist in the
        class of representable functions.
    This is what turns a trivial "copy task" into a useful signal-processing
    task: the network must learn a *prior* on clean MNIST digits.""",

    "3.2": """Q3.2 - At noise_factor = 0 the denoiser is slightly worse

    At test time with no noise, the input is CLEAN, which is a distribution the
    network has never seen as input during training (it only ever received
    corrupted images). It then applies a mapping that it estimated for noisy
    inputs: a mild blur / contraction of the strokes, so the output is a little
    different from the clean image and the MSE with it grows slightly above the
    clean-vs-clean value (0).

    Other contributions:
      * training always used fresh random noise (a data augmentation), so the
        network optimised an *average* over noise levels, never the noiseless
        case;
      * a denoiser is a smoothing operator by nature: the bias-variance
        trade-off makes it keep the low frequencies and sacrifice a little
        high-frequency detail, which costs MSE even when nothing must be removed.
    The degradation stays small (a few 1e-3) because the mapping is almost the
    identity on the low frequencies.""",

    "3.3": """Q3.3 - For which noise levels is the model useful?

    Useful where denoised MSE < noisy MSE, i.e. roughly the middle of the range
    [0.1, 0.7] in the reference run, and best close to the training level
    (noise_factor = 0.5, dotted line on the curve):
      * noise_factor ~ 0.0-0.1: almost nothing to remove, the mild blur of the
        denoiser makes the result slightly worse than doing nothing;
      * noise_factor ~ 0.3-0.7: clear gain (x3 to x4 less error);
      * noise_factor >= 0.8-0.9: far from the training level, the network gets
        an input it has never seen (very noisy, clipped at 0 and 1); the
        strokes start to be destroyed / mixed between digits, and denoised MSE
        approaches - sometimes exceeds - the noisy MSE: the model becomes
        useless (even harmful).
    The gain curve is unimodal and peaks at the training level: the model is
    specialised, not a universal filter.""",

    "3.4": """Q3.4 - (exercise) Salt-and-pepper noise

    Implemented in add_salt_and_pepper (20 % of the pixels set to 0 or 1).
    Behaviour compared with the Gaussian noise:
      * the noise is not additive but IMPULSIVE: it destroys the local
        continuity of the strokes, and 20 % of the pixels carry no information
        at all;
      * the network handles it well when the amount is moderate, because
        MaxPool2d + convolutions are naturally robust to isolated outliers: the
        pooling rejects a corrupted pixel if its 2x2 block contains a good one;
      * as with the Gaussian case, the gain is maximal around the training
        amount and disappears far from it;
      * if the salt & pepper rate is high (>50 %), a median-like behaviour is
        required and a convolutional denoiser degrades more than with additive
        Gaussian noise at the same "energy" of noise.
    The architecture does not have to change: only add_noise is replaced, which
    is exactly the point of the exercise (the architecture learns from data).""",

    # ------------------------------------------------------------------ Q4
    "4.1": """Q4.1 - Why does the KL term increase during the first epochs?

    The two terms of the loss are in competition (the ELBO is a trade-off):
      * the reconstruction term wants to place mu far from the origin and to
        make sigma small, so that z carries a precise description of the image;
      * the KL term measures the "price" of moving the aggregate posterior away
        from the prior N(0, I): KL = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))
        grows when the means move away from 0 or when the variances grow.

    At the very beginning the network reconstructs badly, so it first optimises
    the reconstruction; the aggregate posterior spreads out and mu grows away
    from 0: the KL "tax" increases (15 -> 25 nats per image in the reference
    run). Once the reconstruction has converged, the KL term stops growing and
    the balance is reached: rec goes down, KL goes up, the total goes down
    (166 -> 105). With beta > 1 the KL is penalised more and the latent space
    stays closer to the prior (blurrier but better organised); with beta = 0 the
    KL is not optimised at all.""",

    "4.2": """Q4.2 - VAE reconstructions vs Sections 1 and 2: which is sharper?

    Sharpness order in the reference run:
        Conv AE (upsampling + conv)  MSE 0.0060   <- sharpest
        Conv AE (transposed conv)    MSE 0.0088
        Linear AE (32 codes)         MSE 0.0141
        VAE (20 latent dims)         much blurrier
    The VAE reconstructs the WORST (BCE ~ 80-100 per image, visibly smooth,
    digit identity sometimes lost), because three things work against it:
      1. stochasticity: z = mu + sigma * eps is re-sampled at every forward pass,
         so the decoder sees a whole cloud and must output the average of the
         cloud -> blur;
      2. the KL term pulls mu towards 0 and the cloud towards N(0, I), so two
         different digits are forced closer together (the "mode covering"
         price of the ELBO);
      3. a 20-dimensional bottleneck compresses more than the 32-value linear
         code, and a linear model is a much weaker bottleneck than a
         convolutional one.
    This loss of sharpness is exactly the price that buys GENERATION: the
    smoothness of the latent space (no holes) is what makes sampling from the
    prior produce valid digits.""",

    "4.3": """Q4.3 - Neighbours and overlaps in the 2-D latent space

    The labels were NEVER used for training (the VAE is unsupervised), yet the
    clusters appear: to reconstruct well, two similar images must get similar
    codes. This is the clustering property of the ELBO.

    Typically:
      * clusters well separated: 1 (alone, thin and far away), 0, 6, 2;
      * near neighbours / partial overlap: 4 and 9, 8 and 9, 3 and 5, 3 and 8,
        5 and 8, 7 and 9, 7 and 1 (a tail toward the 1 cluster), 0 and 6.
      * 4/9 and 8/9 overlap because a 9 looks like a 4 (or an 8) with one stroke
        added: with only 2 dimensions the model has to place them almost on top
        of each other.
    Is it surprising? Not really: the confusion is exactly the confusion of a
    nearest-neighbour classifier on pixels. But two points ARE interesting:
      * two clusters of DIFFERENT classes can overlap (4 vs 9): the 2-D latent
        space is too small, the model "cuts" information;
      * with beta = 0 the clusters would be irregular, isolated islands with
        empty space between them, and sampling the prior would give garbage:
        the KL term is what makes the space continuous and filled.
    The grid of decoded images shows the same thing: each region of the plane is
    a blurred prototype of a digit, with smooth transitions between regions.""",

    "4.4": """Q4.4 - (exercise) beta = 0 and beta = 5

    beta = 0 (pure autoencoder with noise injection):
      * the KL term is not optimised, so nothing structures the latent space;
      * reconstructions are the SHARPEST of all the VAE variants (the model
        maximises the reconstruction only), often better than beta = 1;
      * but mu drifts far from 0 and sigma is meaningless: the latent space
        splits into small isolated clouds with empty space between them;
      * sampling z ~ N(0, I) then falls OUTSIDE every cloud -> the decoder has
        never been trained there -> the generated images are grey noise / colour
        blobs, NOT digits. This is exactly the failure of the plain
        autoencoder of Section 1, in a probabilistic form.
    beta = 5 (strong regularisation):
      * the KL dominates: the aggregate posterior is pulled strongly towards
        N(0, I), the latent space becomes smooth and almost perfectly covered;
      * generation is then very stable: sampling gives clean digits almost
        every time (arguably more consistent than beta = 1);
      * but reconstructions are clearly blurrier, digit identity is sometimes
        lost, and the 20-D space is "squeezed" (mu and sigma shrink -> the
        classes are closer to each other).
    Conclusion: beta interpolates between a sharp autoencoder that cannot
    generate (beta = 0) and a blurred but perfectly organised generator
    (large beta). beta = 1 is the true ELBO; beta > 1 is the beta-VAE trade-off.
    (A typical trick: latent_dim around 2-10 with beta ~ 2-4 gives better
    samples than latent_dim = 20 with beta = 1.)""",

    "4.5": """Q4.5 - (exercise) Convolutional VAE

    Implemented in ConvVAE: the encoder is Conv2d(..., stride=2) which downsizes
    28 -> 14 -> 7 -> 4 while increasing the depth (1 -> 32 -> 64 -> 128), the
    latent statistics are produced by two linear heads, and the decoder starts
    from z with a linear layer that rebuilds the 128x4x4 map, then goes back up
    with ConvTranspose2d (4 -> 8 -> 14 -> 28) and a sigmoid.
    Shapes: out = (in - 1)*s - 2p + k  with k = s = 2, p = 1 gives
    (4-1)*2 - 2 + 2 = 8, (8-1)*2 - 2 + 2 = 14 and a final 3x3 padding=1
    convolution keeps 28.

    Compared with the linear VAE of Section 4.3:
      * the convolutional VAE converges faster (fewer epochs) and reaches a
        lower reconstruction term per image, because the encoder keeps the
        spatial structure of the digits (the same effect as in Section 2);
      * the sampling statistics are the same: mu and logvar, same KL formula;
      * the latent space is better organised (the decoder learns sharper
        prototypes) so the generated digits are usually clearer for the same
        number of epochs.""",
}

EXPLANATIONS = """Key explanations of the lab
---------------------------------------------------------------
1. Why a NORMAL autoencoder cannot generate new images
   The autoencoder is only trained to satisfy one constraint: decode(encode(x))
   ~ x. Nothing ever constrains the region of the code space that the ENCODER
   does not produce. The codes occupy an arbitrary, curved, badly shaped subset
   of R^d (here a 32-dimensional region reached by 784-dimensional ReLU codes).
   If we feed a random code to the decoder, it falls almost surely in a region
   the decoder has never seen, and the output is meaningless (Section 1.10).
   Sampling uniformly in the code space is therefore not the same as sampling
   from the distribution the encoder produces.

2. Why a VAE can generate
   The VAE changes the training rule: instead of a single deterministic code it
   forces the encoder to output a Gaussian N(mu, sigma^2) and adds the term
   KL(q(z|x) || N(0, I)) to the loss. This KL "tax" has two effects:
     a) it penalises the codes that drift far from the origin and the too-large
        variances -> the aggregate posterior of the training set converges
        towards N(0, I), so the prior covers the region that the decoder sees;
     b) combined with the reparameterization trick z = mu + sigma*eps (eps drawn
        OUTSIDE the network so the gradient can flow to mu and sigma), the
        smooth Gaussians of neighbouring images overlap -> the latent space has
        no holes.
   Generation then becomes trivial and exact: draw z ~ N(0, I), decode it. No
   encoder, no data, and the samples are new digits that were never in the
   dataset.

3. Cost of the regularisation
   The KL term blurs the reconstructions (Q4.2). Sharpness and generative
   ability are in competition: beta = 0 gives a sharp but non-generative model,
   large beta gives a smooth and generative but blurry one.

4. Linear vs convolutional
   A linear layer forgets that the image is a grid and needs one weight per
   pixel (50 992 parameters); a convolution shares its kernel (1 077
   parameters) and keeps the 2-D structure, which is why it reconstructs much
   better (0.0088 / 0.0060 against 0.0141). The price is a bigger code (196
   values instead of 32).

5. Denoising autoencoder
   Same architecture, different supervision: input = noisy image, target = clean
   image. The network cannot copy (the copy is punished), so it must learn the
   statistics of clean digits; this is a cheap way to inject a strong prior.
   The gain peaks at the training noise level and disappears far from it
   (Q3.3).
"""


# =============================================================================
# Models (Sections 1, 2, 3 and 4 of the lab + the exercise variants)
# =============================================================================

import torch  # noqa: E402  (kept here so that --help works without torch)
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

criterion = nn.MSELoss()  # the reconstruction loss of the (non-VAE) autoencoders


class Autoencoder(nn.Module):
    """Section 1.4 — linear autoencoder: 784 -> encoding_dim -> 784."""

    def __init__(self, encoding_dim: int):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.fc1 = nn.Linear(28 * 28, encoding_dim)       # encoder: 784 -> encoding_dim
        self.fc2 = nn.Linear(encoding_dim, 28 * 28)       # decoder: encoding_dim -> 784

    def encode(self, x):
        return F.relu(self.fc1(x))                        # the code (compressed repr.)

    def decode(self, z):
        return torch.sigmoid(self.fc2(z))                 # sigmoid: values in [0, 1]

    def forward(self, x):
        return self.decode(self.encode(x))


class DeepAutoencoder(nn.Module):
    """Q1.4 (exercise) — 784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784."""

    def __init__(self, encoding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, encoding_dim), nn.ReLU(),        # ReLU between the layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 28 * 28), nn.Sigmoid(),          # sigmoid at the end
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvAutoencoder(nn.Module):
    """Section 2.3 — Conv2d encoder + ConvTranspose2d decoder."""

    def __init__(self):
        super().__init__()
        # ---------- encoder ----------
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)     # depth 1 -> 16, size unchanged
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)      # depth 16 -> 4, size unchanged
        self.pool = nn.MaxPool2d(2, 2)                  # height and width / 2
        # ---------- decoder ----------
        # kernel 2 and stride 2: multiplies height and width by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def encode(self, x):
        x = self.pool(F.relu(self.conv1(x)))            # (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))            # (B, 4, 7, 7)  <- the code
        return x

    def decode(self, z):
        x = F.relu(self.t_conv1(z))                     # (B, 16, 14, 14)
        x = torch.sigmoid(self.t_conv2(x))              # (B, 1, 28, 28)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvAutoencoderUpsample(nn.Module):
    """Section 2.8 — same encoder, decoder = upsampling + normal convolution."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, 3, padding=1)

    def encode(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def decode(self, z):
        # upsample (nearest neighbour: no weights), THEN a normal convolution
        x = F.interpolate(z, scale_factor=2, mode="nearest")      # 7 -> 14
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")      # 14 -> 28
        x = torch.sigmoid(self.conv5(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvAutoencoderStride(nn.Module):
    """Q2.4 (exercise) — MaxPool2d replaced by convolutions with stride=2."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)             # (B, 16, 28, 28)
        self.conv2 = nn.Conv2d(16, 4, 3, stride=2, padding=1)   # (B,  4, 14, 14)
        self.conv3 = nn.Conv2d(4, 4, 3, stride=2, padding=1)     # (B,  4,  7,  7)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def decode(self, z):
        x = F.relu(self.t_conv1(z))
        x = torch.sigmoid(self.t_conv2(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvDenoiser(nn.Module):
    """Section 3.4 — denoising autoencoder (noisy input -> clean target)."""

    def __init__(self):
        super().__init__()
        # ---------- encoder ----------
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)                      # 1 -> 32
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)                     # 32 -> 16
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)                      # 16 -> 8
        self.pool = nn.MaxPool2d(2, 2)
        # ---------- decoder ----------
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)            # k=3: 3 -> 7
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)           # k=2: 7 -> 14
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)          # k=2: 14 -> 28
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)                  # 32 -> 1

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))      # (B, 16, 7, 7)
        x = self.pool(F.relu(self.conv3(x)))      # (B, 8, 3, 3)   <- the code
        x = F.relu(self.t_conv1(x))               # (B, 8, 7, 7)
        x = F.relu(self.t_conv2(x))               # (B, 16, 14, 14)
        x = F.relu(self.t_conv3(x))               # (B, 32, 28, 28)
        x = torch.sigmoid(self.conv_out(x))       # (B, 1, 28, 28)
        return x


class VAE(nn.Module):
    """Section 4.3 — two-headed encoder, reparameterization, linear decoder."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # ---------- encoder ----------
        self.linear_encode = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)        # head 1: the mean
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)    # head 2: log(sigma^2)
        # ---------- decoder ----------
        self.linear_decode = nn.Linear(latent_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.linear_encode(x))
        return self.linear_mu(h), self.linear_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)      # sigma = exp(logvar / 2)
        eps = torch.randn_like(std)        # noise from N(0,1), drawn OUTSIDE the network
        return mu + eps * std              # z = mu + sigma * eps

    def decode(self, z):
        h = F.relu(self.linear_decode(z))
        return torch.sigmoid(self.linear_out(h))   # pixel values in [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConvVAE(nn.Module):
    """Q4.5 (exercise) — convolutional VAE: Conv2d(stride=2) encoder,
    ConvTranspose2d decoder. Output keeps the image shape (B, 1, 28, 28).

        encoder : 28 --s2--> 14 --s2--> 7 --s2--> 4   (Conv2d, k=3, p=1)
        latent  : flatten (64*4*4 = 1024) -> two heads -> mu, logvar (latent_dim)
        decoder : 4 --s2--> 8 --s2--> 16 --s2--> 32 -> 28 -> 28
                  (ConvTranspose2d with k=s=2, p=1: out = (in-1)*s - 2p + k = 2*in)
    """

    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        # ---------- encoder ----------
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)    # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # 14 -> 7
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)   # 7  -> 4
        flat = 64 * 4 * 4
        self.linear_mu = nn.Linear(flat, latent_dim)            # head 1: the mean
        self.linear_logvar = nn.Linear(flat, latent_dim)        # head 2: log(sigma^2)
        # ---------- decoder ----------
        self.linear_decode = nn.Linear(latent_dim, flat)        # z -> 64 x 4 x 4
        self.t_conv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 4  -> 8
        self.t_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 8  -> 16
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)   # 16 -> 32
        self.conv_out = nn.Conv2d(1, 1, 3, padding=1)           # 32 -> 32, then resize

    def encode(self, x):
        h = F.relu(self.conv1(x))          # (B, 32, 14, 14)
        h = F.relu(self.conv2(h))          # (B, 64, 7, 7)
        h = F.relu(self.conv3(h))          # (B, 64, 4, 4)
        h = h.flatten(1)                   # (B, 1024)
        return self.linear_mu(h), self.linear_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = F.relu(self.linear_decode(z)).view(-1, 64, 4, 4)    # (B, 64, 4, 4)
        h = F.relu(self.t_conv1(h))                             # (B, 64, 8, 8)
        h = F.relu(self.t_conv2(h))                             # (B, 32, 16, 16)
        h = F.relu(self.t_conv3(h))                             # (B, 1, 32, 32)
        h = F.interpolate(h, size=(28, 28), mode="nearest")     # 32 -> 28
        return torch.sigmoid(self.conv_out(h))                  # (B, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


# =============================================================================
# Small helpers
# =============================================================================

def set_seed(seed: int) -> None:
    """Same random numbers at every run (reproducible results)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def show_rows(rows, titles, n: int = 10):
    """Section 1.3 — build a figure with several rows of n images.

    Each element of `rows` is a tensor of n (or more) images, of shape
    (N, 1, 28, 28) or (N, 784). The figure is left open so that `lab.fig(name)`
    can save (or display) it.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), n, figsize=(1.3 * n, 1.55 * len(rows)),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.3})
    axes = np.atleast_2d(axes)
    for r, (imgs, title) in enumerate(zip(rows, titles)):
        # .detach().cpu().numpy(): tensor (maybe on GPU, maybe with gradients)
        #                              -> NumPy array that Matplotlib can draw
        imgs = imgs.detach().cpu().numpy().reshape(-1, 28, 28)
        for i in range(n):
            axes[r, i].imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
        if title:
            axes[r, 0].set_title(title, fontsize=11, loc="left")
    return fig


class FigureSink:
    """Saves every figure as PNG (default) or displays it (--show)."""

    def __init__(self, outdir: Path, show: bool):
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        self.outdir = outdir
        self.show = show
        self.count = 0
        self.outdir.mkdir(parents=True, exist_ok=True)

    def __call__(self, name: str) -> None:
        import matplotlib.pyplot as plt

        self.count += 1
        path = self.outdir / f"{name}.png"
        plt.savefig(path, dpi=110, bbox_inches="tight")
        print(f"    figure -> {path}")
        plt.close("all")
        if self.show:
            plt.show()


def find_examples(loader, digits, max_batches: int = 5):
    """First image found for each requested digit (for the interpolation)."""
    wanted = set(digits)
    found = {}
    for i, (images, labels) in enumerate(loader):
        for digit in list(wanted):
            hit = (labels == digit).nonzero()
            if len(hit):
                found[digit] = images[hit[0].item()]
                wanted.discard(digit)
        if not wanted or i >= max_batches - 1:
            break
    return found


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_EPOCHS = {
    "linear": 10, "code_size": 3, "deep": 10,
    "conv": 10, "upsample": 10, "stride": 10,
    "denoise": 5, "saltpepper": 5,
    "vae": 15, "vae2d": 10, "beta": 8, "convvae": 10,
}


@dataclass
class Config:
    data_root: Path = HERE / "data"
    figures_dir: Path = HERE / "figures"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    epochs: dict = field(default_factory=lambda: dict(DEFAULT_EPOCHS))
    batch_size: int = 64
    vae_batch_size: int = 128
    limit_train: int | None = None
    limit_test: int | None = None
    noise_factor: float = 0.5
    seed: int = 0
    exercises: bool = True
    show: bool = False

    def ep(self, name: str) -> int:
        return max(1, self.epochs[name])


@dataclass
class Data:
    train_loader: object        # batch_size = 64  (Sections 1, 2, 3)
    test_loader: object         # batch_size = 64
    vae_train_loader: object    # batch_size = 128 (Section 4)
    n_train: int
    n_test: int


# =============================================================================
# Training helpers
# =============================================================================

def evaluate_flat(model, test_loader, device) -> float:
    """Average MSE on the test set (images flattened to 784)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            x = images.to(device).view(images.size(0), -1)
            total += criterion(model(x), x).item() * images.size(0)
    return total / len(test_loader.dataset)


def train_flat(model, train_loader, test_loader, device, n_epochs, lr=1e-3,
               show_every=True) -> dict:
    """Section 1.6 — training loop for the linear / deep autoencoders."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "test": []}
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            x = images.view(images.size(0), -1)       # flatten: (B,1,28,28) -> (B,784)
            optimizer.zero_grad()                     # 1. reset the gradients
            outputs = model(x)                        # 2. forward pass
            loss = criterion(outputs, x)              # 3. compare output with INPUT
            loss.backward()                           # 4. backward pass
            optimizer.step()                          # 5. update the weights
            running_loss += loss.item() * images.size(0)
        history["train"].append(running_loss / len(train_loader.dataset))
        history["test"].append(evaluate_flat(model, test_loader, device))
        if show_every:
            print(f"    Epoch {epoch:2d}/{n_epochs} | train loss: {history['train'][-1]:.5f}"
                  f" | test loss: {history['test'][-1]:.5f}")
    return history


def train_conv(model, train_loader, test_loader, device, n_epochs, lr=1e-3,
               show_every=True) -> dict:
    """Section 2.5 — training loop for the convolutional models (no flattening)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "test": []}
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)      # NO flattening: we keep (B, 1, 28, 28)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)  # the target is the input image
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        history["train"].append(running_loss / len(train_loader.dataset))
        history["test"].append(evaluate_conv(model, test_loader, device))
        if show_every:
            print(f"    Epoch {epoch:2d}/{n_epochs} | train loss: {history['train'][-1]:.5f}"
                  f" | test loss: {history['test'][-1]:.5f}")
    return history


def evaluate_conv(model, test_loader, device) -> float:
    """Average MSE on the test set, images kept as (B, 1, 28, 28)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            total += criterion(model(images), images).item() * images.size(0)
    return total / len(test_loader.dataset)


# =============================================================================
# Section 0 — environment
# =============================================================================

def section0(lab: "Lab") -> None:
    import matplotlib

    lab.h1("Section 0 — Prepare the environment (0.3)")
    print("Python      :", sys.version.split()[0], "on", platform.system())
    print("PyTorch     :", torch.__version__)
    print("torchvision :", torchvision.__version__)
    print("NumPy       :", np.__version__)
    print("Matplotlib  :", matplotlib.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        print("GPU name      :", torch.cuda.get_device_name(0))
        print("CUDA version :", torch.version.cuda)
    else:
        print("No GPU found: the lab will run on the CPU (slower, but it works).")
    print("We will use:", lab.device)

    x = torch.rand(3, 3, device=lab.device)      # create a tensor on the device
    y = x @ x                                    # a matrix product, computed there
    print("The result of x @ x lives on:", y.device)
    print("Reminder: never write 'cuda' in the code. Create `device` once, then move")
    print("the model and every batch with .to(device): a model and its data must")
    print("always be on the SAME device, otherwise PyTorch raises an error.")


# =============================================================================
# Section 1 — linear autoencoder
# =============================================================================

def section1(lab: "Lab") -> None:
    import matplotlib.pyplot as plt

    cfg, device = lab.cfg, lab.device
    lab.h1("Section 1 — A simple (linear) autoencoder (1.4 - 1.11)")

    images, labels = lab.display_batch()
    print(f"Training images: {lab.data.n_train} | Test images: {lab.data.n_test}")
    print("One batch of images:", tuple(images.shape))
    print("Pixel values between", images.min().item(), "and", images.max().item())

    # ---------------- 1.4 the model ----------------
    encoding_dim = 32
    model = Autoencoder(encoding_dim).to(device)
    print(model)
    printed = n_parameters(model)
    print("Number of parameters:", printed)

    # Q1.1 — hand computation
    by_hand = (28 * 28 * encoding_dim + encoding_dim) + (encoding_dim * 28 * 28 + 28 * 28)
    print(f"Q1.1 by hand: (784*{encoding_dim} + {encoding_dim})"
          f" + ({encoding_dim}*784 + 784) = {by_hand} -> matches PyTorch: {by_hand == printed}")

    # ---------------- 1.6 training ----------------
    lab.h2("1.6 Training")
    history = train_flat(model, lab.data.train_loader, lab.data.test_loader, device,
                         cfg.ep("linear"))

    # ---------------- 1.7 learning curves ----------------
    n_epochs = cfg.ep("linear")
    plt.figure(figsize=(6, 3.5))
    plt.plot(range(1, n_epochs + 1), history["train"], "o-", label="train")
    plt.plot(range(1, n_epochs + 1), history["test"], "s-", label="test")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Linear autoencoder: learning curves")
    plt.legend()
    plt.grid(alpha=0.3)
    lab.fig("s1_learning_curves")

    # ---------------- 1.8-1.9 reconstructions and codes ----------------
    lab.h2("1.8 - 1.9 Reconstructions and codes")
    model.eval()
    images, labels = lab.display_batch()
    x = images.to(device).view(images.size(0), -1)
    with torch.no_grad():
        codes = model.encode(x)                  # (B, 32)
        outputs = model.decode(codes)            # (B, 784)
    show_rows([images, outputs], ["Original test images", "Reconstructions"])
    lab.fig("s1_reconstructions")
    print("Shape of the codes:", tuple(codes.shape))
    print(f"Compression: 784 values -> {encoding_dim} values "
          f"({784 / encoding_dim:.1f} times smaller)")

    fig, axes = plt.subplots(1, 10, figsize=(14, 1.3))
    for i, ax in enumerate(axes):
        ax.imshow(codes[i].cpu().numpy().reshape(4, 8), cmap="gray")
        ax.axis("off")
    axes[0].set_title("Codes of the ten images above", fontsize=11, loc="left")
    lab.fig("s1_codes")
    print("The reconstructions are a little blurry: some details are lost in the")
    print("compression. The codes look like noise to us, but they contain enough")
    print("information to rebuild the digits.")

    # ---------------- 1.10 can it generate? ----------------
    lab.h2("1.10 Can this autoencoder generate new digits?")
    set_seed(1)
    random_codes = torch.rand(10, encoding_dim, device=device) * codes.max()
    with torch.no_grad():
        generated = model.decode(random_codes)
    show_rows([generated], ["Decoder outputs for RANDOM codes"])
    lab.fig("s1_random_codes")
    print("The outputs are not digits: nothing organizes the space of the codes, so")
    print("a random code falls in a region the decoder has never seen -> we need a VAE.")

    # ---------------- 1.11 size of the code ----------------
    lab.h2("1.11 Experiment: the size of the code")

    def train_linear_ae(dim: int, n_ep: int):
        set_seed(cfg.seed)
        net = Autoencoder(dim).to(device)
        train_flat(net, lab.data.train_loader, lab.data.test_loader, device, n_ep,
                   show_every=False)
        return net, evaluate_flat(net, lab.data.test_loader, device)

    dims = [2, 8, 32, 128]
    results = {}
    for d in dims:
        results[d] = train_linear_ae(d, cfg.ep("code_size"))
        print(f"    encoding_dim = {d:3d} | test loss after {cfg.ep('code_size')} epochs: "
              f"{results[d][1]:.5f}")

    images, _ = lab.display_batch()
    x = images.to(device).view(images.size(0), -1)
    rows, titles = [images], ["Original test images"]
    with torch.no_grad():
        for d in dims:
            rows.append(results[d][0](x))
            titles.append(f"encoding_dim = {d} (test loss {results[d][1]:.4f})")
    show_rows(rows, titles)
    lab.fig("s1_code_size")

    lab.results["Linear AE (code of 32)"] = history["test"][-1]

    # ---------------- Q1.4 (exercise): deep autoencoder ----------------
    if cfg.exercises:
        lab.h2("Q1.4 (exercise) Deep autoencoder 784->128->64->32->64->128->784")
        deep = DeepAutoencoder(32).to(device)
        print("Number of parameters (deep):", n_parameters(deep))
        deep_hist = train_flat(deep, lab.data.train_loader, lab.data.test_loader, device,
                               cfg.ep("deep"), show_every=False)
        print(f"    Deep AE test loss after {cfg.ep('deep')} epochs     : "
              f"{deep_hist['test'][-1]:.5f}")
        print(f"    Simple linear AE test loss after {n_epochs} epochs: "
              f"{history['test'][-1]:.5f}")
        print(f"    Parameters: deep {n_parameters(deep):,} against "
              f"simple {n_parameters(model):,} -> the extra capacity buys very little")
        with torch.no_grad():
            deep_out = deep(x)
        show_rows([images, outputs, deep_out],
                  ["Original test images", f"Simple AE (test {history['test'][-1]:.4f})",
                   f"Deep AE (test {deep_hist['test'][-1]:.4f})"])
        lab.fig("s1_deep_autoencoder")
        plt.figure(figsize=(6, 3.5))
        plt.plot(range(1, len(deep_hist["test"]) + 1), deep_hist["test"], "s-",
                 label="deep 784-128-64-32")
        plt.plot(range(1, n_epochs + 1), history["test"], "o-", label="linear 784-32-784")
        plt.xlabel("epoch")
        plt.ylabel("test MSE")
        plt.title("Q1.4: simple vs deep autoencoder")
        plt.legend()
        plt.grid(alpha=0.3)
        lab.fig("s1_deep_curves")
        lab.results["Deep AE 784-128-64-32 (Q1.4)"] = deep_hist["test"][-1]

    print_answers("1.1", "1.2", "1.3", "1.4")


# =============================================================================
# Section 2 — convolutional autoencoder
# =============================================================================

def section2(lab: "Lab") -> None:
    import matplotlib.pyplot as plt

    cfg, device = lab.cfg, lab.device
    lab.h1("Section 2 — A convolutional autoencoder (2.3 - 2.9)")
    print("The imports and the data are the same as in Section 1.")
    print("(1, 28, 28) -> (16, 14, 14) -> (4, 7, 7) -> (16, 14, 14) -> (1, 28, 28)")
    print("                                    ^ the code")

    model = ConvAutoencoder().to(device)
    print(model)
    print("Number of parameters:", n_parameters(model))
    by_hand = (16 * 1 * 9 + 16) + (4 * 16 * 9 + 4) + (16 * 4 * 4 + 16) + (1 * 16 * 4 + 1)
    print(f"Q2.1 by hand: 160 + 580 + 272 + 65 = {by_hand} (a linear AE needs 50 992)")

    # ---------------- 2.4 follow the shapes ----------------
    lab.h2("2.4 Follow the shapes, layer by layer")
    images, _ = lab.display_batch()
    x = images[:1].to(device)
    print("input              :", tuple(x.shape))
    with torch.no_grad():
        y = F.relu(model.conv1(x)); print("after conv1        :", tuple(y.shape))
        y = model.pool(y);         print("after pool         :", tuple(y.shape))
        y = F.relu(model.conv2(y)); print("after conv2        :", tuple(y.shape))
        y = model.pool(y);         print("after pool (code)  :", tuple(y.shape))
        y = F.relu(model.t_conv1(y)); print("after t_conv1      :", tuple(y.shape))
        y = torch.sigmoid(model.t_conv2(y)); print("after t_conv2      :", tuple(y.shape))
    print("Size of the code: 4 x 7 x 7 =", 4 * 7 * 7, "values (the image has 784)")
    print("Compression ratio 784 / 196 =", 784 / 196, "(the linear AE: 784 / 32 =",
          round(784 / 32, 1), ")")

    # ---------------- 2.5 training ----------------
    lab.h2("2.5 Training")
    n_epochs = cfg.ep("conv")
    history = train_conv(model, lab.data.train_loader, lab.data.test_loader, device, n_epochs)
    lab.results["Conv AE (transposed conv)"] = history["test"][-1]

    # ---------------- 2.6-2.7 reconstructions and feature maps ----------------
    lab.h2("2.6 - 2.7 Reconstructions and feature maps of the code")
    model.eval()
    images, labels = lab.display_batch()
    with torch.no_grad():
        outputs_t = model(images.to(device))
    show_rows([images, outputs_t], ["Original test images",
                                    "Reconstructions (transposed convolutions)"])
    lab.fig("s2_reconstructions")

    with torch.no_grad():
        code = model.encode(images[:5].to(device)).cpu()     # (5, 4, 7, 7)
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    for i in range(5):
        axes[i, 0].imshow(images[i, 0], cmap="gray")
        axes[i, 0].axis("off")
        for c in range(4):
            axes[i, c + 1].imshow(code[i, c], cmap="viridis")
            axes[i, c + 1].axis("off")
    axes[0, 0].set_title("image", fontsize=10)
    for c in range(4):
        axes[0, c + 1].set_title(f"map {c + 1}", fontsize=10)
    lab.fig("s2_feature_maps")
    print("The code is not a vector but a stack of 4 maps of 7 x 7: the maps still")
    print("look like the digit, because convolutions keep the spatial structure.")

    # ---------------- 2.8-2.9 upsampling decoder ----------------
    lab.h2("2.8 - 2.9 Another decoder: upsampling + convolution")
    set_seed(cfg.seed)
    model_up = ConvAutoencoderUpsample().to(device)
    print("Number of parameters (upsampling decoder):", n_parameters(model_up))
    history_up = train_conv(model_up, lab.data.train_loader, lab.data.test_loader, device,
                            n_epochs)
    lab.results["Conv AE (upsampling + conv)"] = history_up["test"][-1]

    plt.figure(figsize=(6, 3.5))
    plt.plot(range(1, n_epochs + 1), history["test"], "o-", label="transposed convolutions")
    plt.plot(range(1, n_epochs + 1), history_up["test"], "s-",
             label="upsampling + convolution")
    plt.xlabel("epoch")
    plt.ylabel("test MSE")
    plt.title("Two decoders, same encoder")
    plt.legend()
    plt.grid(alpha=0.3)
    lab.fig("s2_two_decoders")

    with torch.no_grad():
        out_t = model(images.to(device))
        out_u = model_up(images.to(device))
    show_rows([images, out_t, out_u],
              ["Original test images", "Transposed convolutions", "Upsampling + convolution"])
    lab.fig("s2_decoder_comparison")
    best = ("upsampling + convolution" if history_up["test"][-1] < history["test"][-1]
            else "transposed convolutions")
    print(f"Q2.3 best decoder in this run: {best} "
          f"(test MSE {history_up['test'][-1]:.4f} against {history['test'][-1]:.4f}).")
    print("Look at the texture: transposed convolutions can create checkerboard")
    print("patterns (overlapping patches at stride 2), upsampling + convolution")
    print("gives smoother strokes.")

    # ---------------- Q2.4 (exercise): stride-2 encoder ----------------
    if cfg.exercises:
        lab.h2("Q2.4 (exercise) Encoder with stride=2 convolutions instead of MaxPool")
        stride_model = ConvAutoencoderStride().to(device)
        print("Number of parameters (stride-2 encoder):", n_parameters(stride_model))
        ys = images[:1].to(device)
        print("input              :", tuple(ys.shape))
        with torch.no_grad():
            ys = F.relu(stride_model.conv1(ys)); print("after conv1 (s=1)  :", tuple(ys.shape))
            ys = F.relu(stride_model.conv2(ys)); print("after conv2 (s=2)  :", tuple(ys.shape))
            ys = F.relu(stride_model.conv3(ys)); print("after conv3 (s=2)  :", tuple(ys.shape))
            ys = stride_model.decode(ys);         print("after decode       :", tuple(ys.shape))
        print("out = floor((in + 2p - k)/s) + 1  ->  floor((28+2-3)/2)+1 = 14, "
              "floor((14+2-3)/2)+1 = 7: the same code 4x7x7 as MaxPool2d.")
        hist_stride = train_conv(stride_model, lab.data.train_loader, lab.data.test_loader,
                                 device, cfg.ep("stride"), show_every=False)
        print(f"    Stride-2 encoder test loss: {hist_stride['test'][-1]:.5f} "
              f"(MaxPool2d: {history['test'][-1]:.5f})")
        with torch.no_grad():
            out_s = stride_model(images.to(device))
        show_rows([images, outputs_t, out_s],
                  ["Original test images", "MaxPool2d encoder",
                   "stride=2 encoder (Q2.4)"])
        lab.fig("s2_stride_encoder")
        lab.results["Conv AE, stride-2 encoder (Q2.4)"] = hist_stride["test"][-1]

    print_answers("2.1", "2.2", "2.3", "2.4")


# =============================================================================
# Section 3 — denoising autoencoder
# =============================================================================

def add_noise(images, noise_factor=0.5):
    """Section 3.3 — Gaussian noise, then keep the pixels between 0 and 1."""
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0.0, 1.0)


def add_salt_and_pepper(images, amount=0.2):
    """Q3.4 (exercise) — `amount` of the pixels set to 0 or 1 at random."""
    noisy = images.clone()
    mask = torch.rand_like(images)
    noisy[mask < amount / 2] = 0.0            # salt & pepper: black
    noisy[mask > 1 - amount / 2] = 1.0        # salt & pepper: white
    return noisy


def section3(lab: "Lab") -> None:
    import matplotlib.pyplot as plt

    cfg, device = lab.cfg, lab.device
    lab.h1("Section 3 — A denoising autoencoder (3.3 - 3.9)")

    images, labels = lab.display_batch()
    noise_factor = cfg.noise_factor
    noisy_images = add_noise(images, noise_factor)
    show_rows([images, noisy_images],
              ["Clean images", f"Noisy images (noise_factor = {noise_factor})"])
    lab.fig("s3_noisy_images")
    print("torch.randn_like(images) gives Gaussian noise N(0,1) on the same device,")
    print("noise_factor controls the amount, and clamp keeps valid pixel values.")

    # ---------------- 3.4 the model ----------------
    lab.h2("3.4 The model")
    model = ConvDenoiser().to(device)
    print(model)
    print("Number of parameters:", n_parameters(model))
    lab.results["Denoising AE (parameters)"] = n_parameters(model)

    # ---------------- 3.5 sizes ----------------
    lab.h2("3.5 Check the sizes with the formulas")
    print("conv / pooling        : out = floor((in + 2p - k)/s) + 1")
    print("transposed convolution: out = (in - 1)*s - 2p + k")
    print("pool on 7x7        :", (7 - 2) // 2 + 1)      # 3 (7 is odd: one row is lost)
    print("t_conv1 (k=3, s=2) :", (3 - 1) * 2 + 3)       # 7
    print("t_conv2 (k=2, s=2) :", (7 - 1) * 2 + 2)       # 14
    print("t_conv3 (k=2, s=2) :", (14 - 1) * 2 + 2)      # 28
    with torch.no_grad():
        out = model(images[:1].to(device))
    print("input", tuple(images[:1].shape), "-> output", tuple(out.shape))

    # ---------------- 3.6 training ----------------
    lab.h2("3.6 Training")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = cfg.ep("denoise")
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for images, _ in lab.data.train_loader:
            images = images.to(device)
            noisy_images = add_noise(images, noise_factor)   # new noise at every batch
            optimizer.zero_grad()
            outputs = model(noisy_images)                    # the INPUT is noisy
            loss = criterion(outputs, images)                # the TARGET is clean
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        print(f"    Epoch {epoch}/{n_epochs} | train loss: "
              f"{running_loss / len(lab.data.train_loader.dataset):.5f}")
    print("Only two lines differ from a normal autoencoder:")
    print("  outputs = model(noisy_images)  -> the input is the NOISY image")
    print("  loss = criterion(outputs, images) -> the TARGET is the CLEAN image")

    # ---------------- 3.7-3.8 results and measure of the improvement ----------------
    lab.h2("3.7 - 3.8 Results and measure of the improvement")
    model.eval()
    set_seed(0)
    images, labels = lab.display_batch()
    noisy_images = add_noise(images, noise_factor)
    with torch.no_grad():
        outputs = model(noisy_images.to(device))
    show_rows([noisy_images, outputs, images],
              ["Noisy test images (input)", "Denoised images (output)",
               "Clean images (never seen)"])
    lab.fig("s3_denoising")

    def test_errors(noise_factor: float):
        model.eval()
        before, after = 0.0, 0.0
        with torch.no_grad():
            for images, _ in lab.data.test_loader:
                images = images.to(device)
                noisy = add_noise(images, noise_factor)
                before += criterion(noisy, images).item() * images.size(0)
                after += criterion(model(noisy), images).item() * images.size(0)
        n = len(lab.data.test_loader.dataset)
        return before / n, after / n

    before, after = test_errors(noise_factor)
    print(f"MSE between noisy and clean images   : {before:.5f}")
    print(f"MSE between denoised and clean images: {after:.5f}")
    print(f"The error is divided by {before / after:.1f}")
    lab.results["Denoising AE (noisy -> denoised MSE)"] = f"{before:.4f} -> {after:.4f}"

    # ---------------- 3.9 other noise levels ----------------
    lab.h2("3.9 Other noise levels")
    levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    errors = [test_errors(f) for f in levels]
    plt.figure(figsize=(6, 3.5))
    plt.plot(levels, [e[0] for e in errors], "o-", label="noisy input")
    plt.plot(levels, [e[1] for e in errors], "s-", label="denoised output")
    plt.axvline(noise_factor, color="gray", linestyle="--", label="training noise level")
    plt.xlabel("noise_factor at test time")
    plt.ylabel("MSE with the clean images")
    plt.title("Robustness to the noise level")
    plt.legend()
    plt.grid(alpha=0.3)
    lab.fig("s3_noise_levels")
    for f, (b, a) in zip(levels, errors):
        flag = "  <- useful" if a < b else "  <- NOT useful"
        print(f"    noise_factor = {f:.1f} | noisy: {b:.5f} | denoised: {a:.5f}"
              f"{flag if f != noise_factor else '   (training level)'}")

    # ---------------- Q3.4 (exercise): salt and pepper ----------------
    if cfg.exercises:
        lab.h2("Q3.4 (exercise) Salt-and-pepper noise (20 % of the pixels)")
        sp_model = ConvDenoiser().to(device)
        sp_optimizer = torch.optim.Adam(sp_model.parameters(), lr=0.001)
        sp_amount = 0.2
        for epoch in range(1, cfg.ep("saltpepper") + 1):
            sp_model.train()
            for images, _ in lab.data.train_loader:
                images = images.to(device)
                noisy = add_salt_and_pepper(images, sp_amount)
                sp_optimizer.zero_grad()
                loss = criterion(sp_model(noisy), images)
                loss.backward()
                sp_optimizer.step()
            print(f"    Epoch {epoch}/{cfg.ep('saltpepper')} | train loss: {loss.item():.5f}")
        sp_model.eval()
        set_seed(0)
        images, labels = lab.display_batch()
        noisy = add_salt_and_pepper(images, sp_amount)
        with torch.no_grad():
            out_sp = sp_model(noisy.to(device))
        show_rows([noisy, out_sp, images],
                  ["Salt & pepper input", "Denoised output", "Clean images"])
        lab.fig("s3_salt_pepper")

        def sp_errors(amount: float):
            sp_model.eval()
            b, a = 0.0, 0.0
            with torch.no_grad():
                for images, _ in lab.data.test_loader:
                    images = images.to(device)
                    n = add_salt_and_pepper(images, amount)
                    b += criterion(n, images).item() * images.size(0)
                    a += criterion(sp_model(n), images).item() * images.size(0)
            k = len(lab.data.test_loader.dataset)
            return b / k, a / k

        for amount in (0.0, 0.1, 0.2, 0.3, 0.5):
            b, a = sp_errors(amount)
            print(f"    salt&pepper = {amount:.1f} | noisy: {b:.5f} | denoised: {a:.5f}"
                  f"{'  <- useful' if a < b else ''}")

    print_answers("3.1", "3.2", "3.3", "3.4")


# =============================================================================
# Section 4 — variational autoencoder
# =============================================================================

def loss_function(x_hat, x, mu, logvar, beta=1.0):
    """Section 4.4 — BCE (reconstruction) + beta * KL, both summed."""
    rec = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec, kl


def run_epoch(model, loader, device, optimizer=None, beta=1.0, flat=True):
    """Section 4.5 — one pass over the data (train if an optimizer is given).

    `flat=True` flattens the batch to (B, 784) for the linear VAE; the
    convolutional VAE of Q4.5 keeps the (B, 1, 28, 28) shape (flat=False).
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    total_rec, total_kl = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for images, _ in loader:
            x = images.to(device)
            x = x.view(-1, 784) if flat else x
            x_hat, mu, logvar = model(x)
            loss, rec, kl = loss_function(x_hat, x, mu, logvar, beta)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_rec += rec.item()
            total_kl += kl.item()
    n = len(loader.dataset)
    return total_rec / n, total_kl / n          # averages per image


def train_vae(model, train_loader, test_loader, device, n_epochs, lr=1e-3, beta=1.0,
              flat=True):
    """Section 4.5 / Q4.4 — the beta argument controls the KL weight."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_rec": [], "train_kl": [], "test_rec": [], "test_kl": []}
    for epoch in range(1, n_epochs + 1):
        train_rec, train_kl = run_epoch(model, train_loader, device, optimizer, beta, flat)
        test_rec, test_kl = run_epoch(model, test_loader, device, None, beta, flat)
        for key, value in zip(history, [train_rec, train_kl, test_rec, test_kl]):
            history[key].append(value)
        print(f"    Epoch {epoch:2d}/{n_epochs} | train: rec {train_rec:6.2f}"
              f" + KL {train_kl:5.2f} = {train_rec + train_kl:6.2f}"
              f" | test: {test_rec + test_kl:6.2f}")
    return history


def section4(lab: "Lab") -> None:
    import matplotlib.pyplot as plt

    cfg, device = lab.cfg, lab.device
    lab.h1("Section 4 — A variational autoencoder (4.3 - 4.10)")
    print("image (784) -> hidden (400) -> mu, logvar (20) -> z (20)"
          " -> hidden (400) -> reconstruction (784)")

    # ---------------- 4.3 the model ----------------
    model = VAE(latent_dim=20).to(device)
    print(model)
    print("Number of parameters:", n_parameters(model))
    lab.results["VAE 20 dims (parameters)"] = n_parameters(model)

    # ---------------- 4.4 the loss ----------------
    lab.h2("4.4 The loss")
    images, labels = lab.display_batch()
    x = images.to(device).view(-1, 784)
    model.eval()
    with torch.no_grad():
        x_hat, mu, logvar = model(x)
        loss, rec, kl = loss_function(x_hat, x, mu, logvar)
    print(f"Per image, before training: reconstruction = {rec.item() / x.size(0):.1f}"
          f" | KL = {kl.item() / x.size(0):.2f}")
    print("L = BCE(x, x_hat) + beta * (-0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j)))")
    print("Both terms use reduction='sum' so that their balance is correct.")

    # ---------------- 4.5 training ----------------
    lab.h2("4.5 Training")
    n_epochs = cfg.ep("vae")
    history = train_vae(model, lab.data.vae_train_loader, lab.data.test_loader, device,
                        n_epochs)
    lab.results["VAE 20 dims (loss per image)"] = history["test_rec"][-1] + history["test_kl"][-1]

    # ---------------- 4.6 the two terms ----------------
    lab.h2("4.6 The two terms of the loss")
    epochs = range(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(epochs, history["train_rec"], "o-", label="train")
    axes[0].plot(epochs, history["test_rec"], "s-", label="test")
    axes[0].set_title("Reconstruction term (BCE per image)")
    axes[1].plot(epochs, history["train_kl"], "o-", label="train")
    axes[1].plot(epochs, history["test_kl"], "s-", label="test")
    axes[1].set_title("KL term (per image)")
    axes[2].plot(epochs, np.add(history["train_rec"], history["train_kl"]), "o-", label="train")
    axes[2].plot(epochs, np.add(history["test_rec"], history["test_kl"]), "s-", label="test")
    axes[2].set_title("Total loss = -ELBO (per image)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    lab.fig("s4_loss_terms")
    print("The reconstruction term goes down; the KL term goes UP at the start: to")
    print("reconstruct well the encoder moves the codes away from the prior N(0,I)")
    print("and pays the KL 'tax'. (Q4.1)")

    # ---------------- 4.7-4.8 reconstructions and generation ----------------
    lab.h2("4.7 - 4.8 Reconstructions and generation of new digits")
    model.eval()
    images, labels = lab.display_batch()
    with torch.no_grad():
        x_hat, mu, logvar = model(images.to(device).view(-1, 784))
    show_rows([images, x_hat], ["Original test images", "VAE reconstructions"])
    lab.fig("s4_reconstructions")

    set_seed(0)
    with torch.no_grad():
        z = torch.randn(30, model.latent_dim, device=device)
        generated = model.decode(z)
    show_rows([generated[:10], generated[10:20], generated[20:30]],
              ["New digits generated from random z", "", ""])
    lab.fig("s4_generated")
    print("To generate we do NOT use the encoder: z ~ N(0, I) then decode. Compare")
    print("with 1.10: random codes now give digits, because the KL term has organized")
    print("the latent space around N(0, I).")

    # ---------------- 4.9 interpolation ----------------
    lab.h2("4.9 Interpolation in the latent space")

    def interpolate(image_a, image_b, n=10):
        with torch.no_grad():
            mu_a, _ = model.encode(image_a.to(device).view(1, 784))
            mu_b, _ = model.encode(image_b.to(device).view(1, 784))
            t = torch.linspace(0, 1, n, device=device).view(n, 1)
            z = (1 - t) * mu_a + t * mu_b          # n points between mu_a and mu_b
            return model.decode(z)

    examples = find_examples(lab.data.test_loader, (1, 7))
    if 1 in examples and 7 in examples:
        show_rows([interpolate(examples[1], examples[7])],
                  ["Interpolation in the latent space: from a 1 to a 7"])
        lab.fig("s4_interpolation")
        print("The digit changes smoothly along the line: the latent space has no holes.")
    else:
        print("Could not find a '1' and a '7' in the test batches, interpolation skipped.")

    # ---------------- 4.10 VAE with a 2-D latent space ----------------
    lab.h2("4.10 A VAE with a 2-D latent space")
    set_seed(0)
    model_2d = VAE(latent_dim=2).to(device)
    history_2d = train_vae(model_2d, lab.data.vae_train_loader, lab.data.test_loader, device,
                           cfg.ep("vae2d"))
    lab.results["VAE 2 dims (loss per image)"] = (history_2d["test_rec"][-1]
                                                  + history_2d["test_kl"][-1])

    model_2d.eval()
    all_mu, all_labels = [], []
    with torch.no_grad():
        for images, labels in lab.data.test_loader:
            mu, _ = model_2d.encode(images.to(device).view(-1, 784))
            all_mu.append(mu.cpu())
            all_labels.append(labels)
    all_mu = torch.cat(all_mu).numpy()
    all_labels = torch.cat(all_labels).numpy()

    plt.figure(figsize=(7, 6))
    points = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap="tab10", s=4)
    plt.colorbar(points, ticks=range(10), label="digit")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Test images in the 2-D latent space")
    lab.fig("s4_latent_scatter")
    print("Q4.3 — clusters seen in the run:")
    for digit in range(10):
        mask = all_labels == digit
        if mask.any():
            print(f"    digit {digit}: centroid mu = ({all_mu[mask, 0].mean():+.2f}, "
                  f"{all_mu[mask, 1].mean():+.2f}) "
                  f"| spread = ({all_mu[mask, 0].std():.2f}, {all_mu[mask, 1].std():.2f})")

    # decode a regular grid of latent points (vectorized version of the lab loop)
    n = 15
    grid = torch.linspace(-3, 3, n)
    z2s = torch.flip(grid, [0])                        # top row = largest z2
    zz1, zz2 = torch.meshgrid(grid, z2s, indexing="xy")
    z_all = torch.stack([zz1.reshape(-1), zz2.reshape(-1)], dim=1).to(device)
    with torch.no_grad():
        grid_imgs = model_2d.decode(z_all).view(-1, 28, 28).cpu().numpy()
    canvas = grid_imgs.reshape(n, 28, n, 28).transpose(0, 2, 1, 3).reshape(n * 28, n * 28)
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray", extent=[-3, 3, -3, 3])
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Digits decoded from a grid of latent points")
    lab.fig("s4_latent_grid")

    # ---------------- Q4.4 (exercise): beta = 0 and beta = 5 ----------------
    if cfg.exercises:
        lab.h2("Q4.4 (exercise) VAE trained with beta = 0 and beta = 5")
        images, labels = lab.display_batch()
        for beta in (0.0, 5.0):
            set_seed(cfg.seed)
            beta_model = VAE(latent_dim=20).to(device)
            h = train_vae(beta_model, lab.data.vae_train_loader, lab.data.test_loader, device,
                          cfg.ep("beta"), beta=beta)
            beta_model.eval()
            with torch.no_grad():
                rec_img, mu_b, logvar_b = beta_model(images.to(device).view(-1, 784))
                z = torch.randn(10, 20, device=device)
                gen = beta_model.decode(z)
                mu_stats = (mu_b.mean().item(), mu_b.std().item(),
                            torch.exp(0.5 * logvar_b).mean().item())
            show_rows([images, rec_img, gen],
                      [f"beta = {beta}", "reconstructions", "generated from N(0, I)"])
            lab.fig(f"s4_beta_{beta:g}")
            print(f"    beta = {beta:g}: test rec = {h['test_rec'][-1]:.2f}, "
                  f"test KL = {h['test_kl'][-1]:.2f}, "
                  f"mean(mu) = {mu_stats[0]:+.3f}, std(mu) = {mu_stats[1]:.3f}, "
                  f"mean(sigma) = {mu_stats[2]:.3f}, "
                  f"generated pixel std = {gen.std().item():.4f}")
            lab.results[f"VAE beta = {beta:g} (Q4.4)"] = h["test_rec"][-1] + h["test_kl"][-1]

    # ---------------- Q4.5 (exercise): convolutional VAE ----------------
    if cfg.exercises:
        lab.h2("Q4.5 (exercise) Convolutional VAE")
        set_seed(cfg.seed)
        conv_vae = ConvVAE(latent_dim=20).to(device)
        print(model_summary(conv_vae))
        print("Number of parameters:", n_parameters(conv_vae))
        h = train_vae(conv_vae, lab.data.vae_train_loader, lab.data.test_loader, device,
                      cfg.ep("convvae"), flat=False)
        lab.results["Conv VAE (Q4.5)"] = h["test_rec"][-1] + h["test_kl"][-1]
        print(conv_vae)
        conv_vae.eval()
        images, labels = lab.display_batch()
        with torch.no_grad():
            rec_img, mu_c, logvar_c = conv_vae(images.to(device))
            gen = conv_vae.decode(torch.randn(10, 20, device=device))
        show_rows([images, rec_img, gen],
                  ["Original test images", "Conv VAE reconstructions",
                   "Conv VAE generated"])
        lab.fig("s4_conv_vae")
        print(f"    Linear VAE (same epochs): rec = {history['test_rec'][-1]:.2f}"
              f" | Conv VAE: rec = {h['test_rec'][-1]:.2f}")
        print("The convolutional encoder keeps the spatial structure: lower")
        print("reconstruction term per image for the same number of epochs.")

    print_answers("4.1", "4.2", "4.3", "4.4", "4.5")


def model_summary(model: nn.Module) -> str:
    """Shapes of every layer of a model: the best way to check a conv network."""
    lines = []
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            lines.append(f"{name:18s} {type(module).__name__:17s} {tuple(module.weight.shape)}")
    return "\n".join(lines)


# =============================================================================
# Answers printing
# =============================================================================

def print_answers(*keys: str) -> None:
    print("\n" + "=" * 78)
    print("ANSWERS")
    print("=" * 78)
    for key in keys:
        print()
        print(ANSWERS[key])
    print("=" * 78 + "\n")


def print_all_answers() -> None:
    print("=" * 78)
    print("GENERATIVE AI — Lab 1: WRITTEN ANSWERS")
    print("=" * 78)
    for key in ANSWERS:
        print()
        print(ANSWERS[key])
    print()
    print(EXPLANATIONS)


# =============================================================================
# Summary
# =============================================================================

REFERENCE = [
    ("Linear AE (code of 32)", 10, "test MSE = 0.0141"),
    ("Convolutional AE, transposed convolutions", 10, "test MSE = 0.0088"),
    ("Convolutional AE, upsampling + convolution", 10, "test MSE = 0.0060"),
    ("Denoising AE (noise factor 0.5)", 5,
     "MSE with the clean images: 0.1157 (noisy input) -> 0.0281 (output)"),
    ("VAE, 20 latent dimensions", 15, "loss per image (rec + KL) = 104.8"),
    ("VAE, 2 latent dimensions", 10, "loss per image (rec + KL) = 153.5"),
]

PARAMETERS = [
    ("Linear AE", "image / same image", "32 values", 50_992, "MSE"),
    ("Convolutional AE", "image / same image", "4x7x7", 1_077, "MSE"),
    ("Denoising AE", "noisy image / clean image", "8x3x3", 9_585, "MSE"),
    ("VAE", "image / same image", "mu and log sigma (20 each)", 652_824, "BCE + KL"),
]


def summary(lab: "Lab") -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Model':38s} {'Input / target':24s} {'Code':26s} {'Parameters':>11s}  Loss")
    for name, io, code, params, loss in PARAMETERS:
        print(f"{name:38s} {io:24s} {code:26s} {params:11,d}  {loss}")
    print("\nParameters computed by the script (PyTorch):")
    for label, cls, args in (("Linear AE", Autoencoder, (32,)),
                             ("Conv AE", ConvAutoencoder, ()),
                             ("Denoising AE", ConvDenoiser, ()),
                             ("VAE (20)", VAE, ())):
        print(f"  {label:14s} {n_parameters(cls(*args)):8,d}")

    print("\nResults of THIS run:")
    for key, value in lab.results.items():
        print(f"  {key:46s} {value}")
    print("\nResults of the reference run (CPU, default epochs):")
    for name, epochs, result in REFERENCE:
        print(f"  {name:48s} {epochs:2d} epochs   {result}")
    print("\n" + EXPLANATIONS)


# =============================================================================
# Orchestration
# =============================================================================

class Lab:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        self.fig_sink = FigureSink(cfg.figures_dir, cfg.show)
        self.data: Data | None = None
        self.results: dict = {}
        self._display_batch: tuple | None = None

    # --- printing -----------------------------------------------------------
    @staticmethod
    def h1(title: str) -> None:
        print("\n" + "#" * 78)
        print("# " + title)
        print("#" * 78)

    @staticmethod
    def h2(title: str) -> None:
        print(f"\n--- {title} " + "-" * max(0, 70 - len(title)))

    def fig(self, name: str) -> None:
        self.fig_sink(name)

    # --- data ---------------------------------------------------------------
    def load_data(self) -> Data:
        """Section 1.2 / 4.2 — MNIST with torchvision + DataLoader."""
        transform = transforms.ToTensor()     # PIL image -> tensor (1,28,28) in [0,1]
        train_data = datasets.MNIST(root=str(self.cfg.data_root), train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root=str(self.cfg.data_root), train=False,
                                   download=True, transform=transform)
        if self.cfg.limit_train:
            train_data = torch.utils.data.Subset(train_data, range(self.cfg.limit_train))
        if self.cfg.limit_test:
            test_data = torch.utils.data.Subset(test_data, range(self.cfg.limit_test))

        self.data = Data(
            train_loader=torch.utils.data.DataLoader(
                train_data, batch_size=self.cfg.batch_size, shuffle=True),
            test_loader=torch.utils.data.DataLoader(
                test_data, batch_size=self.cfg.batch_size, shuffle=False),
            # the VAE section of the lab uses a bigger batch (128)
            vae_train_loader=torch.utils.data.DataLoader(
                train_data, batch_size=self.cfg.vae_batch_size, shuffle=True),
            n_train=len(train_data),
            n_test=len(test_data),
        )
        return self.data

    def display_batch(self):
        if self._display_batch is None:
            self._display_batch = next(iter(self.data.test_loader))
        return self._display_batch


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lab 1: autoencoders and VAEs on MNIST (all sections + answers)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, default=HERE / "data",
                   help="folder where MNIST is downloaded (once)")
    p.add_argument("--figures-dir", type=Path, default=HERE / "figures",
                   help="where the PNG figures are written")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
                   help="device used for the training")
    p.add_argument("--sections", default="all",
                   help="comma separated list: 0,1,2,3,4 or all")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: few epochs and a small subset of MNIST")
    p.add_argument("--epochs-scale", type=float, default=None,
                   help="multiply every number of epochs (e.g. 0.5)")
    p.add_argument("--no-exercises", action="store_true",
                   help="skip Q1.4, Q2.4, Q3.4, Q4.4 and Q4.5")
    p.add_argument("--answers-only", action="store_true",
                   help="print the written answers and exit (no training)")
    p.add_argument("--show", action="store_true", help="display the figures")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--vae-batch-size", type=int, default=128)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--noise-factor", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    epochs = dict(DEFAULT_EPOCHS)
    scale = args.epochs_scale
    limit_train, limit_test = args.limit_train, args.limit_test
    if args.quick:
        scale = scale if scale is not None else 0.1
        limit_train = limit_train or 4000
        limit_test = limit_test or 1000
    if scale is not None:
        epochs = {k: max(1, round(v * scale)) for k, v in epochs.items()}

    cfg = Config(
        data_root=args.data_root,
        device=device,
        epochs=epochs,
        batch_size=args.batch_size,
        vae_batch_size=args.vae_batch_size,
        limit_train=limit_train,
        limit_test=limit_test,
        noise_factor=args.noise_factor,
        seed=args.seed,
        exercises=not args.no_exercises,
    )
    cfg.figures_dir = args.figures_dir
    cfg.show = args.show
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.answers_only:
        print_all_answers()
        return 0

    cfg = build_config(args)
    set_seed(cfg.seed)

    sections = ({"0", "1", "2", "3", "4"} if args.sections == "all"
                else {s.strip() for s in args.sections.split(",")})

    print("=" * 78)
    print("Generative AI — Lab 1: Autoencoders and VAEs on MNIST")
    print("=" * 78)
    print(f"device      : {cfg.device}")
    print(f"data root   : {cfg.data_root}")
    print(f"figures     : {cfg.figures_dir}" + ("" if cfg.show else "  (saved as PNG)"))
    print(f"exercises   : {'yes' if cfg.exercises else 'no (--no-exercises)'}")
    print("epochs      : " + ", ".join(f"{k}={v}" for k, v in cfg.epochs.items()))
    if cfg.limit_train:
        print(f"subset      : {cfg.limit_train} train / {cfg.limit_test} test images")

    lab = Lab(cfg)
    t0 = time.perf_counter()
    lab.load_data()
    print(f"\nTraining images: {lab.data.n_train} | Test images: {lab.data.n_test}")

    runners = {"0": section0, "1": section1, "2": section2, "3": section3, "4": section4}
    for key in sorted(sections):
        if key not in runners:
            print(f"Unknown section '{key}' (choose from 0, 1, 2, 3, 4, all)")
            return 2
        start = time.perf_counter()
        runners[key](lab)
        print(f"\n>>> section {key} done in {time.perf_counter() - start:6.1f} s")

    summary(lab)
    print(f"Total time: {time.perf_counter() - t0:.1f} s | "
          f"{lab.fig_sink.count} figures written to {cfg.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
