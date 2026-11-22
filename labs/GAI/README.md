# GAI Lab 2: GANs and conditional GANs

What was done to [`lab2_gans.ipynb`](lab2_gans.ipynb) on 2026-11-22, what came out of it, and what to watch out for.

The code, the answers and this README were written by Claude Code (Claude Opus 5.5), which also ran the training.

## What happened

1. **Filled the notebook.** Completed the TODOs, wrote Exercises A and B, and added 5 code cells for the questions that need an experiment.
2. **Trained everything** in one top-to-bottom run on an RTX 3050 (8 GB), with the GPU epoch counts suggested by the lab. It took 27 minutes.
3. **Answered the 20 questions** from the outputs and figures of that run, in 4 new markdown cells. The original text of the lab is unchanged.
4. **Saved the final weights** in [`checkpoints/`](checkpoints).
5. **Committed on `gai/lab2-gans`**: `1ab9691` is the notebook, `5186c71` the weights.

The environment was Python 3.14.7, PyTorch 2.15.0+cu132 and torchvision 0.29.2+cu132. The CPU-only outputs previously in the notebook came from an older Python 3.12 environment and were replaced.

The cells were run by a script, in order and in one namespace, because the venv has no nbconvert. *Run All* in Jupyter has not been tested.

## Results

| Section | Model | Epochs | Time | Result |
|---|---|--:|--:|---|
| 1.5 | Linear GAN, MNIST | 30 | 3.6 min | Recognizable but speckled digits |
| 1.8 | DCGAN, MNIST | 20 | 3.4 min | Clean, sharp digits, a few malformed |
| A | DCGAN, Pokémon | 500 | 6.4 min | Good colours and silhouettes, no real anatomy |
| 2.3 | cGAN, MNIST | 20 | 3.4 min | 98.6% of generated digits have the requested class |
| Q2.4 | cGAN with one-hot label | 20 | 3.3 min | 98.0% have the requested class |
| Q2.5 | CNN classifier, used to measure the two scores above | 3 | 0.3 min | 98.76% on real test digits |
| B | cGAN, flowers | 100 | 6.3 min | Right colours per class, blurry shapes, dandelion is weakest |

## Changes beyond filling the blanks

- **Epochs** depend on the device: the GPU values on CUDA, the original values on CPU.
- **Initialization:** BatchNorm weights use N(1, 0.02), not a literal N(0, 0.02), because they are scale factors. Embeddings keep N(0, 1), the scale of the noise.
- **Memory:** the Pokémon training cell keeps only the three snapshots it displays. Until that cell ends, the samples still take about 1.5 GB of RAM.

## Findings and caveats

### Question 1.6 does not behave as the lab predicts

The lab says that without `Normalize` the discriminator wins at once. It did not, in the notebook's run or in a side check with three seeds. In that check, clipping the fakes to [0, 1] did not make them harder to detect, so there was no evidence that the discriminator relies on negative pixels. The notebook's answer calls them a "possible clue", which is true in theory but was not what we observed.

### The Pokémon discriminator wins

In the second half of training its median loss is 0.028. Memorization of the 819 images is a plausible cause, but it was not tested. Stronger augmentation is the first thing to try.

The nearest-neighbour test found no sign of copying. It compares pixels, so it would miss shifted or recoloured copies.

### The flowers dataset has duplicates

- 4 identical photos are filed as both rose and tulip. The lab's "keep each file name once" rule leaves them as roses, hence 4,313 images instead of 4,317.
- 2 sunflower photos appear twice under different file names. That rule cannot catch them, so they are still in the training set.

### Other traps

- The "Expected output" blocks of the lab assume the short CPU runs, so they do not match these outputs.
- The lab mentions a `requirements.txt`, which is not in this repo. Dependencies are in [`pyproject.toml`](../../pyproject.toml).
- Exercises A and B both use the variable `path`. After running B, re-run A.1 before re-running A.2, or A will load the flowers.
- Retraining will not give identical results: two runs with the same seed already differed after the first DCGAN epoch. Several answers (especially 1.3, 1.6, A.4 and B.2) quote this run and would need editing.

## Data

Nothing here is committed. The notebook downloads it when needed.

| Dataset | Location | Images used |
|---|---|--:|
| MNIST | `labs/GAI/data/` (gitignored) | 60,000 train, 10,000 test |
| [Pokémon](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset) | kagglehub cache in `~/.cache/kagglehub/` | 819 (the JPG versions) |
| [Flowers](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) | kagglehub cache in `~/.cache/kagglehub/` | 4,313 |

## Checkpoints

Each file is a `state_dict` from the last epoch, without optimizer state: good for generating, not for resuming training exactly.

| Files in `checkpoints/` | Notebook variables | Class defined in |
|---|---|---|
| `linear_gan_{G,D}.pth` | `G`, `D` | 1.3 |
| `dcgan_mnist_{G,D}.pth` | `G_dc`, `D_dc` | 1.7 |
| `dcgan_pokemon_{G,D}.pth` | `G_p`, `D_p` | A.2, step 3 |
| `cgan_mnist_{G,D}.pth` | `G_c`, `D_c` | 2.2 |
| `cgan_mnist_onehot_{G,D}.pth` | `G_onehot`, `D_onehot` | Q2.4 (same cell as its training) |
| `mnist_classifier.pth` | `classifier` | Q2.5 (same cell as its training) |
| `cgan_flowers_{G,D}.pth` | `G_f`, `D_f` | B.2, step 3 |

The two throwaway models of Q1.6 were not saved.

### Generate without retraining

**Pass `64` explicitly for the Pokémon and flower models.** On a CPU the notebook builds them with 32 channels, and loading these weights into them fails.

1. Run the import cell of 1.1.
2. Run `n_classes = 10` and `classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]`.
3. Run the class cells of 1.3, 1.7, A.2 step 3, 2.2 and B.2 step 3. None of them trains anything.
4. Run this, with `labs/GAI` as the working directory:

```python
def load(model, name):
    state = torch.load(f"checkpoints/{name}.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


G = load(Generator(100), "linear_gan_G")
G_dc = load(DCGenerator(100), "dcgan_mnist_G")
G_p = load(PokemonGenerator(100, 64), "dcgan_pokemon_G")
G_c = load(CondGenerator(100, 10), "cgan_mnist_G")
G_f = load(FlowerGenerator(100, 5, ngf=64), "cgan_flowers_G")

with torch.no_grad():
    z = torch.randn(8, 100, device=device)
    show_images(G_p(z), title="Pokémon")
    show_images(G_f(z, torch.full((8,), 3, device=device)), title="Sunflowers")
```

For the discriminators, use `PokemonDiscriminator(64)` and `FlowerDiscriminator(5, 64)`.

`*.pth` is gitignored and these files were force-added, so a new checkpoint needs `git add -f`.
