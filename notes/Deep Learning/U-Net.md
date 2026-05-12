<div align="center" width="100%" height="100%"> <img src="images/unet_diagram.png"> </div>

# Downsampling (Encoders)
# Upsampling (Decoder)

The decoder process at each stage is 3 steps:

1. Transpose convolution (upsampling) — doubles spatial size, halves channels
2. Concatenation with skip connection — doubles the channels again
3. Double 3×3 convolution — fuses and refines

Channel progression: 3 → 64 → 128 → 256 → 512 → 1024 (bottleneck) → back up.

Final layer: 1×1 convolution → one output channel per class.

## 2. Approaches
### 2.1 Hourglass approach
a basic encoder-decoder compresses the image to extract features, then reconstructs it — but loses fine spatial details in the process.

### 2.2 Skip Connections (U-Net)
direct links between encoder and decoder layers of the same resolution. The decoder gets both the upsampled features AND the original fine details from the encoder. Skip connections add no parameters.

<div align="center" width="100%" height="100%"> <img src="images/unet_TL.png"> </div>