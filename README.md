# neural radiance fields (NeRF)

![a realistic(?) future](hologram.jpg)

A spelled-out study of neural radiance fields (NeRFs). Contains a PyTorch implementation + ground-up explanation (or at least my logical interpretation) of [this paper](https://arxiv.org/pdf/2003.08934.pdf).

# positional encodings
A key contributor to the success of neural radiance fields is positional encoding. In the paper, the authors explained that positional encoding helps transform the 5-dimensional input positional data (x, y, z, $\theta$, $\phi$) to a higher frequency domain, which in turn allows a neural network to learn data with higher frequency variation, such as color and geometry.

The encoding function used is as follows
$$\gamma(p) = (sin(2^0{/pi}p))$$

