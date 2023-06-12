"""
Implementation of a Positional Encoding Function
as per Section 5.1 of the paper - NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (https://arxiv.org/pdf/2003.08934.pdf)

Includes an expansion pack of other positional encoders
"""
import torch
import matplotlib.pyplot as plt

class PositionalEncoder():
    def __init__(self):
        self.encoding_fns = self.get_encoding_fn()
    
    def get_encoding_fn(self):
        return [torch.sin, torch.cos]
    
    def encode(self, input, num_freqs):
        encodings = []
        for i in range(num_freqs):
            for fn in self.encoding_fns:
                encodings.append(fn((2 ** i) * input * torch.pi))
        return torch.cat(encodings, dim=-1)
    
    def plot_encodings(self, input, num_freqs):
        fig = plt.figure(figsize=(12, 6))
        X = torch.arange(-1, 1, 0.01)
        for j, pos in enumerate([-0.51, -0.42, -0.33, -0.24, -0.15, 0.06, 0.17, 0.28, 0.39, 0.44, 0.55]):
            pos = torch.tensor([[pos]])
            ys = []
            for i in range(num_freqs):
                Y = torch.sin(X*torch.pi*2**i)
                y = torch.sin(pos*torch.pi*2**i)
                ys.append(y)
                plt.plot(X, Y)
                plt.scatter(pos, y, s=30, c='black')
            plt.savefig(f"{j}.jpg")
            print(ys)
            plt.pause(0.5)
            plt.cla()

    def plot_points(self, input):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(input[:, 0], input[:, 1], input[:, 2])
        plt.show()


#################
### Unit Test ###
#################
if __name__ == "__main__":
    encoder = PositionalEncoder()
    dummy_input = torch.randn((1, 3), dtype=torch.float32)
    dummy_output = encoder.encode(dummy_input, 10)
    print(dummy_output.shape)
    encoder.plot_encodings(dummy_input, 4)
    # encoder.plot_points(dummy_input)
