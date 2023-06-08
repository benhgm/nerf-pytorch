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
        encodings = [input]
        for i in range(num_freqs):
            for fn in self.encoding_fns:
                encodings.append(fn((2 ** i) * input * torch.pi))
        return torch.cat(encodings, dim=-1)
    
    def plot_encodings(self, input, num_freqs):
        pass


        
            

    
#################
### Unit Test ###
#################
if __name__ == "__main__":
    encoder = PositionalEncoder()
    dummy_input = torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float32)
    dummy_output = encoder.encode(dummy_input, 10)
    
    encoder.plot_encodings(dummy_input, 4)
