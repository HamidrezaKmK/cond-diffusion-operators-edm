import torch

class NoisingKernel(object):
    def sample(self, N):
        raise NotImplementedError()

class RBFIrregular(NoisingKernel):
    """
    An irregular noising kernel that samples from a Gaussian process with a RBF kernel.
    """
    @torch.no_grad()
    def __init__(
        self, 
        scale: float = 0.05,
        eps: float = 0.01,
    ):
        self.scale = scale
        self.eps = eps
        self._last_coords = None
        self._L = None

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,  # (batch_size, dim, K)
    ):
        coords = coords.permute(0, 2, 1)
        batch_size, K, dim = coords.shape
        # if the coordinates are the same as before, no need to recompute the cholesky
        if self._last_coords is not None and torch.allclose(self._last_coords, coords - coords[0, 0]):
            L = self._L
        else:
            self._last_coords = coords - coords[0, 0]
            C = torch.exp(-torch.cdist(coords, coords, p=2) / (2 * self.scale**2))
            I = torch.eye(K).to(device=coords.device) 
            I.mul_(self.eps**2)
            C.add_(I.unsqueeze(0).repeat(batch_size, 1, 1)) # make it PSD
            L = torch.linalg.cholesky(C)
            del C, I
            self._L = L
        
        base_noise = torch.randn((batch_size, K)).to(dtype=coords.dtype).to(device=coords.device)
        return torch.einsum("bij,bj->bi", L, base_noise)
