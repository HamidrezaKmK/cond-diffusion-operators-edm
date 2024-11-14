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
        accuracy: int = 20,
    ):
        self.scale = scale
        self.accuracy = accuracy
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
        if self._last_coords is not None and \
            self._last_coords.shape == coords[0].shape and \
            torch.allclose(self._last_coords, coords - coords[0, 0]):
            L = self._L
        else:
            self._last_coords = coords - coords[0, 0]
            C = torch.exp(-torch.cdist(coords, coords, p=2) / (2 * self.scale**2))
            I = torch.eye(K).to(device=coords.device) 
            
            # do an inline binary search to find the minimal jittering value to get the Cholesky to work
            l_eps = 0.0
            r_eps = 5.0
            for _ in range(self.accuracy):
                mid_eps = (l_eps + r_eps) / 2
                C_eps = C + (mid_eps ** 2 * I).repeat(batch_size, 1, 1) # make it PSD
                try:
                    L = torch.linalg.cholesky(C_eps)
                    r_eps = mid_eps
                except:
                    l_eps = mid_eps
            C_eps = C + (r_eps ** 2 * I).repeat(batch_size, 1, 1) 
            
            L = torch.linalg.cholesky(C_eps)
            del C, I, C_eps
            self._L = L
        
        base_noise = torch.randn((batch_size, K)).to(dtype=coords.dtype).to(device=coords.device)
        return torch.einsum("bij,bj->bi", L, base_noise)
