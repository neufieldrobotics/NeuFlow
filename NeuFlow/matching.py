import torch.nn.functional as F

from NeuFlow import utils


class Matching:

    def init_grid(self, batch_size, height, width):
        self.grid = utils.coords_grid(batch_size, height, width).cuda()  # [B, 2, H, W]
        self.flatten_grid = self.grid.view(batch_size, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    def global_correlation_softmax(self, feature0, feature1):

        b, c, h, w = feature0.shape

        feature0 = feature0.flatten(-2).permute(0, 2, 1)
        feature1 = feature1.flatten(-2).permute(0, 2, 1)

        correspondence = F.scaled_dot_product_attention(feature0, feature1, self.flatten_grid)

        correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        flow = correspondence - self.grid

        return flow
