import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityClassifier(nn.Module):
    """域判别器"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        """

        Args:
            input_dim: 768
            hidden_dim: 768 // 4
            output_dim: 1
            num_layers: 2 / 3
            BN: False
        """
        super().__init__()
        self.num_layers = num_layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """
        利用 GRL 辅助模态判别
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, output_dim)
        """
        x = self.gap(x).view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GradReverse(torch.autograd.Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class AlphaScheduler:
    """alpha 值调度器"""

    def __init__(self, mode='linear', initial_alpha=1.0, decay_epoch=10, decay_rate=0.1):
        self.mode = mode
        self.initial_alpha = initial_alpha
        self.decay_epoch = decay_epoch
        self.decay_rate = decay_rate

    def get_alpha(self, epoch):
        if self.mode == 'step':
            return self._step_schedule(epoch)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _step_schedule(self, epoch):
        if epoch < self.decay_epoch:
            return self.initial_alpha
        else:
            return self.initial_alpha * self.decay_rate


class AdaptiveGradReverse(nn.Module):
    def __init__(self, alpha_scheduler):
        super().__init__()
        self.alpha_scheduler = alpha_scheduler
        self.current_alpha = 0.0

    def forward(self, x):
        return GradReverse.apply(x, self.current_alpha)

    def update_alpha(self, epoch):
        """
        实现方案: model.grl.update_alpha(epoch)
        """
        self.current_alpha = self.alpha_scheduler.get_alpha(epoch)


def grad_reverse(x, alpha=1.0):
    """
    在反向传播时，将梯度乘以 -1 或乘以 alpha, 实现梯度的反转
    Args:
        x:
        alpha: 梯度反转系数, 默认值为 1.0, 如果越大, 更侧重与辅助任务, 主任务收敛变慢; 如果越小, 更侧重与主任务, 辅助任务收敛变慢

    Returns:

    """
    return GradReverse.apply(x, alpha)
