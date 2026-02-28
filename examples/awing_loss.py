# Adaptive Wing Loss for heatmap-based keypoint detection (ICCV 2019)

class AdaptiveWingLoss(nn.Module):
    """
    Near peaks (small error):  logarithmic penalty -> strict, precise localization
    Far from peaks (large error): linear penalty   -> lenient, avoids over-penalizing

    AWing(y, y_hat) = {
        w * ln(1 + |y - y_hat|^(a-y) / e)   if |y - y_hat| < theta
        A * |y - y_hat| - C                  otherwise
    }
    """
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        delta = (target - pred).abs()

        # A adapts based on target value (higher target = higher penalty)
        A = self.omega * (
            1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        )
        C = self.theta * A - self.omega * torch.log(
            torch.tensor(1.0 + self.theta / self.epsilon)
        )

        losses = torch.where(
            delta < self.theta,
            # Logarithmic: strict near peaks
            self.omega * torch.log(
                1.0 + torch.pow(delta / self.epsilon, self.alpha - target)),
            # Linear: lenient far from peaks
            A * delta - C
        )
        return losses.mean()
