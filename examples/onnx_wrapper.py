# Bakes post-processing into the ONNX graph -- browser gets corners directly

class BlazeDocONNXWrapper(nn.Module):
    """
    Wraps heatmap model with argmax + coordinate extraction.
    Browser receives corners + confidence, no post-processing needed.
    """
    def __init__(self, blazedoc_model, heatmap_size=104, image_size=416):
        super().__init__()
        self.model = blazedoc_model
        self.scale = float(image_size) / float(heatmap_size)  # 4.0

    def forward(self, image):
        heatmaps = self.model(image)           # (B, 4, 104, 104)
        B, K, H, W = heatmaps.shape

        heatmaps_flat = heatmaps.view(B, K, -1)
        max_idx = torch.argmax(heatmaps_flat, dim=2)

        # Confidence = peak heatmap value
        scores = torch.gather(heatmaps_flat, 2, max_idx.unsqueeze(2)).squeeze(2)

        # Convert flat index -> (x, y) coordinates
        y = (max_idx // W).float()
        x = (max_idx % W).float()
        corners = torch.stack([x, y], dim=-1) * self.scale

        return corners, scores  # (B,4,2), (B,4)


# Export: post-processing is now part of the ONNX graph
model = BlazeDocDetector()
model.load_state_dict(torch.load("best.pth"))
wrapped = BlazeDocONNXWrapper(model)
torch.onnx.export(wrapped, dummy_input, "blazedoc.onnx")
