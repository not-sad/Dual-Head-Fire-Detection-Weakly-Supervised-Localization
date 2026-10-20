import gradio as gr
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn as nn
from torchvision.models import densenet121

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls_model = densenet121()
in_features = cls_model.classifier.in_features
cls_model.classifier = nn.Sequential(
    nn.Dropout(p=0.30),
    nn.Linear(in_features, 2)
)

try:
    cls_model.load_state_dict(torch.load("checkpoint_DenseNet121.pth", map_location=DEVICE))
    cls_model.to(DEVICE).eval()
except Exception as e:
    print(f"Warning loading classification weights: {e}")

seg_model = UNet().to(DEVICE)
try:
    seg_model.load_state_dict(torch.load("unet_fire_segmentation.pth", map_location=DEVICE))
    seg_model.eval()
except Exception as e:
    print(f"Warning loading segmentation weights: {e}")

def process_image(input_image, threshold):
    if input_image is None:
        return None, None, "<p>Upload an image to begin.</p>"
    orig_img = Image.fromarray(input_image).convert("RGB")
    resized_img = orig_img.resize((224, 224))
    img_tensor = TF.to_tensor(resized_img)
    norm_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_logits = cls_model(norm_tensor)
        cls_prob = torch.softmax(cls_logits, dim=1)[0, 1].item()
        seg_pred = seg_model(norm_tensor)[0, 0].cpu().numpy()

    binary_mask = (seg_pred >= threshold).astype(np.uint8)
    coverage_pct = float(binary_mask.mean() * 100)
    overlay = np.array(resized_img).copy()
    overlay[binary_mask == 1] = [255, 0, 0]

    is_fire = cls_prob >= threshold and coverage_pct > 0.5
    status_color = "#dc2626" if is_fire else "#16a34a"
    status_text = "FIRE DETECTED" if is_fire else "NO FIRE DETECTED"

    summary_html = f"""
    <div style="padding: 16px; border-radius: 8px; background-color: #f8fafc; border: 1px solid #e2e8f0;">
        <h3 style="margin-top:0; color: {status_color}; font-size: 20px;">STATUS: {status_text}</h3>
        <hr style="border: 0; border-top: 1px solid #cbd5e1; margin: 12px 0;">
        <p style="margin: 6px 0; font-size: 15px;"><b>Classification Confidence:</b> {cls_prob:.2%}</p>
        <p style="margin: 6px 0; font-size: 15px;"><b>Segmented Coverage Area:</b> {coverage_pct:.2f}%</p>
    </div>
    """
    return seg_pred, overlay, summary_html

demo = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(label="Source Image"), gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Threshold")],
    outputs=[gr.Image(label="Probability Heatmap"), gr.Image(label="Boundary Overlay"), gr.HTML(label="Metrics")],
    title="🔥 Dual-Head Fire Detection & Weakly Supervised Localization"
)

if __name__ == "__main__":
    demo.launch()