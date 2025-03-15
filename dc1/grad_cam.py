import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        """ Store activations of the target layer """
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        """ Store gradients of the target layer """
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_tensor, class_idx=None):
        """ Compute Grad-CAM heatmap for a given input image """
        self.model.eval()

        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor.requires_grad = True

        # Forward pass
        output = self.model(image_tensor)

        # If no class index is provided, take the highest prediction
        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Compute Grad-CAM heatmap
        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        weights = np.mean(gradients, axis=(2, 3))  # Global Average Pooling
        heatmap = np.sum(weights[:, :, None, None] * activations, axis=1)

        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap = heatmap[0]  # Remove batch dimension

        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4):
        """ Overlay Grad-CAM heatmap onto the original X-ray image """
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(heatmap, alpha, original_image, 1 - alpha, 0)
        return overlay

    def visualize(self, image_tensor, original_image, class_idx=None):
        """ Display the Grad-CAM heatmap overlaid on the original image """
        heatmap = self.generate_heatmap(image_tensor, class_idx)
        overlay = self.overlay_heatmap(heatmap, original_image)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")

        plt.show()
