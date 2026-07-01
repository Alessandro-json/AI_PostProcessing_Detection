import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_depth_frequency import RRDepthFrequencyDatasetFromCSV
from model_depth_frequency_gated import RGBDepthFrequencyMultiTaskModel


FAKE_LABEL_NAMES = ["real", "fake"]
TRANSFORM_LABEL_NAMES = ["original", "transfer", "redigital"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grad-CAM for RGB+Depth+Frequency multi-task model."
    )

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--depth_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--task", type=str, required=True, choices=["fake", "transform"])
    parser.add_argument("--target", type=str, default="predicted", choices=["predicted", "true"])
    parser.add_argument("--target_class", type=int, default=None)

    parser.add_argument("--max_images", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--fake_filter", type=int, default=None, choices=[0, 1])
    parser.add_argument("--transform_filter", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")

    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key.replace("module.", "", 1)] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)

    return model


class RGBBranchGradCAM:
    """
    Grad-CAM computed on the RGB branch of the RGB+Depth+Frequency model.

    The final score comes from the full multimodal model, but the spatial
    activations are taken from the last convolutional block of the RGB ResNet.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self.save_activations
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, input_tensor, output_tensor):
        self.activations = output_tensor.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, images, depth, frequency, task, target_mode, target_class, fake_labels, transform_labels):
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(
            images=images,
            depth=depth,
            frequency=frequency,
        )

        if task == "fake":
            logits = outputs["fake_logits"]
            true_labels = fake_labels
        else:
            logits = outputs["transform_logits"]
            true_labels = transform_labels

        predicted_labels = torch.argmax(logits, dim=1)

        if target_class is not None:
            class_index = torch.tensor(
                [target_class],
                dtype=torch.long,
                device=logits.device,
            )
        elif target_mode == "true":
            class_index = true_labels
        else:
            class_index = predicted_labels

        score = logits[0, class_index.item()]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]
        cam = normalize_map(cam)

        return cam.cpu().numpy(), outputs, predicted_labels.detach()


def normalize_map(tensor):
    tensor = tensor.detach()

    min_value = tensor.min()
    max_value = tensor.max()

    if (max_value - min_value) < 1e-8:
        return torch.zeros_like(tensor)

    return (tensor - min_value) / (max_value - min_value)


def denormalize_rgb(image_tensor):
    image = image_tensor.detach().cpu()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()

    return image


def prepare_single_channel_map(map_tensor):
    map_array = map_tensor.detach().cpu().squeeze().numpy()

    min_value = map_array.min()
    max_value = map_array.max()

    if max_value - min_value < 1e-8:
        return np.zeros_like(map_array)

    return (map_array - min_value) / (max_value - min_value)


def make_overlay(rgb_image, cam):
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = 0.55 * rgb_image + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def save_gradcam_figure(
    output_path,
    rgb_image,
    depth_map,
    frequency_map,
    cam,
    overlay,
    title,
):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    axes[1].imshow(depth_map, cmap="gray")
    axes[1].set_title("Depth map")
    axes[1].axis("off")

    axes[2].imshow(frequency_map, cmap="gray")
    axes[2].set_title("Frequency map")
    axes[2].axis("off")

    axes[3].imshow(cam, cmap="jet")
    axes[3].set_title("Grad-CAM RGB branch")
    axes[3].axis("off")

    axes[4].imshow(overlay)
    axes[4].set_title("Overlay")
    axes[4].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RRDepthFrequencyDatasetFromCSV(
        csv_path=args.csv_path,
        image_root=args.image_root,
        depth_root=args.depth_root,
        image_size=args.image_size,
        train=False,
    )

    if args.fake_filter is not None:
        dataset.data = dataset.data[
            dataset.data["fake_label"] == args.fake_filter
        ].reset_index(drop=True)

    if args.transform_filter is not None:
        dataset.data = dataset.data[
            dataset.data["transform_label"] == args.transform_filter
        ].reset_index(drop=True)

    if len(dataset) == 0:
        raise ValueError("No images found after applying the selected filters.")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        generator=generator if args.shuffle else None,
    )

    model = RGBDepthFrequencyMultiTaskModel(
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        use_attention=not args.no_attention,
    )

    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.to(device)
    model.eval()

    target_layer = model.rgb_backbone.layer4[-1]
    gradcam = RGBBranchGradCAM(model, target_layer)

    saved_count = 0

    for batch in dataloader:
        if saved_count >= args.max_images:
            break

        images = batch["image"].to(device)
        depth = batch["depth"].to(device)
        frequency = batch["frequency"].to(device)

        fake_labels = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        cam, outputs, predicted_labels = gradcam.generate(
            images=images,
            depth=depth,
            frequency=frequency,
            task=args.task,
            target_mode=args.target,
            target_class=args.target_class,
            fake_labels=fake_labels,
            transform_labels=transform_labels,
        )

        if args.task == "fake":
            true_label = fake_labels.item()
            pred_label = torch.argmax(outputs["fake_logits"], dim=1).item()
            true_name = FAKE_LABEL_NAMES[true_label]
            pred_name = FAKE_LABEL_NAMES[pred_label]
        else:
            true_label = transform_labels.item()
            pred_label = torch.argmax(outputs["transform_logits"], dim=1).item()
            true_name = TRANSFORM_LABEL_NAMES[true_label]
            pred_name = TRANSFORM_LABEL_NAMES[pred_label]

        status = "CORRECT" if true_label == pred_label else "WRONG"

        rgb_image = denormalize_rgb(images[0])
        depth_map = prepare_single_channel_map(depth[0])
        frequency_map = prepare_single_channel_map(frequency[0])
        overlay = make_overlay(rgb_image, cam)

        image_path = batch["image_path"][0]
        image_name = Path(image_path).stem

        title = (
            f"Model: RGB+Depth+Frequency | "
            f"Task: {args.task} | "
            f"Real value: {true_name} | "
            f"Predicted: {pred_name} | "
            f"{status}"
        )

        output_path = output_dir / (
            f"{saved_count:03d}_{image_name}_{args.task}_{status.lower()}.png"
        )

        save_gradcam_figure(
            output_path=output_path,
            rgb_image=rgb_image,
            depth_map=depth_map,
            frequency_map=frequency_map,
            cam=cam,
            overlay=overlay,
            title=title,
        )

        print(f"Saved: {output_path}")

        saved_count += 1

    gradcam.remove_hooks()

    print()
    print(f"Saved {saved_count} Grad-CAM figures in: {output_dir}")


if __name__ == "__main__":
    main()