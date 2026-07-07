import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_depth import RRGeometricDatasetFromCSV
from model_depth import GeometricMultiTaskModel


FAKE_LABEL_NAMES = ["real", "ai"]
TRANSFORM_LABEL_NAMES = ["original", "transfer", "redigital"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_checkpoint(model, checkpoint_path, device):
    """
    Load trained weights into the RGB + depth model.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


class RGBBranchGradCAM:
    """
    Grad-CAM for the RGB branch of the RGB + depth model.

    The final prediction depends on RGB features, depth features, and the fusion module.
    This Grad-CAM visualizes only the spatial contribution of the RGB branch.
    """

    def __init__(self, model, target_layer):
        """
        Register hooks on the target RGB convolutional layer.
        """

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, images, depth, edge_consistency, task, target_class=None):
        """
        Generate Grad-CAM heatmaps for a batch of images.
        """

        self.model.zero_grad(set_to_none=True)

        outputs = self.model(
            images=images,
            depth=depth,
            edge_consistency=edge_consistency,
        )

        if task == "fake":
            logits = outputs["fake_logits"]
        elif task == "transform":
            logits = outputs["transform_logits"]
        else:
            raise ValueError("task must be either 'fake' or 'transform'.")

        predicted_class = torch.argmax(logits, dim=1)

        if target_class is None:
            selected_class = predicted_class
        else:
            selected_class = torch.full(
                size=(images.size(0),),
                fill_value=target_class,
                device=images.device,
                dtype=torch.long,
            )

        score = logits.gather(1, selected_class.view(-1, 1)).sum()

        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach(), predicted_class.detach()


def denormalize_rgb(image_tensor):
    """
    Convert an ImageNet-normalized tensor into a displayable RGB image.
    """

    image = image_tensor.detach().cpu()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()

    return image


def normalize_map(map_tensor):
    """
    Normalize a single-channel tensor to the [0, 1] range for visualization.
    """

    x = map_tensor.detach().cpu().squeeze().numpy()
    x_min = x.min()
    x_max = x.max()

    if x_max - x_min < 1e-8:
        return np.zeros_like(x)

    return (x - x_min) / (x_max - x_min)


def make_overlay(rgb_image, cam):
    """
    Overlay the Grad-CAM heatmap on the original RGB image.
    """

    heatmap = plt.get_cmap("jet")(cam.squeeze())[:, :, :3]
    overlay = 0.55 * rgb_image + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def save_gradcam_figure(
    rgb_image,
    depth_map,
    cam,
    overlay,
    output_path,
    title,
):
    """
    Save a figure with RGB image, depth map, Grad-CAM heatmap, and overlay.
    """

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    axes[1].imshow(depth_map, cmap="gray")
    axes[1].set_title("Depth map")
    axes[1].axis("off")

    axes[2].imshow(cam.squeeze(), cmap="jet")
    axes[2].set_title("Grad-CAM RGB branch")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    """
    Read command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for the RGB branch of the RGB + depth model."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="CSV file containing image paths and labels.",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root folder containing RGB images.",
    )

    parser.add_argument(
        "--depth_root",
        type=str,
        required=True,
        help="Root folder containing precomputed depth maps.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained RGB + depth checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gradcam_depth",
        help="Folder where Grad-CAM images will be saved.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="fake",
        choices=["fake", "transform"],
        help="Model head used for Grad-CAM.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="predicted",
        choices=["predicted", "true"],
        help="Use the predicted class or the true label as the Grad-CAM target.",
    )

    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Optional fixed target class. If set, it overrides --target.",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=12,
        help="Maximum number of images to visualize.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--no_edge",
        action="store_true",
        help="Use this flag if the checkpoint was trained without edge consistency.",
    )

    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Use this flag if the checkpoint was trained without attention.",
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Use this flag if the checkpoint was trained without ImageNet pretrained weights.",
    )

    parser.add_argument(
	"--fake_filter",
	type=int,
	default=None,
	choices=[0, 1],
	help=(
		"Optional filter for the real/fake ground-truth label. "
		"Use 0 for real images and 1 for AI-generated images."
	),
	)

    parser.add_argument(
		"--transform_filter",
		type=int,
		default=None,
		choices=[0, 1, 2],
		help=(
			"Optional filter for the transformation ground-truth label. "
			"Use 0 for original, 1 for transfer, and 2 for redigital."
		),
	)

    parser.add_argument(
		"--shuffle",
		action="store_true",
		help="Shuffle the filtered dataset before generating Grad-CAM images.",
	)

    parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used when --shuffle is enabled.",
	)

    return parser.parse_args()


def main():
    """
    Main Grad-CAM pipeline.
    """

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RRGeometricDatasetFromCSV(
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
	    raise ValueError(
			"No samples found after applying the selected filters. "
			"Check --fake_filter and --transform_filter."
		)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=args.shuffle,
		num_workers=args.num_workers,
		generator=generator if args.shuffle else None,
	)

    model = GeometricMultiTaskModel(
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        use_edge=not args.no_edge,
        use_attention=not args.no_attention,
    )

    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model = model.to(device)
    model.eval()

    target_layer = model.rgb_backbone.layer4[-1]

    gradcam = RGBBranchGradCAM(
        model=model,
        target_layer=target_layer,
    )

    saved_count = 0

    for batch in dataloader:
        if saved_count >= args.max_images:
            break

        images = batch["image"].to(device)
        depth = batch["depth"].to(device)

        if args.no_edge:
            edge_consistency = None
        else:
            edge_consistency = batch["edge_consistency"].to(device)

        true_fake = int(batch["fake_label"][0])
        true_transform = int(batch["transform_label"][0])

        if args.target_class is not None:
            target_class = args.target_class
        elif args.target == "true":
            if args.task == "fake":
                target_class = true_fake
            else:
                target_class = true_transform
        else:
            target_class = None

        cam, predicted_class = gradcam.generate(
            images=images,
            depth=depth,
            edge_consistency=edge_consistency,
            task=args.task,
            target_class=target_class,
        )

        rgb_image = denormalize_rgb(images[0])
        depth_map = normalize_map(depth[0])
        cam_np = cam[0].cpu().numpy()
        overlay = make_overlay(rgb_image, cam_np)

        if args.task == "fake":
            true_name = FAKE_LABEL_NAMES[true_fake]
            pred_name = FAKE_LABEL_NAMES[int(predicted_class[0])]
        else:
            true_name = TRANSFORM_LABEL_NAMES[true_transform]
            pred_name = TRANSFORM_LABEL_NAMES[int(predicted_class[0])]

        image_name = Path(batch["image_path"][0]).stem

        title = (
            f"Task: {args.task} | "
            f"True: {true_name} | "
            f"Pred: {pred_name}"
        )

        output_path = output_dir / f"{saved_count:03d}_{image_name}_{args.task}.png"

        save_gradcam_figure(
            rgb_image=rgb_image,
            depth_map=depth_map,
            cam=cam_np,
            overlay=overlay,
            output_path=output_path,
            title=title,
        )

        print(f"Saved: {output_path}")

        saved_count += 1

    gradcam.remove_hooks()

    print(f"\nSaved {saved_count} Grad-CAM visualizations in:")
    print(output_dir)


if __name__ == "__main__":
    main()