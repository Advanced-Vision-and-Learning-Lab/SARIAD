"""Anomaly Map Generator for the SARATRX model implementation.

This module generates anomaly heatmaps for the SARATRX model by computing
reconstruction errors between input images and their masked autoencoder
reconstructions.

The anomaly map generation process involves:
1. Unpatching the reconstruction from the MAE
2. Computing pixel-wise reconstruction error (L1 or L2)
3. Upsampling the error map to match input image size
4. Applying Gaussian smoothing to obtain the final anomaly map

Example:
    >>> from SARIAD.models.image.SARATRX.anomaly_map import AnomalyMapGenerator
    >>> generator = AnomalyMapGenerator(sigma=4)
    >>> input_img = torch.randn(32, 1, 224, 224)
    >>> reconstruction = torch.randn(32, 196, 768)  # Patched output
    >>> mask = torch.randn(32, 196)
    >>> anomaly_map = generator(
    ...     input_image=input_img,
    ...     reconstruction=reconstruction,
    ...     mask=mask,
    ...     patch_size=16
    ... )

See Also:
    - :class:`SARIAD.models.image.SARATRX.lightning_model.SARATRX`:
        Lightning implementation of the SARATRX model
    - :class:`anomalib.models.components.GaussianBlur2d`:
        Gaussian blur module used for smoothing anomaly maps
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
from torchvision.transforms.v2 import Grayscale

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap for SARATRX.

    This class implements anomaly map generation for the SARATRX model by
    computing reconstruction errors between the input and MAE reconstruction.

    Args:
        sigma (int, optional): Standard deviation for Gaussian smoothing kernel.
            Higher values produce smoother anomaly maps. Defaults to ``4``.
        error_mode (str, optional): Type of error to compute. Options are
            ``'l1'`` (mean absolute error) or ``'l2'`` (mean squared error).
            Defaults to ``'l2'``.

    Example:
        >>> generator = AnomalyMapGenerator(sigma=4, error_mode='l2')
        >>> input_img = torch.randn(32, 1, 224, 224)
        >>> reconstruction = torch.randn(32, 196, 768)
        >>> mask = torch.randn(32, 196)
        >>> anomaly_map = generator(
        ...     input_image=input_img,
        ...     reconstruction=reconstruction,
        ...     mask=mask,
        ...     patch_size=16
        ... )
    """

    def __init__(self, sigma: int = 4, error_mode: str = 'l2') -> None:
        super().__init__()
        
        if error_mode not in ['l1', 'l2']:
            msg = f"error_mode must be 'l1' or 'l2', got {error_mode}"
            raise ValueError(msg)
        
        self.error_mode = error_mode
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            channels=3
        )

    @staticmethod
    def unpatch_reconstruction(
        reconstruction: torch.Tensor,
        patch_size: int = 16,
        image_size: int = 224,
        channels: int = 3,
    ) -> torch.Tensor:
        """Convert patched reconstruction back to image format.

        Args:
            reconstruction (torch.Tensor): Reconstruction from MAE in patch format.
                Expected shape: ``(batch_size, num_patches, patch_dim)`` where
                ``patch_dim = patch_size * patch_size * channels``.
            patch_size (int, optional): Size of each patch. Defaults to ``16``.
            image_size (int, optional): Target image size (assumes square image).
                Defaults to ``224``.
            channels (int, optional): Number of image channels. Defaults to ``1``
                for grayscale.

        Returns:
            torch.Tensor: Reconstructed image with shape 
                ``(batch_size, channels, image_size, image_size)``

        Example:
            >>> reconstruction = torch.randn(8, 196, 256)  # 8 images, 14x14 patches
            >>> img = AnomalyMapGenerator.unpatch_reconstruction(
            ...     reconstruction, patch_size=16, image_size=224, channels=1
            ... )
            >>> img.shape
            torch.Size([8, 1, 224, 224])
        """
        batch_size = reconstruction.shape[0]
        num_patches_per_side = image_size // patch_size
        
        # Reshape from (B, num_patches, patch_dim) to spatial patch layout
        reconstruction = reconstruction.reshape(
            batch_size,
            num_patches_per_side,
            num_patches_per_side,
            patch_size,
            patch_size,
            channels
        )
        
        # Rearrange dimensions to standard image format (B, C, H, W)
        # Permute: (B, H_patches, W_patches, patch_h, patch_w, C) 
        #       -> (B, C, H_patches, patch_h, W_patches, patch_w)
        reconstruction = reconstruction.permute(0, 5, 1, 3, 2, 4)
        
        # Reshape to combine patches into full image
        reconstruction = reconstruction.reshape(
            batch_size,
            channels,
            image_size,
            image_size
        )
        
        return reconstruction

    def compute_reconstruction_error(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pixel-wise reconstruction error.

        Calculates the difference between the input image and its reconstruction
        using either L1 (absolute) or L2 (squared) error.

        Args:
            input_image (torch.Tensor): Original input image with shape
                ``(batch_size, channels, height, width)``
            reconstruction (torch.Tensor): Reconstructed image with same shape
                as input_image
            mask (torch.Tensor | None, optional): Optional binary mask indicating
                which regions to compute error on. Shape should match input_image.
                If provided, error is only computed on masked regions (where
                mask=1). Defaults to ``None``.

        Returns:
            torch.Tensor: Pixel-wise reconstruction error with shape
                ``(batch_size, channels, height, width)``

        Example:
            >>> generator = AnomalyMapGenerator(error_mode='l2')
            >>> input_img = torch.randn(4, 1, 224, 224)
            >>> recon_img = torch.randn(4, 1, 224, 224)
            >>> error = generator.compute_reconstruction_error(input_img, recon_img)
            >>> error.shape
            torch.Size([4, 1, 224, 224])
        """
        if self.error_mode == 'l1':
            error = torch.abs(input_image - reconstruction)
        else:  # l2
            error = (input_image - reconstruction) ** 2
        
        # If mask is provided, focus on masked regions
        if mask is not None:
            error = error * mask
        
        return error

    @staticmethod
    def process_mask(
        mask: torch.Tensor,
        image_size: tuple[int, int] | int,
        patch_size: int = 16,
    ) -> torch.Tensor:
        """Process mask from patch-level to pixel-level.

        Converts a binary mask defined at the patch level to pixel level by
        upsampling using nearest neighbor interpolation.

        Args:
            mask (torch.Tensor): Binary mask indicating masked patches with shape
                ``(batch_size, num_patches)`` where values are 0 or 1
            image_size (tuple[int, int] | int): Target image size. If int, assumes
                square image.
            patch_size (int, optional): Size of each patch. Defaults to ``16``.

        Returns:
            torch.Tensor: Pixel-level mask with shape
                ``(batch_size, 1, height, width)``

        Example:
            >>> mask = torch.randint(0, 2, (4, 196))  # 4 images, 14x14 patches
            >>> pixel_mask = AnomalyMapGenerator.process_mask(
            ...     mask, image_size=224, patch_size=16
            ... )
            >>> pixel_mask.shape
            torch.Size([4, 1, 224, 224])
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        batch_size = mask.shape[0]
        num_patches = mask.shape[1]
        h = w = int(num_patches ** 0.5)
        
        # Reshape mask from flat to spatial dimensions
        mask = mask.reshape(batch_size, 1, h, w)
        
        # Upsample to target image size using nearest neighbor interpolation
        mask = F.interpolate(
            mask.float(),
            size=image_size,
            mode="nearest"
        )
        
        return mask

    def smooth_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to the anomaly map.

        Reduces noise in the anomaly map by applying a Gaussian blur filter.

        Args:
            anomaly_map (torch.Tensor): Raw anomaly scores with shape
                ``(batch_size, 1, height, width)``

        Returns:
            torch.Tensor: Smoothed anomaly scores with same shape as input

        Example:
            >>> generator = AnomalyMapGenerator(sigma=4)
            >>> raw_map = torch.randn(4, 1, 224, 224)
            >>> smooth_map = generator.smooth_anomaly_map(raw_map)
            >>> smooth_map.shape
            torch.Size([4, 1, 224, 224])
        """
        return self.blur(anomaly_map)

    def compute_anomaly_map(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor | None = None,
        patch_size: int = 16,
    ) -> torch.Tensor:
        """Compute anomaly map from input and reconstruction.

        This is the main method that orchestrates the entire anomaly map generation
        pipeline: unpatching, error computation, mask processing, and smoothing.

        Args:
            input_image (torch.Tensor): Original input image with shape
                ``(batch_size, 1, 224, 224)``
            reconstruction (torch.Tensor): Reconstruction from MAE in patch format
                with shape ``(batch_size, num_patches, patch_dim)``
            mask (torch.Tensor | None, optional): Optional mask indicating masked
                patches with shape ``(batch_size, num_patches)``. Defaults to ``None``.
            patch_size (int, optional): Size of each patch. Defaults to ``16``.

        Returns:
            torch.Tensor: Final anomaly map after all processing steps with shape
                ``(batch_size, 1, 224, 224)``

        Example:
            >>> generator = AnomalyMapGenerator(sigma=4, error_mode='l2')
            >>> input_img = torch.randn(8, 1, 224, 224)
            >>> reconstruction = torch.randn(8, 196, 256)
            >>> mask = torch.randint(0, 2, (8, 196))
            >>> anomaly_map = generator.compute_anomaly_map(
            ...     input_img, reconstruction, mask, patch_size=16
            ... )
            >>> anomaly_map.shape
            torch.Size([8, 1, 224, 224])
        """
        image_size = input_image.shape[-1]
        channels = 3
        
        # Convert reconstruction from patches to image format
        reconstruction_img = self.unpatch_reconstruction(
            reconstruction,
            patch_size=patch_size,
            image_size=image_size,
            channels=channels,
        )
        
        # Process mask if provided
        pixel_mask = None
        if mask is not None:
            pixel_mask = self.process_mask(
                mask,
                image_size=image_size,
                patch_size=patch_size
            )
        
        # Compute reconstruction error
        error_map = self.compute_reconstruction_error(
            input_image,
            reconstruction_img,
            mask=pixel_mask
        )
        
        # Apply Gaussian smoothing to reduce noise
        anomaly_map = self.smooth_anomaly_map(error_map)

        to_gray = T.Grayscale(num_output_channels=1)
        anomaly_map = to_gray(anomaly_map)       

        return anomaly_map

    def forward(self, **kwargs) -> torch.Tensor:
        """Generate anomaly map from the provided input and reconstruction.

        This method provides a flexible interface for anomaly map generation.
        Expects ``input_image`` and ``reconstruction`` keywords to be passed
        explicitly. Optionally accepts ``mask`` and ``patch_size``.

        Args:
            **kwargs: Keyword arguments containing:
                - ``input_image`` (torch.Tensor): Required. Original input image
                - ``reconstruction`` (torch.Tensor): Required. Patched reconstruction
                - ``mask`` (torch.Tensor | None): Optional. Patch-level mask
                - ``patch_size`` (int): Optional. Patch size, defaults to 16

        Returns:
            torch.Tensor: Generated anomaly map

        Raises:
            ValueError: If required keys ``input_image`` or ``reconstruction``
                are not found in kwargs

        Example:
            >>> generator = AnomalyMapGenerator(sigma=4)
            >>> input_img = torch.randn(8, 1, 224, 224)
            >>> reconstruction = torch.randn(8, 196, 256)
            >>> anomaly_map = generator(
            ...     input_image=input_img,
            ...     reconstruction=reconstruction,
            ...     patch_size=16
            ... )
            >>> anomaly_map.shape
            torch.Size([8, 1, 224, 224])
        """
        if not ("input_image" in kwargs and "reconstruction" in kwargs):
            msg = f"Expected keys `input_image` and `reconstruction`. Found {kwargs.keys()}"
            raise ValueError(msg)

        input_image: torch.Tensor = kwargs["input_image"]
        reconstruction: torch.Tensor = kwargs["reconstruction"]
        mask: torch.Tensor | None = kwargs.get("mask")
        patch_size: int = kwargs.get("patch_size", 16)

        return self.compute_anomaly_map(
            input_image,
            reconstruction,
            mask=mask,
            patch_size=patch_size
        )
