import matplotlib.pyplot as plt
import cv2

def img_debug(orig, mask, norm, title="Image Comparison"):
    """
    Displays three images (original, mask, and normal) side by side.

    Parameters:
        orig_img (np.array): The original image.
        mask_img (np.array): The mask image.
        norm_img (np.array): The generated normal image.
        title (str): The main title for the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)) # Matplot expects RGB so we convert
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Mask Image
    axes[1].imshow(mask, cmap='gray') # Use grayscale colormap for masks
    axes[1].set_title("Mask")
    axes[1].axis('off')

    # Normal Image
    axes[2].imshow(cv2.cvtColor(norm, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Generated Normal Image")
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
