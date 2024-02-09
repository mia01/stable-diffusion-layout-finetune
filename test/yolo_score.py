# see https://towardsdatascience.com/object-detection-part2-6a265827efe1
# see https://github.com/ZGCTroy/LayoutDiffusion/blob/master/scripts/lama_yoloscore_test.py
# https://github.com/ultralytics/yolov5/issues/1911
# https://keras.io/examples/vision/yolov8/
# https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html - not sure of this one
def calculate_yolo_score(real_images, generated_images):
    r"""Calculate `YOLO Score`_ which is a metric used to evaluate the quality of generated images using the YOLO model.

    YOLO Score is a reference free metric that can be used to evaluate the quality of generated images using the YOLO model.

    Args:
        real_images (Union[Tensor, List[Tensor]]): A list of real images or a single tensor of real images.
        generated_images (Union[Tensor, List[Tensor]]): A list of generated images or a single tensor of generated images.

    Returns:
        Tensor: The YOLO Score.

    Example:
        >>> from torchvision.models.detection import fasterrcnn_resnet50_fpn
        >>> from torchmetrics.image.yolo import calculate_yolo_score
        >>> real_images = [torch.rand(3, 256, 256) for _ in range(10)]
        >>> generated_images = [torch.rand(3, 256, 256) for _ in range(10)]
        >>> score = calculate_yolo_score(real_images, generated_images)
    """
    raise NotImplementedError