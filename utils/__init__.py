"""Utility modules for shoplifting detection system."""

from .helpers import (
    load_config,
    setup_logging,
    set_seed,
    get_device,
    create_dirs,
    count_parameters,
    save_checkpoint,
    load_checkpoint
)

from .video_processing import (
    extract_frames,
    load_video_frames,
    preprocess_frame,
    compute_optical_flow,
    compute_motion_statistics,
    create_video_from_frames,
    sliding_window_clips
)

from .visualization import (
    draw_label_on_frame,
    draw_bounding_boxes,
    draw_fps,
    plot_confusion_matrix,
    plot_training_history,
    print_classification_report,
    create_side_by_side_comparison
)

__all__ = [
    'load_config',
    'setup_logging',
    'set_seed',
    'get_device',
    'create_dirs',
    'count_parameters',
    'save_checkpoint',
    'load_checkpoint',
    'extract_frames',
    'load_video_frames',
    'preprocess_frame',
    'compute_optical_flow',
    'compute_motion_statistics',
    'create_video_from_frames',
    'sliding_window_clips',
    'draw_label_on_frame',
    'draw_bounding_boxes',
    'draw_fps',
    'plot_confusion_matrix',
    'plot_training_history',
    'print_classification_report',
    'create_side_by_side_comparison'
]
