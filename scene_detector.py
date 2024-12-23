import os
import cv2
import logging
import shutil  # <-- for deleting an existing directory
from typing import List, Tuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneDetector:
    """
    Detect scenes in a video file using PySceneDetect (v0.6+),
    then save representative images for each scene.
    """
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        threshold: float = 30.0,
        min_scene_len: int = 10
    ):
        """
        :param video_path: Path to the video file.
        :param output_dir: Directory to save detected scene images.
        :param threshold: Content detector threshold for scene detection.
        :param min_scene_len: Minimum scene length in frames.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_scene_len = min_scene_len

        # Will be initialized in initialize()
        self.video = None
        self.scene_manager = None

    def initialize(self) -> None:
        """
        Initialize the video handle (open_video) and SceneManager with a ContentDetector.
        """
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # open_video replaces the deprecated VideoManager in PySceneDetect v0.6+
        self.video = open_video(self.video_path)

        # Create a SceneManager and add the ContentDetector
        self.scene_manager = SceneManager()
        content_detector = ContentDetector(
            threshold=self.threshold,
            min_scene_len=self.min_scene_len
        )
        self.scene_manager.add_detector(content_detector)

    def detect_scenes(self) -> List[Tuple[int, int]]:
        """
        Detect scenes in the video and return a list of (start_frame, end_frame).
        """
        # Perform scene detection
        self.scene_manager.detect_scenes(video=self.video)

        # get_scene_list() returns a list of tuples: [(start_time, end_time), ...],
        # each element is a FrameTimecode pair.
        raw_scenes = self.scene_manager.get_scene_list()

        frame_ranges = []
        for start_timecode, end_timecode in raw_scenes:
            start_frame = start_timecode.get_frames()
            end_frame = end_timecode.get_frames()
            frame_ranges.append((start_frame, end_frame))

        logger.info(f"Detected {len(frame_ranges)} scenes.")
        return frame_ranges

    def save_scene_images(self, scene_list: List[Tuple[int, int]]) -> None:
        """
        For each scene, save a representative image (the middle frame).
        Skips black or nearly black frames.
        """
        if not scene_list:
            logger.info("No scenes to process.")
            return

        # Ensure the output directory exists (will be recreated if it was deleted)
        os.makedirs(self.output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        saved_count = 0

        for i, (start_frame, end_frame) in enumerate(scene_list, start=1):
            # Pick a middle frame to avoid transitional frames
            frame_to_capture = (start_frame + end_frame) // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Could not read frame %d for scene %d.", frame_to_capture, i)
                continue

            # Skip black frames
            if self.is_black_frame(frame):
                logger.info("Skipping scene %d as it appears to be black.", i)
                continue

            image_path = os.path.join(self.output_dir, f"{i}.jpg")
            cv2.imwrite(image_path, frame)
            saved_count += 1
            logger.debug("Saved scene image: %s", image_path)

        cap.release()
        logger.info("Saved %d representative scene images to '%s'.", saved_count, self.output_dir)

    def is_black_frame(self, frame) -> bool:
        """
        Check if a frame is black or nearly black by measuring its average brightness.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray)[0]
        # Adjust threshold based on your needs
        return mean_brightness < 10

    def process(self) -> None:
        """
        Main method to initialize detection, detect scenes, optionally skip
        the first black scene, and save representative images.
        """
        try:
            # 1. If the old scenes folder exists, delete it
            if os.path.isdir(self.output_dir):
                logger.info(f"Deleting old output directory: {self.output_dir}")
                shutil.rmtree(self.output_dir)

            # 2. Initialize PySceneDetect
            self.initialize()

            # 3. Detect scenes
            scene_list = self.detect_scenes()

            # 4. Optionally skip the first scene if it is black
            if scene_list:
                first_start, _ = scene_list[0]
                cap = cv2.VideoCapture(self.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_start)
                ret, frame = cap.read()
                if ret and self.is_black_frame(frame):
                    logger.info("First scene is black; skipping.")
                    scene_list = scene_list[1:]
                cap.release()

            # 5. Save representative images
            self.save_scene_images(scene_list)

        except Exception as e:
            logger.error("An error occurred during scene detection: %s", e, exc_info=True)
        finally:
            # 'open_video' returns a VideoStreamCv2, which doesn't require explicit release.
            logger.debug("Processing complete.")


def run_scene_detection(
    video_path: str,
    output_dir: str,
    threshold: float = 25.0,
    min_scene_len: int = 15
) -> None:
    """
    Convenience function to run scene detection with default parameters.

    :param video_path: Path to the video file.
    :param output_dir: Directory to save detected scene images.
    :param threshold: Content detection threshold (lower = more sensitive).
    :param min_scene_len: Minimum scene length in frames.
    """
    detector = SceneDetector(video_path, output_dir, threshold, min_scene_len)
    detector.process()


if __name__ == "__main__":
    # Example usage:
    video_path_example = "video.mp4"
    output_dir_example = "scenes"
    run_scene_detection(video_path_example, output_dir_example)
