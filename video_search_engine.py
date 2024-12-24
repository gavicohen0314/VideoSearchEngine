from yt_dlp import YoutubeDL
import sys
import logging
import os
import shutil
from scenedetect import SceneManager, AdaptiveDetector, save_images, open_video

VIDEO_PATH = 'video.mp4'
SCENES_PATH = 'scenes'
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def progress_hook(d):
    """
    Inline progress hook for the download.
    """
    if d['status'] == 'downloading':
        sys.stdout.write(f"\r{d.get('downloaded_bytes', 0)}")
        sys.stdout.flush()
    elif d['status'] == 'finished':
        logger.info("\nDownload complete!")


def download_video(query):
    """
    Function that downloads the first youtube video result for an inputted query.
    """
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': VIDEO_PATH,
        'default_search': 'ytsearch1',  # Use ytsearch1 to search and download the first result
        'no_warnings': True,
        'progress_hooks': [progress_hook],
        'quiet': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])


def detect_scenes():
    if os.path.isdir(SCENES_PATH):
        logger.info(f"Deleting old output directory: {SCENES_PATH}")
        shutil.rmtree(SCENES_PATH)
    video = open_video(VIDEO_PATH)
    scene_manager = SceneManager()
    detector = AdaptiveDetector()
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    save_images(
        scene_list,
        video,
        num_images=1,
        output_dir='scenes',
        image_name_template="${SCENE_NUMBER}"
    )


def main():
    try:
        logger.info("Downloading the video from YouTube...")
        download_video("super mario movie trailer")
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return
    detect_scenes()


if __name__ == "__main__":
    main()
