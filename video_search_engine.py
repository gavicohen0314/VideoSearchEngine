from yt_dlp import YoutubeDL
import sys
import logging
import os
import shutil
from scenedetect import SceneManager, AdaptiveDetector, save_images, open_video
import moondream
from PIL import Image
import json
import requests
import gzip
from tqdm import tqdm

VIDEO_PATH = 'video.mp4'
SCENES_DIR = 'scenes'
MOONDREAM_MODEL_PATH = "moondream-0_5b-int8.mf"
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
        print("\nDownload complete!")


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
    """
    Function that detects scenes within a video and saves the scenes to an output directory.
    """
    if os.path.isdir(SCENES_DIR):
        print(f"Deleting old output directory: {SCENES_DIR}")
        shutil.rmtree(SCENES_DIR)
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
        output_dir=SCENES_DIR,
        image_name_template="${SCENE_NUMBER}"
    )


def download_moondream_model():
    url = "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz?download=true"
    print(f"Downloading moondream model from {url}...")
    response = requests.get(url, stream=True)
    gz_file_path = "moondream-0_5b-int8.mf.gz"
    if response.status_code == 200:
        with open(gz_file_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        
        print("Extracting compressed model...")
        with gzip.open(gz_file_path, "rb") as f_in:
            with open(MOONDREAM_MODEL_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_file_path)
    else:
        logger.error(f"Failed to download moondream model. Status code: {response.status_code}")


def generate_captions():
    """
    Function that uses a locally running vision model to caption each scene image in the scenes directory.
    The captions are saved to a JSON file.
    """
    captions = {}
    if not os.path.exists(MOONDREAM_MODEL_PATH):
        download_moondream_model()

    model = moondream.vl(model=MOONDREAM_MODEL_PATH)
    for i, scene in enumerate(tqdm(os.listdir(SCENES_DIR), desc="Generating captions"), start=1):
        scene_path = os.path.join(SCENES_DIR, scene)
        # Load and process image
        image = Image.open(scene_path)
        encoded_image = model.encode_image(image)
        captions[i] = model.caption(encoded_image, length='short')["caption"]  # Generate caption

    with open("scene_captions.json", "w") as json_file:
        json.dump(captions, json_file, indent=4)


def search_captions(query):
    with open("scene_captions.json", "r") as json_file:
        captions = json.load(json_file)

    matching_scenes = [scene_number for scene_number, caption in captions.items() if query.lower() in caption.lower()]
    return matching_scenes


def main():
    if not os.path.exists("scene_captions.json"):
        try:
            print("Downloading the video from YouTube...")
            download_video("super mario movie trailer")
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return
        detect_scenes()
        generate_captions()

    print("Search the video using a word:")
    search_query = input()
    print(search_captions(search_query))


if __name__ == "__main__":
    main()
