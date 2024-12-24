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
from rapidfuzz import process
import matplotlib.pyplot as plt
import math
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import time
import google.generativeai as genai
from dotenv import load_dotenv
import questionary
import cv2

VIDEO_PATH = 'video.mp4'
SCENES_DIR = 'scenes'
MOONDREAM_MODEL_PATH = "moondream-0_5b-int8.mf"
# Configure logging
logging.basicConfig(level=logging.ERROR)
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
    for scene in tqdm(os.listdir(SCENES_DIR), desc="Generating captions"):
        scene_path = os.path.join(SCENES_DIR, scene)
        # Load and process image
        image = Image.open(scene_path)
        encoded_image = model.encode_image(image)
        captions[scene_path] = model.caption(encoded_image, length='short')["caption"]  # Generate caption

    with open("scene_captions.json", "w") as json_file:
        json.dump(captions, json_file, indent=4)


def load_captions():
    with open("scene_captions.json", "r") as json_file:
        captions = json.load(json_file)
    return captions


def get_words_from_captions(captions_dict):
    """
    Extracts unique words from a dictionary of captions.
    Args:
        captions_dict (dict): A dictionary where values are strings (captions).
    Returns:
        list: A list of unique words.
    """
    words = set()
    for caption in captions_dict.values():
        words.update(caption.split())
    return sorted(words)


def get_user_input(captions_dict):
    """
    Prompts the user for input with autocomplete, using words from captions_dict.
    Args:
        captions_dict (dict): A dictionary where values are captions (strings).
    Returns:
        str: User input.
    """
    # Extract unique words from the captions
    suggestions = get_words_from_captions(captions_dict)
    # Create a WordCompleter instance with the suggestions
    autocomplete = WordCompleter(suggestions, ignore_case=True)
    # Prompt the user with autocomplete
    user_input = prompt("Search the video using a word: ", completer=autocomplete)
    return user_input


def search_captions(search_query, captions):
    matching_scenes = [
        scene for scene,
        caption in captions.items()
        if process.extractOne(search_query.lower(), [caption.lower()], score_cutoff=50)
    ]
    return matching_scenes


def create_collage(image_paths, output_path="collage.png", images_per_row=5, image_size=(640, 360)):
    """
    Creates a collage of images from the provided paths and saves it as a PNG file.

    Args:
        image_paths (list): List of file paths to images.
        output_path (str): Path to save the collage as a PNG file.
        images_per_row (int): Number of images per row in the collage.
        image_size (tuple): Size of each image in the collage (width, height).

    Returns:
        None
    """
    if not image_paths:
        print("No images provided.")
        return

    # Calculate the number of rows and columns
    num_images = len(image_paths)
    num_rows = math.ceil(num_images / images_per_row)

    # Create a blank canvas for the collage
    collage_width = images_per_row * image_size[0]
    collage_height = num_rows * image_size[1]
    collage = Image.new("RGB", (collage_width, collage_height), "white")

    # Add images to the collage
    for index, image_path in enumerate(image_paths):
        row = index // images_per_row
        col = index % images_per_row
        try:
            img = Image.open(image_path).resize(image_size)
            collage.paste(img, (col * image_size[0], row * image_size[1]))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    # Save the collage
    collage.save(output_path)
    print(f"Collage saved to {output_path}")

    # Display the collage
    plt.figure(figsize=(10, 10))
    plt.imshow(collage)
    plt.axis("off")
    plt.show()


def upload_and_prompt(file_path, search_query, mime_type="video/mp4"):
    """
    Uploads a single video file to Gemini, waits for it to become active, prompts the model,
    and returns the JSON response.

    Args:
        file_path (str): Path to the video file to upload.
        prompt (str): The prompt to send to the model.
        mime_type (str): The MIME type of the video file (default: "video/mp4").

    Returns:
        dict: The JSON response from the model.
    """
    # Configure the API
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Upload the video file
    file = genai.upload_file(file_path, mime_type=mime_type)
    print("Uploaded video...")

    # Wait for the file to be active
    print("Waiting for file processing...")
    while True:
        file_status = genai.get_file(file.name)
        if file_status.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
        elif file_status.state.name == "ACTIVE":
            print("..file is ready!\n")
            break
        else:
            raise Exception(f"File {file.name} failed to process")

    # Create the model
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Start a chat session with the model
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [file],
            },
        ]
    )

    # Send the prompt and get the response
    prompt = f"""Expected output is a JSON as follows:
{{'timestamps': ['00:02', '01:12', ...]}}
Make sure that hours, seconds and minutes are returned accurately. Use leading 0s for hours and minutes.
Provide all the timestamps in mm:ss fitting the following user query: {search_query}"""
    response = chat_session.send_message(prompt)

    return response.text


def image_search():
    if not os.path.exists("scene_captions.json"):
        detect_scenes()
        generate_captions()

    captions = load_captions()
    search_query = get_user_input(captions)
    create_collage(search_captions(search_query, captions))


def process_timestamps(timestamps):
    folder_name = "temp_images"
    os.makedirs(folder_name, exist_ok=True)
    video = cv2.VideoCapture(VIDEO_PATH)
    fps = video.get(cv2.CAP_PROP_FPS)
    image_paths = []

    for timestamp in timestamps:
        timestamp_list = timestamp.split(':')
        mm, ss = timestamp_list
        timestamp_list_floats = [float(i) for i in timestamp_list]
        minutes, seconds = timestamp_list_floats
        frame_nr = fps*(minutes * 60 + seconds)
        video.set(1, frame_nr)
        _, frame = video.read()
        frame_path = os.path.join(folder_name, f'{mm}-{ss}.jpg')
        cv2.imwrite(frame_path, frame)
        image_paths.append(frame_path)

    create_collage(image_paths)
    # Cleanup: Delete the folder and its contents
    try:
        shutil.rmtree(folder_name)
    except Exception as e:
        logger.error(f"Error deleting folder: {e}")


def video_search():
    print("Using a video model. What would you like me to find in the video?")
    search_query = input()
    response_json = upload_and_prompt(VIDEO_PATH, search_query)
    try:
        json_data = json.loads(response_json)
        timestamps = json_data['timestamps']
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Gemini: {e}")
        return
    process_timestamps(timestamps)


def choose_model():
    answer = questionary.select(
        "How would you like to search the video?",
        choices=["Search using Image Model", "Search using Video Model"]
    ).ask()

    if answer == "Search using Image Model":
        print("You selected: Search using Image Model")
        image_search()
    elif answer == "Search using Video Model":
        print("You selected: Search using Video Model")
        video_search()


def main():
    if not os.path.exists("video.mp4"):
        try:
            print("Downloading the video from YouTube...")
            download_video("super mario movie trailer")
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return

    choose_model()



if __name__ == "__main__":
    main()
