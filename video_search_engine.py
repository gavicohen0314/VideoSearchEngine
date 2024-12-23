from yt_dlp import YoutubeDL
import sys
from scene_detector import run_scene_detection


# Progress hook for download
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
        'outtmpl': 'video.mp4',
        'default_search': 'ytsearch1',  # Use ytsearch1 to search and download the first result
        'no_warnings': True,
        'progress_hooks': [progress_hook],
        'quiet': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])


if __name__ == "__main__":
    try:
        print("Downloading the video from YouTube...")
        download_video("super mario movie trailer")
    except Exception as e:
        print(f"Error downloading video: {e}")
