import cv2
import os
from youtube_dlc import YoutubeDL
from pathlib import Path

def download_and_extract_frames(youtube_url):
    download_path = "Guitar_Chord_Recognition/youtube_frames"

    # Ensure the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Set up youtube-dl options with headers to avoid HTTP 403 errors
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
    }

    # Download the YouTube video
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        video_file = ydl.prepare_filename(info_dict)
    
    # Extract frames from the downloaded video
    extract_frames(video_file, download_path, frame_skip=30)  # Adjust frame_skip as needed


# Function to crop the right half of the frame
def crop_right_half(frame):
    height, width, _ = frame.shape
    right_half = frame[:, width // 2:width]
    return right_half

# Modify the extract_frames function to incorporate frame cropping
def extract_frames(video_path, download_path, frame_skip=30):
    video_name = Path(video_path).stem
    frame_folder = os.path.join(download_path, f"{video_name}_frames")

    # Ensure the frame save path exists
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    cap = cv2.VideoCapture(video_path)
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % frame_skip == 0:
            cropped_frame = crop_right_half(frame)  # Crop right half of the frame

            # Resize cropped frame to 224x224 pixels
            resized_frame = cv2.resize(cropped_frame, (224, 224))

            frame_filename = os.path.join(frame_folder, f"{video_name}_frame_{current_frame}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
        current_frame += 1

    cap.release()
    print(f"Frames are saved in {frame_folder}")
