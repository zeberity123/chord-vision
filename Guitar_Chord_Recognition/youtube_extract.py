import cv2
import os
from youtube_dlc import YoutubeDL
from pathlib import Path

# Replace this URL with the YouTube link you want to download
youtube_url = "https://www.youtube.com/watch?v=fHC5KZtcljo"
download_path = "Guitar_Chord_Recognition/youtube_frames"

def download_video(youtube_url, download_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        video_file = ydl.prepare_filename(info_dict)
        return video_file

# Function to perform YOLO hand detection and crop frames
def detect_and_crop_hand(frame):
    # Implement YOLO hand detection and cropping logic here
    # This is just a placeholder, replace it with your actual implementation
    cropped_frame = frame[50:250, 50:250]  # Placeholder cropping, replace with YOLO logic
    return cropped_frame

# Modify the extract_frames function to incorporate YOLO hand detection and cropping
def extract_frames(video_path, frame_skip=30):
    video_name = Path(video_path).stem
    frame_folder = os.path.join(download_path, f"{video_name}_frames")
    
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
    
    cap = cv2.VideoCapture(video_path)
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % frame_skip == 0:
            cropped_frame = detect_and_crop_hand(frame)  # Get cropped frame with hand detection

            # Resize cropped frame to 224x224 pixels
            resized_frame = cv2.resize(cropped_frame, (224, 224))

            frame_filename = os.path.join(frame_folder, f"{video_name}_frame_{current_frame}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
        current_frame += 1

    cap.release()
    print(f"Frames are saved in {frame_folder}")

if __name__ == "__main__":
    video_file = download_video(youtube_url, download_path)
    extract_frames(video_file, frame_skip=30)  # Change to 60 when needed