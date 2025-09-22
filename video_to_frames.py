import cv2
import os
import sys

def video_to_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_num:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_num += 1
    cap.release()
    print(f"Extracted {frame_num} frames to '{output_dir}'")

def process_directory(input_base, output_base, video_exts=(".mp4", ".avi", ".mov", ".mkv")):
    for root, dirs, files in os.walk(input_base):
        for file in files:
            if file.lower().endswith(video_exts):
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_base)
                video_name = os.path.splitext(file)[0]
                output_dir = os.path.join(output_base, rel_path, video_name)
                video_to_frames(video_path, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video_to_frames.py <input_data_dir> <output_base_dir>")
        sys.exit(1)
    process_directory(sys.argv[1], sys.argv[2])
