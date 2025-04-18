import cv2
import numpy as np
import imageio
 

def get_video_fps(video_path:str) -> float:
    input_video = cv2.VideoCapture(video_path)
    if not input_video.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = input_video.get(cv2.CAP_PROP_FPS)
    input_video.release()
    return fps

def get_video_hw(video_path:str) -> tuple[int,int]:
    input_video = cv2.VideoCapture(video_path)
    if not input_video.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_video.release()
    return width, height    

def save_video(frames:list, output_path:str, fps:float=30, method:str="imageio", bgr:bool=False):
    """
    Save a list of frames (ndarray) to a video file.
    """
    if not frames:
        raise ValueError("No frames to save.")
    if bgr:
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    if method == "cv2":
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()
    elif method == "imageio":
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=3,
        )
        for frame in frames:
            # print(frame.shape)
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()
    else:
        raise ValueError(f"Unsupported method: {method}")
    return output_path

def rotate_video(video_path:str):
    # Open the video file
    input_video = cv2.VideoCapture(video_path)

    # Get video properties including width, height, and frames per second (FPS)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_video_path = video_path+".mp4"
    output_video = cv2.VideoWriter(new_video_path, fourcc, fps, (frame_width, frame_height))

    # A loop to read frames from the input video and flip each frame one by one
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        flipped_frame = cv2.rotate(frame, cv2.ROTATE_180)
        # flipped_frame = cv2.flip(frame, 0)  # Use '0' for vertical flipping and '1' for horizontal flipping and '-1' for both.
        output_video.write(flipped_frame)

    # After the loop ends, release the video capture and writer objects and close all windows
    input_video.release() 
    output_video.release()
    cv2.destroyAllWindows()
    
    return new_video_path