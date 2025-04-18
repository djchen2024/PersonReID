import os, base64
import subprocess
import tempfile
from IPython.display import HTML, display

def vis_video(video_path:str="", start_time:int=0, end_time:int=100):
    with tempfile.NamedTemporaryFile(suffix=".MP4", delete=False) as tmpfile:
        output_filename = tmpfile.name
    try:
        # If both start_time and end_time are numeric, we can compute the duration.
        duration = None
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            duration = end_time - start_time
            if duration <= 0:
                raise ValueError("end_time must be greater than start_time to get a valid segment.")
        # Construct the ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output if it already exists
            "-ss",
            str(start_time),
            "-i",
            video_path,
        ]
        # If we have a numeric duration, explicitly specify it with -t
        if duration:
            cmd += ["-t", str(duration)]
        # "-c copy" copies the video/audio data without re-encoding, which is faster
        cmd += ["-c", "copy", output_filename]
        # Run the ffmpeg command
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Read the temporary MP4 file
        with open(output_filename, "rb") as f:
            video_data = f.read()
        # Encode file content as base64
        encoded = base64.b64encode(video_data).decode("utf-8")
        # Build the HTML5 <video> element with the base64-encoded data
        html_video = f"""
        <video controls>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        # Display the embedded video in the notebook
        display(HTML(html_video))
    finally:
        # Ensure the temporary file is removed, no matter what happened above
        if os.path.exists(output_filename):
            os.remove(output_filename)

import time
# def show_crops(crops:list, time_waits:float=0.1):
#     display_handle = display(None, display_id=True)
#     for c in crops:
#         display_handle.update(c)
#         time.sleep(time_waits)

def show_crops(frames, start=0, end=None, sleep_time=1):
    display_handle = display(None, display_id=True)
    end = end or len(frames)
    for frame in frames[start:end]:
        display_handle.update(frame)
        time.sleep(sleep_time)        