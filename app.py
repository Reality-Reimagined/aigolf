from dotenv import load_dotenv
from IPython.display import display, Image, Audio
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

import cv2  # We're using OpenCV to read video
import base64
import time
import io
from openai import OpenAI
import os
import requests

import streamlit as st
import tempfile
import numpy as np

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)


def video_to_frames(video_file):
    # Save the uploaded video file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.read())
        video_filename = tmpfile.name
# def video_to_frames(video_file):
#     # Get the file extension from the uploaded file
#     file_extension = os.path.splitext(video_file.name)[1]

#     # Save the uploaded video file to a temporary file with the same extension
#     with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmpfile:
#         tmpfile.write(video_file.read())
#         video_filename = tmpfile.name

    # ... rest of the function remains the same


    video_duration = VideoFileClip(video_filename).duration

    video = cv2.VideoCapture(video_filename)
    base64Frames = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames, video_filename, video_duration


def frames_to_story(base64Frames, prompt):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 768},
                     base64Frames[0::40]),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1500,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content


def text_to_audio(text):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": "onyx",
        },
    )

    # audio_file_path = "output_audio.wav"
    # with open(audio_file_path, "wb") as audio_file:
    #     for chunk in response.iter_content(chunk_size=1024 * 1024):
    #         audio_file.write(chunk)

    # # To play the audio in Jupyter after saving
    # Audio(audio_file_path)
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception("Request failed with status code")
    # ...
    # Create an in-memory bytes buffer
    audio_bytes_io = io.BytesIO()

    # Write audio data to the in-memory bytes buffer
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # Important: Seek to the start of the BytesIO buffer before returning
    audio_bytes_io.seek(0)

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmpfile.write(chunk)
        audio_filename = tmpfile.name

    return audio_filename, audio_bytes_io


def merge_audio_video(video_filename, audio_filename, output_filename):
    print("Merging audio and video...")
    print("Video filename:", video_filename)
    print("Audio filename:", audio_filename)

    # Load the video file
    video_clip = VideoFileClip(video_filename)

    # Load the audio file
    audio_clip = AudioFileClip(audio_filename)

    # Set the audio of the video clip as the audio file
    final_clip = video_clip.set_audio(audio_clip)

    # Write the result to a file (without audio)
    final_clip.write_videofile(
        output_filename, codec='libx264', audio_codec='aac')

    # Close the clips
    video_clip.close()
    audio_clip.close()

    # Return the path to the new video file
    return output_filename


def main():
    st.set_page_config(page_title="Golf Swing Analysis", page_icon="🏌️")
    st.header("Golf Swing Analysis 🏌️")
    uploaded_file = st.file_uploader("Upload your golf swing video")


    if uploaded_file is not None:
        st.video(uploaded_file)
        prompt = st.text_area(
            "Prompt", value="These are still images of my golf swing. I want you to imagine that you are a golf pro watching me swing my club. give me personalized tips based on what you have seen in the stills. do not tell me you cannot give real time analysis or voiceover, just look at the stills and give some tips")
        # prompt ="These are still images of my golf swing. I want you to imagine that you are a golf pro watching me swing my club. give me personalized tips based on what you have seen in the stills. do not tell me you cannot give real time analysis or voiceover, just look at the stills and give some tips"

    if st.button('Generate', type="primary") and uploaded_file is not None:
        with st.spinner('Processing...'):
            base64Frames, video_filename, video_duration = video_to_frames(
                uploaded_file)

            est_word_count = video_duration * 2
            final_prompt = prompt + f"(This video is ONLY {video_duration} seconds long, so make sure the voice over MUST be able to be explained in less than {est_word_count} words)"

            # st.write(final_prompt)
            text = frames_to_story(base64Frames, final_prompt)
            st.write(text)

            # Generate audio from text
            audio_filename, audio_bytes_io = text_to_audio(text)

            # Merge audio and video
            output_video_filename = os.path.splitext(video_filename)[
                0] + '_output.mp4'
            final_video_filename = merge_audio_video(
                video_filename, audio_filename, output_video_filename)

            # Display the result
            st.video(final_video_filename)

            # Clean up the temporary files
            os.unlink(video_filename)
            os.unlink(audio_filename)
            os.unlink(final_video_filename)


if __name__ == '__main__':
    main()
