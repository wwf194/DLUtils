import os, ffmpeg
import DLUtils
def Test():
    import sys
    # Compress input.mp4 to 50MB and save as output.mp4
    CompressVideo('input.mp4', 'output.mp4', 50 * 1000)

def VideoDuration(FilePath):
    FilePath = DLUtils.file.StandardizeFilePath(FilePath)
    VideoInfo = ffmpeg.probe(FilePath)
    Duration = VideoInfo['format']['duration']
    return Duration # unit: s

def VideoBitRate(FilePath):
    FilePath = DLUtils.file.StandardizeFilePath(FilePath)
    VideoInfo = ffmpeg.probe(FilePath)
    BitRate = float(next((s for s in VideoInfo['streams'] if s['codec_type'] == 'audio'), None)['bit_rate'])
    return BitRate # unit: s

def CompressVideo(FilePath, SavePath, Ratio=None, FileSizeTarget=None):
    # FileSizeTarget: unit: B
    FilePath = DLUtils.file.StandardizeFilePath(FilePath)
    SavePath = DLUtils.file.StandardizeFilePath(SavePath)
    FileSize = DLUtils.file.FileSize(FilePath) # unit: B
    
    if FileSizeTarget is None:
        assert Ratio is not None
        assert isinstance(Ratio, float)
        FileSizeTarget = FileSize * Ratio
        
    else:
        if isinstance(FileSizeTarget, str):
            FileSizeTarget = DLUtils.ToByteNum(FileSizeTarget)

    # Reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
    min_audio_bitrate = 32000
    max_audio_bitrate = 256000
    
    # FilePath = "\"" + FilePath.replace("/", "\\") + "\""
    probe = ffmpeg.probe(FilePath)
    # Video duration, in s.
    duration = float(probe['format']['duration'])
    # Audio bitrate, in bps.
    audio_bitrate = float(next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)['bit_rate'])
    # Target total bitrate, in bps.
    # target_total_bitrate = (FileSizeTarget * 1024 * 8) / (1.073741824 * duration)
    target_total_bitrate = (FileSizeTarget * 8) / (1.073741824 * duration)

    # Target audio bitrate, in bps
    if 10 * audio_bitrate > target_total_bitrate:
        audio_bitrate = target_total_bitrate / 10
        if audio_bitrate < min_audio_bitrate < target_total_bitrate:
            audio_bitrate = min_audio_bitrate
        elif audio_bitrate > max_audio_bitrate:
            audio_bitrate = max_audio_bitrate
    # Target video bitrate, in bps.
    video_bitrate = target_total_bitrate - audio_bitrate
    i = ffmpeg.input(FilePath)
    
    # ffmpeg.output(i, SavePath,
    #               **{'c:v': 'libx264', 'b:v': video_bitrate, 'c:a': 'aac', 'b:a': audio_bitrate}
    #               ).overwrite_output().run()
    
    
    ffmpeg.output(i, os.devnull,
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4',
                        "threads": 2, # control cpu usage
                        "max_muxing_queue_size": 1024 # try increase when encounter error: Too many packets buffered for output stream 0:1
                    }
                  ).overwrite_output().run()
    ffmpeg.output(i, SavePath,
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate, "threads": 2,
                        "max_muxing_queue_size": 1024
                    }
                  ).overwrite_output().run()

if __name__ == "__main__":
    Test()
    
    
    

import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import moviepy
from moviepy.editor import concatenate_videoclips
import cv2
def ImageList2Video(ImageList, FilePath, FPS=24):
    # ImageList: value range: [0, 255]
    FilePath = DLUtils.EnsureFileDir(FilePath)
    # for Frame in ImageList:
    #     pass
    # FrameList = concatenate_videoclips(ImageList, method="compose")
    # FrameList.write_videofile(FilePath, fps=24)
    clip = ImageSequenceClip(ImageList, fps=24)
    moviepy_logger = "bar" # or None
    clip.write_videofile(FilePath, logger=moviepy_logger)

def ImageList2VideoCv(ImageList, FilePath):
    FilePath = DLUtils.EnsureFileDir(FilePath)
    
    # Each video has a frame per second which is number of frames in every second
    frame_per_second = 15
    w, h = None, None
    for Frame in ImageList:
        # frame = cv2.imread(file)

        if w is None:
            # Setting up the video writer
            h, w, _ = Frame.shape
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(FilePath, fourcc, frame_per_second, (w, h))

        # # Repating the frame to fill the duration
        # for repeat in range(duration * frame_per_second):
        #     writer.write(frame)
    writer.release()