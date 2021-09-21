import io
import logging
import subprocess
import sys
import threading
import time
from queue import Queue, Empty, Full

import boto3
import cv2
import ffmpeg
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STREAM_NAME = "fattah-stream-1"
# TODO: get resolution dynamically. how?
WIDTH = 640
HEIGHT = 480

def get_data_endpoint():
    client = boto3.client("kinesisvideo")
    response = client.get_data_endpoint(
                            StreamName=STREAM_NAME,
                            APIName="GET_MEDIA"
                            )
    return response["DataEndpoint"]

def write_bytes_to_buffer(process, payload):
    logger.info("Writing bytes to buffer")
    for chunk in payload.iter_chunks():
        process.stdin.write(chunk)

    logger.info("No bytes received. Triggering exit signal")
    is_no_bytes_received.set()

def read_frame_from_buffer(proc):
    frame = None
    frame_size = WIDTH * HEIGHT * 3

    in_bytes = proc.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
                 np
                 .frombuffer(in_bytes, np.uint8)
                 .reshape([HEIGHT, WIDTH, 3])
                )
    return frame

# TODO: do fancy inference process here
def process_frame(frame):
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def read_and_process_frame(process, frame_q):
    while True:
        in_frame = read_frame_from_buffer(process)
        if in_frame is None:
            logger.info("End of input stream. Ending stream")
            break

        out_frame = process_frame(in_frame)
        try:
            frame_q.put(out_frame, timeout=1)
        except Full:
            logger.warn("Queue is full")

def safe_exit(payload, process, write_t, read_t):
    cv2.destroyAllWindows()
    payload.close()
    process.stdin.close()
    process.wait()
    read_t.join()
    write_t.join()

if __name__ == "__main__":
    endpoint = get_data_endpoint()
    client = boto3.client("kinesis-video-media", endpoint_url=endpoint)
    start_selector = { "StartSelectorType": "NOW" }
    response = client.get_media(
                        StreamName=STREAM_NAME,
                        StartSelector=start_selector
                        )
    logger.info(response)
    payload = response["Payload"]

    process = (
        ffmpeg
        .input("pipe:")
        .video
        .output("pipe:", format="rawvideo", pix_fmt="bgr24", video_bitrate=1500)
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )

    is_no_bytes_received = threading.Event()
    write_t = threading.Thread(target=write_bytes_to_buffer, args=(process, payload,), daemon=True)
    frame_q = Queue()
    read_t = threading.Thread(target=read_and_process_frame, args=(process, frame_q,), daemon=True)

    write_t.start()
    read_t.start()

    while True:
        if is_no_bytes_received.is_set():
            break
        try:
            out_frame = frame_q.get(timeout=1)
        except Empty:
            logger.warn("Queue is empty")
        else:
            cv2.imshow("out_frame", out_frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    safe_exit(payload, process, write_t, read_t)
    logger.info("Done")
