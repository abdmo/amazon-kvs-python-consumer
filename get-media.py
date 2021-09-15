import io
import logging
import threading
import subprocess

import boto3
import cv2
import ffmpeg
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
    for chunk in payload.iter_chunks():
        stream = io.BytesIO(chunk)
        process.stdin.write(stream.getvalue())

def read_frame_from_buffer(proc, width, height):
    logger.info("Reading frame")

    frame_size = width * height * 3
    in_bytes = proc.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
                 np
                 .frombuffer(in_bytes, np.uint8)
                 .reshape([height, width, 3])
                )
    return frame

# TODO: do fancy inference process here
def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

if __name__ == "__main__":
    endpoint = get_data_endpoint()
    client = boto3.client("kinesis-video-media", endpoint_url=endpoint)
    start_selector = { "StartSelectorType": "NOW" }
    response = client.get_media(
                        StreamName=STREAM_NAME,
                        StartSelector=start_selector
                        )
    logger.info(response)
    # TODO: check if response is successful
    payload = response["Payload"]

    process = (
               ffmpeg
               .input("pipe:")
               .video
               .output("pipe:", format="rawvideo", pix_fmt="bgr24")
               .run_async(pipe_stdin=True, pipe_stdout=True)
              )
    write_t = threading.Thread(target=write_bytes_to_buffer, args=(process, payload,), daemon=True)
    write_t.start()

    while True:
        in_frame = read_frame_from_buffer(process, WIDTH, HEIGHT)
        if in_frame is None:
            logger.info("End of input stream")
            break

        out_frame = process_frame(in_frame)

        cv2.imshow("out_frame", out_frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()

    logger.info("Waiting for write_to_buffer process")
    process.stdin.close()
    process.wait()
    write_t.join()
    logger.info("Done")

