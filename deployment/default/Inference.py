from ctypes import *
from Video import gstreamer_pipeline
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, currentThread
from queue import Queue
from Video import bgr8_to_jpeg
from pathlib import Path
import ipywidgets
from IPython.display import display


class Inference:

    def __init__(self, video: str, videoOutput, configFilePath: str, dataFilePath: str, weightsFilePath: str, border: bool = False, thresh: float = 0.7):
        """Inference class

        Args:
            video (str): Either 'camera' or path to a video file
            videoOutput (any): Jupyter video widget
            configFilePath (str): Path to yolo.cfg
            dataFilePath (str): Path to darknet.data
            weightsFilePath (str): Path to yolo weights file
            border (bool, optional): Draw a black border around the image. Defaults to False.
            thresh (float, optional): Inference score threshold. Defaults to 0.7.
        """
        # Is the inference active
        self.running = False
        self.video = video

        self.frame_queue = Queue()
        self.darknet_image_queue = Queue(maxsize=1)
        self.detections_queue = Queue(maxsize=1)
        self.fps_queue = Queue(maxsize=1)

        # Output (widget)
        self.videoOutput = videoOutput

        # Initialize Darknet
        if not Path(configFilePath).exists():
            raise Exception("Yolo config file not found")
        if not Path(dataFilePath).exists():
            raise Exception("Yolo config file not found")
        if not Path(weightsFilePath).exists():
            raise Exception("Yolo config file not found")

        self.network, self.class_names, self.class_colors = darknet.load_network(
            configFilePath,
            dataFilePath,
            weightsFilePath,
            1
        )
        self.darknet_width = darknet.network_width(self.network)
        self.darknet_height = darknet.network_height(self.network)

        self.border = border
        self.thresh = thresh

        # Initialize button widgets
        self.start_button = ipywidgets.Button(
            description='Start',
            disabled=False,
            button_style='success',
            tooltip='Click me',
            icon='check'
        )

        self.stop_button = ipywidgets.Button(
            description='Stop',
            disabled=False,
            button_style='danger',
            tooltip='Click me',
            icon='stop'
        )
        self.start_button.on_click(self.startInference)
        self.stop_button.on_click(self.stopInference)

        # Initialize image widget
        self.image_widget = ipywidgets.Image(format='jpeg')
        self.startCamera()
        if self.cap.isOpened():
            _, image = self.cap.read()
            self.cap.release()
            self.image_widget.value = bgr8_to_jpeg(image)

        # Display all widgets
        display(self.start_button)
        display(self.stop_button)
        display(self.image_widget)

    def startCamera(self):
        """Start a Video capture session either from camera or video file"""
        if self.video == "camera":
            self.cap = cv2.VideoCapture(gstreamer_pipeline(
                capture_width=416, capture_height=416, flip_method=0), cv2.CAP_GSTREAMER)
        else:
            video_path = Path(self.video)
            if not video_path.exists():
                raise Exception("Video file not found")
            self.cap = cv2.VideoCapture(str(video_path))

    def startInference(self, *args):
        """Start the inference

        Args:
            button: Jupyter button
        """
        if self.running:
            print("Already running")
            return
        # Initialize capture session
        self.startCamera()
        if not self.cap.isOpened():
            print("Cant open video")
            return

        # Initialize Threads
        self.video_capture_thread = Thread(target=self.video_capture, args=(
            self.frame_queue, self.darknet_image_queue))
        self.inference_thread = Thread(target=self.inference, args=(
            self.darknet_image_queue, self.detections_queue, self.fps_queue))
        self.drawing_thread = Thread(target=self.drawing, args=(
            self.frame_queue, self.detections_queue, self.fps_queue))

        self.video_capture_thread.start()
        self.inference_thread.start()
        self.drawing_thread.start()

        self.running = True

        print("Running")

    def stopInference(self, *args):
        """Stop the inference

        Args:
            button: Jupyter button
        """
        if not self.running:
            print("Not running")
            return

        self.video_capture_thread.running = False
        self.inference_thread.running = False
        self.drawing_thread.running = False
        self.cap.release()
        self.running = False
        print("Stopped")

    def str2int(self, video_path):
        """
        argparse returns and string althout webcam uses int (0, 1 ...)
        Cast to int if needed
        """
        try:
            return int(video_path)
        except ValueError:
            return video_path

    def set_saved_video(self, input_video, output_video, size):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        video = cv2.VideoWriter(output_video, fourcc, fps, size)
        return video

    def convert2relative(self, bbox):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h = bbox
        _height = self.darknet_height
        _width = self.darknet_width
        return x/_width, y/_height, w/_width, h/_height

    def convert2original(self, image, bbox):
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_x = int(x * image_w)
        orig_y = int(y * image_h)
        orig_width = int(w * image_w)
        orig_height = int(h * image_h)

        bbox_converted = (orig_x, orig_y, orig_width, orig_height)

        return bbox_converted

    def convert4cropping(self, image, bbox):
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_left = int((x - w / 2.) * image_w)
        orig_right = int((x + w / 2.) * image_w)
        orig_top = int((y - h / 2.) * image_h)
        orig_bottom = int((y + h / 2.) * image_h)

        if (orig_left < 0):
            orig_left = 0
        if (orig_right > image_w - 1):
            orig_right = image_w - 1
        if (orig_top < 0):
            orig_top = 0
        if (orig_bottom > image_h - 1):
            orig_bottom = image_h - 1

        bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

        return bbox_cropping

    def video_capture(self, frame_queue, darknet_image_queue):
        t = currentThread()
        while getattr(t, "running", True):
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.border:
                bordersize = 40
                frame = cv2.copyMakeBorder(
                    frame,
                    top=bordersize,
                    bottom=bordersize,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.darknet_width, self.darknet_height),
                                       interpolation=cv2.INTER_LINEAR)
            frame_queue.put(frame)
            img_for_detect = darknet.make_image(
                self.darknet_width, self.darknet_height, 3)
            darknet.copy_image_from_bytes(
                img_for_detect, frame_resized.tobytes())
            darknet_image_queue.put(img_for_detect)

    def inference(self, darknet_image_queue, detections_queue, fps_queue):
        t = currentThread()
        while getattr(t, "running", True):
            darknet_image = darknet_image_queue.get()
            prev_time = time.time()
            detections = darknet.detect_image(
                self.network, self.class_names, darknet_image, thresh=self.thresh)
            detections_queue.put(detections)
            fps = int(1/(time.time() - prev_time))
            fps_queue.put(fps)
            print("FPS: {}".format(fps))
            darknet.print_detections(detections, "store_true")
            darknet.free_image(darknet_image)

    def drawing(self, frame_queue, detections_queue, fps_queue):
        t = currentThread()
        random.seed(3)  # deterministic bbox colors

        while getattr(t, "running", True):
            # Break if the inference is not be active (anymore)
            frame = frame_queue.get()
            detections = detections_queue.get()
            fps = fps_queue.get()
            detections_adjusted = []

            if frame is not None:
                for label, confidence, bbox in detections:
                    bbox_adjusted = self.convert2original(frame, bbox)
                    detections_adjusted.append(
                        (str(label), confidence, bbox_adjusted))
                image = darknet.draw_boxes(
                    detections_adjusted, frame, self.class_colors)

                # Output the inferred image
                self.image_widget.value = bgr8_to_jpeg(image)
