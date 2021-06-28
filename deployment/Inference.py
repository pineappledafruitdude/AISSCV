from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from Video import bgr8_to_jpeg


class Inference:

    def __init__(self, videoCapture: cv2.VideoCapture, videoOutput, configFilePath: str, dataFilePath: str, weightsFilePath: str, batch_size: str = 1, saveVideo: bool = False, out_filename: str = None):
        # Is the inference active
        self.running = False

        self.cap = videoCapture
        self.frame_queue = Queue()
        self.darknet_image_queue = Queue(maxsize=1)
        self.detections_queue = Queue(maxsize=1)
        self.fps_queue = Queue(maxsize=1)

        # Output (widget & file)
        self.videoOutput = videoOutput
        self.saveVideo = saveVideo
        self.out_filename = out_filename

        # Initialize Darknet
        self.network, self.class_names, self.class_colors = darknet.load_network(
            configFilePath,
            dataFilePath,
            weightsFilePath,
            batch_size
        )
        self.darknet_width = darknet.network_width(self.network)
        self.darknet_height = darknet.network_height(self.network)

        # Initialize Threads
        self.video_capture_thread = Thread(target=self.video_capture, args=(
            self.frame_queue, self.darknet_image_queue))
        self.inference_thread = Thread(target=self.inference, args=(
            self.darknet_image_queue, self.detections_queue, self.fps_queue))
        self.drawing_thread = Thread(target=self.drawing, args=(
            self.frame_queue, self.detections_queue, self.fps_queue))

    def startInference(self, button):
        if self.running:
            print("Already running")
            return
        if not self.cap.isOpened():
            print("Cant open video")
            return
        self.running = True
        self.video_capture_thread.start()
        self.inference_thread.start()
        self.drawing_thread.start()

        print("Running")

    def stopInference(self, button):
        if not self.running:
            print("Not running")
            return

        self.running = False
        self.video_capture_thread = Thread(target=self.video_capture, args=(
            self.frame_queue, self.darknet_image_queue))
        self.inference_thread = Thread(target=self.inference, args=(
            self.darknet_image_queue, self.detections_queue, self.fps_queue))
        self.drawing_thread = Thread(target=self.drawing, args=(
            self.frame_queue, self.detections_queue, self.fps_queue))
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
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.darknet_width, self.darknet_height),
                                       interpolation=cv2.INTER_LINEAR)
            frame_queue.put(frame)
            img_for_detect = darknet.make_image(
                self.darknet_width, self.darknet_height, 3)
            darknet.copy_image_from_bytes(
                img_for_detect, frame_resized.tobytes())
            darknet_image_queue.put(img_for_detect)
        self.cap.release()

    def inference(self, darknet_image_queue, detections_queue, fps_queue):
        while self.cap.isOpened():
            darknet_image = darknet_image_queue.get()
            prev_time = time.time()
            detections = darknet.detect_image(
                self.network, self.class_names, darknet_image, thresh=0.25)
            detections_queue.put(detections)
            fps = int(1/(time.time() - prev_time))
            fps_queue.put(fps)
            print("FPS: {}".format(fps))
            darknet.print_detections(detections, "store_true")
            darknet.free_image(darknet_image)
        self.cap.release()

    def drawing(self, frame_queue, detections_queue, fps_queue):
        random.seed(3)  # deterministic bbox colors

        # Save the video if necessary
        if self.saveVideo:
            video = self.set_saved_video(
                self.cap, self.out_filename, (self.darknet_width, self.darknet_height))

        while self.cap.isOpened():
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

                # Save to video if desired
                if self.saveVideo:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    video.write(image)

                # Output the inferred image
                self.videoOutput.value = bgr8_to_jpeg(image)

                if not self.running:
                    break

        self.cap.release()
        if self.saveVideo:
            video.release()
