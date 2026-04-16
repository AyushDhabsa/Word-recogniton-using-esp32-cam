#!/usr/bin/env python3
"""
ESP32-CAM Handwriting OCR (Tkinter)

UI:
- Live feed panel
- Capture panel (shows the frame used for OCR)
- Digital result bar for recognized text
- Capture + Analyze and Save buttons

OCR:
- Basic mode: 2-pass OCR
- Advanced mode: multi-pass OCR tuned for handwriting
- Ultra mode: heavy multi-pass OCR with extra preprocessing
"""

import argparse
import os
import queue
import threading
import time
from datetime import datetime
from urllib.parse import urlparse

import cv2
import easyocr
import numpy as np
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="ESP32-CAM Tkinter Handwriting OCR")
    parser.add_argument("--ip", default="192.168.4.1", help="ESP32 IP/host or full URL")
    parser.add_argument("--port", default=80, type=int, help="HTTP port")
    parser.add_argument("--stream-path", default="/stream", help="MJPEG stream path")
    parser.add_argument(
        "--snapshot-paths",
        default="/capture,/cam-hi,/cam-mid,/cam-lo",
        help="Comma-separated snapshot paths fallback",
    )
    parser.add_argument("--lang", default="en", help="OCR language list, e.g. en or en,hi")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU for EasyOCR when available")
    parser.add_argument(
        "--ocr-mode",
        choices=["basic", "advanced", "ultra"],
        default="advanced",
        help="OCR strategy: ultra is slowest but strongest for difficult handwriting",
    )
    parser.add_argument("--mirror", action="store_true", help="Mirror frames")
    parser.add_argument("--save-dir", default="captures", help="Directory for saved results")
    parser.add_argument("--refresh-ms", default=80, type=int, help="GUI refresh interval in ms")
    parser.add_argument("--connect-timeout", default=3.0, type=float, help="HTTP connect timeout")
    parser.add_argument("--read-timeout", default=8.0, type=float, help="HTTP read timeout")
    return parser.parse_args()


def build_base_url(ip_or_url, default_port):
    raw = ip_or_url.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        host = parsed.hostname or raw
        port = parsed.port or default_port
    else:
        host = raw
        port = default_port
        if ":" in raw and raw.count(":") == 1:
            maybe_host, maybe_port = raw.split(":", 1)
            if maybe_port.isdigit():
                host = maybe_host
                port = int(maybe_port)
    return f"http://{host}:{port}"


def decode_jpeg_bytes(data):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_cuda():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def scale_result_bboxes(results, scale_x, scale_y):
    scaled = []
    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.float32)
        pts[:, 0] = pts[:, 0] / scale_x
        pts[:, 1] = pts[:, 1] / scale_y
        scaled.append((pts.tolist(), text, conf))
    return scaled


def normalize_ocr_results(results):
    normalized = []
    if results is None:
        return normalized

    for item in results:
        if not isinstance(item, (list, tuple)):
            continue

        if len(item) >= 3:
            bbox, text, conf = item[0], item[1], item[2]
        elif len(item) == 2:
            bbox, text = item[0], item[1]
            conf = 0.55
        else:
            continue

        text = str(text)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0

        normalized.append((bbox, text, conf))

    return normalized


def readtext_safe(reader, image, **kwargs):
    try:
        results = reader.readtext(image, **kwargs)
    except Exception:
        return []
    return normalize_ocr_results(results)


class ESP32Client:
    def __init__(self, base_url, stream_path, snapshot_paths, connect_timeout, read_timeout, mirror=False):
        self.base_url = base_url
        self.stream_url = f"{base_url}{stream_path}"
        self.snapshot_urls = [f"{base_url}{p.strip()}" for p in snapshot_paths if p.strip()]
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.mirror = mirror
        self.active_snapshot_url = self.snapshot_urls[0] if self.snapshot_urls else None

    def _post(self, frame):
        if frame is None:
            return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def get_snapshot(self):
        if not self.snapshot_urls:
            return None

        ordered = []
        if self.active_snapshot_url:
            ordered.append(self.active_snapshot_url)
        ordered.extend([u for u in self.snapshot_urls if u != self.active_snapshot_url])

        for url in ordered:
            try:
                resp = requests.get(url, timeout=(self.connect_timeout, self.read_timeout))
                if resp.status_code != 200:
                    continue
                frame = decode_jpeg_bytes(resp.content)
                if frame is None:
                    continue
                self.active_snapshot_url = url
                return self._post(frame)
            except requests.RequestException:
                continue
        return None

    def stream_frames(self, stop_event):
        while not stop_event.is_set():
            try:
                with requests.get(
                    self.stream_url,
                    stream=True,
                    timeout=(self.connect_timeout, self.read_timeout),
                ) as resp:
                    if resp.status_code != 200:
                        time.sleep(0.6)
                        continue

                    buf = b""
                    for chunk in resp.iter_content(chunk_size=1024):
                        if stop_event.is_set():
                            break
                        if not chunk:
                            continue

                        buf += chunk
                        a = buf.find(b"\xff\xd8")
                        b = buf.find(b"\xff\xd9")
                        if a != -1 and b != -1 and b > a:
                            jpg = buf[a : b + 2]
                            buf = buf[b + 2 :]
                            frame = decode_jpeg_bytes(jpg)
                            frame = self._post(frame)
                            if frame is not None:
                                yield frame

                        if len(buf) > 2_000_000:
                            buf = buf[-256_000:]

            except requests.RequestException:
                time.sleep(0.8)


def preprocess_for_handwriting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=60, sigmaSpace=60)

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=8,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.dilate(binary, kernel, iterations=1)

    return processed, denoised


def merge_ocr_results(*result_sets):
    merged = {}
    for results in result_sets:
        for bbox, text, conf in normalize_ocr_results(results):
            cleaned = " ".join(text.strip().split())
            if not cleaned or conf < 0.22:
                continue

            pts = np.array(bbox, dtype=np.float32)
            center = tuple(np.round(pts.mean(axis=0) / 20).astype(int))
            key = (cleaned.lower(), center)
            if key not in merged or conf > merged[key][2]:
                merged[key] = (bbox, cleaned, conf)

    return sorted(
        merged.values(),
        key=lambda item: (
            int(np.array(item[0], dtype=np.float32)[:, 1].min()),
            int(np.array(item[0], dtype=np.float32)[:, 0].min()),
        ),
    )


def run_ocr_basic(reader, frame):
    processed, cursive = preprocess_for_handwriting(frame)

    pass_1 = readtext_safe(
        reader,
        processed,
        detail=1,
        paragraph=False,
        decoder="greedy",
        contrast_ths=0.05,
        adjust_contrast=0.9,
        text_threshold=0.25,
        low_text=0.15,
        link_threshold=0.15,
        width_ths=0.8,
        height_ths=0.5,
    )

    pass_2 = readtext_safe(
        reader,
        cursive,
        detail=1,
        paragraph=True,
        decoder="beamsearch",
        contrast_ths=0.05,
        adjust_contrast=0.9,
        text_threshold=0.20,
        low_text=0.10,
        link_threshold=0.10,
        width_ths=0.8,
        height_ths=0.5,
    )

    return merge_ocr_results(pass_1, pass_2), processed


def run_ocr_advanced(reader, frame):
    processed, cursive = preprocess_for_handwriting(frame)

    blur = cv2.GaussianBlur(cursive, (0, 0), 1.1)
    sharpen = cv2.addWeighted(cursive, 1.7, blur, -0.7, 0)

    inverted = cv2.bitwise_not(processed)

    scale = 1.6
    upscaled = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    r1 = readtext_safe(
        reader,
        processed,
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        contrast_ths=0.05,
        adjust_contrast=0.9,
        text_threshold=0.22,
        low_text=0.12,
        link_threshold=0.14,
        width_ths=0.8,
        height_ths=0.5,
    )

    r2 = readtext_safe(
        reader,
        sharpen,
        detail=1,
        paragraph=True,
        decoder="beamsearch",
        contrast_ths=0.05,
        adjust_contrast=0.9,
        text_threshold=0.18,
        low_text=0.08,
        link_threshold=0.10,
        width_ths=0.9,
        height_ths=0.5,
    )

    r3 = readtext_safe(
        reader,
        processed,
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        rotation_info=[0, 90, 270],
        text_threshold=0.20,
        low_text=0.09,
        link_threshold=0.10,
        width_ths=0.9,
        height_ths=0.5,
    )

    r4 = readtext_safe(
        reader,
        upscaled,
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        text_threshold=0.21,
        low_text=0.10,
        link_threshold=0.12,
        width_ths=0.9,
        height_ths=0.5,
    )
    r4 = scale_result_bboxes(r4, scale_x=scale, scale_y=scale)

    r5 = readtext_safe(
        reader,
        inverted,
        detail=1,
        paragraph=False,
        decoder="greedy",
        text_threshold=0.20,
        low_text=0.10,
        link_threshold=0.12,
        width_ths=0.9,
        height_ths=0.5,
    )

    return merge_ocr_results(r1, r2, r3, r4, r5), processed


def run_ocr_ultra(reader, frame):
    processed, cursive = preprocess_for_handwriting(frame)

    blur = cv2.GaussianBlur(cursive, (0, 0), 1.0)
    sharpen = cv2.addWeighted(cursive, 1.9, blur, -0.9, 0)

    inverted = cv2.bitwise_not(processed)
    otsu = cv2.threshold(cursive, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(processed, cv2.MORPH_OPEN, close_kernel, iterations=1)
    closed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    scale = 2.0
    upscaled = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    passes = []
    passes.append(
        readtext_safe(
            reader,
            processed,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            text_threshold=0.20,
            low_text=0.08,
            link_threshold=0.10,
            width_ths=0.9,
            height_ths=0.5,
        )
    )
    passes.append(
        readtext_safe(
            reader,
            sharpen,
            detail=1,
            paragraph=True,
            decoder="beamsearch",
            text_threshold=0.16,
            low_text=0.06,
            link_threshold=0.08,
            width_ths=0.9,
            height_ths=0.5,
        )
    )
    passes.append(
        readtext_safe(
            reader,
            upscaled,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            text_threshold=0.16,
            low_text=0.06,
            link_threshold=0.08,
            width_ths=1.0,
            height_ths=0.5,
            mag_ratio=1.2,
        )
    )
    passes[-1] = scale_result_bboxes(passes[-1], scale_x=scale, scale_y=scale)

    passes.append(
        readtext_safe(
            reader,
            otsu,
            detail=1,
            paragraph=False,
            decoder="greedy",
            text_threshold=0.18,
            low_text=0.06,
            link_threshold=0.09,
            width_ths=1.0,
            height_ths=0.5,
        )
    )
    passes.append(
        readtext_safe(
            reader,
            opened,
            detail=1,
            paragraph=False,
            decoder="greedy",
            text_threshold=0.18,
            low_text=0.06,
            link_threshold=0.09,
            width_ths=1.0,
            height_ths=0.5,
        )
    )
    passes.append(
        readtext_safe(
            reader,
            closed,
            detail=1,
            paragraph=False,
            decoder="greedy",
            text_threshold=0.18,
            low_text=0.06,
            link_threshold=0.09,
            width_ths=1.0,
            height_ths=0.5,
        )
    )
    passes.append(
        readtext_safe(
            reader,
            inverted,
            detail=1,
            paragraph=False,
            decoder="greedy",
            text_threshold=0.18,
            low_text=0.06,
            link_threshold=0.09,
            width_ths=1.0,
            height_ths=0.5,
        )
    )

    return merge_ocr_results(*passes), processed


def draw_results(frame, ocr_results):
    display = frame.copy()
    lines = []

    for bbox, text, conf in ocr_results:
        if conf < 0.15:
            continue

        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(display, [pts], True, (0, 200, 0), 2)
        x, y = int(pts[0][0]), int(pts[0][1]) - 8
        label = f"{text} ({conf:.0%})"
        cv2.putText(display, label, (x, max(y, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)
        lines.append((text, conf))

    return display, lines


def save_result(save_dir, frame, annotated, processed, text):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)

    raw_path = os.path.join(save_dir, f"{ts}_raw.jpg")
    ann_path = os.path.join(save_dir, f"{ts}_annotated.jpg")
    proc_path = os.path.join(save_dir, f"{ts}_processed.jpg")
    txt_path = os.path.join(save_dir, f"{ts}_text.txt")

    cv2.imwrite(raw_path, frame)
    cv2.imwrite(ann_path, annotated)
    cv2.imwrite(proc_path, processed)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return raw_path, ann_path, proc_path, txt_path


def save_processed_outputs(save_dir, ts, annotated, processed, text):
    os.makedirs(save_dir, exist_ok=True)
    ann_path = os.path.join(save_dir, f"{ts}_annotated.jpg")
    proc_path = os.path.join(save_dir, f"{ts}_processed.jpg")
    txt_path = os.path.join(save_dir, f"{ts}_text.txt")
    text_to_write = text.strip() if text and text.strip() else "No text recognized"

    cv2.imwrite(ann_path, annotated)
    cv2.imwrite(proc_path, processed)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_to_write)

    return ann_path, proc_path, txt_path


class TkinterOCRApp:
    def __init__(self, root, client, reader, save_dir, refresh_ms, ocr_mode):
        import tkinter as tk
        from tkinter import scrolledtext

        self.tk = tk
        self.root = root
        self.client = client
        self.reader = reader
        self.save_dir = save_dir
        self.refresh_ms = max(refresh_ms, 30)
        self.ocr_mode = ocr_mode

        self.root.title("ESP32-CAM Handwriting OCR")
        self.root.geometry("1400x950")

        top = tk.Frame(root)
        top.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 6))

        live_wrap = tk.LabelFrame(top, text="Live Feed", padx=6, pady=6)
        live_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        right_panel = tk.Frame(top)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        cap_wrap = tk.LabelFrame(right_panel, text="Captured Frame (Processed)", padx=6, pady=6)
        cap_wrap.pack(fill=tk.BOTH, expand=True, padx=0)

        self.live_label = tk.Label(live_wrap, text="Connecting to stream...", bg="#121212", fg="#dedede")
        self.live_label.pack(fill=tk.BOTH, expand=True)

        self.capture_label = tk.Label(cap_wrap, text="Capture + Analyze to preview", bg="#121212", fg="#dedede")
        self.capture_label.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(root)
        controls.pack(fill=tk.X, padx=12, pady=6)

        self.capture_btn = tk.Button(controls, text="Capture + Analyze", width=18, command=self.capture_and_detect, bg="#00cc88", fg="#000", font=("Arial", 10, "bold"))
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.save_btn = tk.Button(controls, text="Save Last Result", width=16, command=self.save_last)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(controls, text="Quit", width=10, command=self.on_close).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value=f"Stream: {self.client.stream_url}")
        tk.Label(root, textvariable=self.status_var, anchor="w", padx=12, fg="#0a7").pack(fill=tk.X)

        self.text_box = scrolledtext.ScrolledText(root, height=5, wrap=tk.WORD, bg="#1a1a1a", fg="#0f0", font=("Courier New", 9))
        self.text_box.pack(fill=tk.BOTH, expand=False, padx=12, pady=(0, 12))
        self.text_box.insert(tk.END, "Detailed OCR output will appear here.\n")

        self.live_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()

        self.ocr_running = False
        self.live_img = None
        self.capture_img = None
        self.current_frame = None
        self.last_raw = None
        self.last_annotated = None
        self.last_processed = None
        self.last_text = ""
        self.last_saved_paths = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._pump_live_frames()

    def _stream_worker(self):
        for frame in self.client.stream_frames(self.stop_event):
            if self.stop_event.is_set():
                break
            self._push_live(frame)

        while not self.stop_event.is_set():
            frame = self.client.get_snapshot()
            if frame is not None:
                self._push_live(frame)
            time.sleep(0.3)

    def _push_live(self, frame):
        try:
            while True:
                self.live_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.live_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _render_for_label(self, frame, max_w=600, max_h=420):
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        w, h = image.size
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return ImageTk.PhotoImage(image=image)

    def _pump_live_frames(self):
        if self.stop_event.is_set():
            return

        latest = None
        while True:
            try:
                latest = self.live_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            self.current_frame = latest
            self.live_img = self._render_for_label(latest)
            self.live_label.configure(image=self.live_img, text="")
            self.status_var.set(f"Live feed OK ({self.client.stream_url})")
        else:
            self.status_var.set("Waiting for live feed from ESP32...")

        self.root.after(self.refresh_ms, self._pump_live_frames)

    def capture_and_detect(self):
        if self.ocr_running:
            return

        self.ocr_running = True
        self.capture_btn.configure(state=self.tk.DISABLED)
        self.status_var.set("Capturing image...")

        frame = self.current_frame.copy() if self.current_frame is not None else self.client.get_snapshot()
        if frame is None:
            self._on_ocr_failed("Capture failed. Check ESP32 stream/network.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)
        raw_path = os.path.join(self.save_dir, f"{ts}_raw.jpg")
        cv2.imwrite(raw_path, frame)

        self.status_var.set(f"Raw saved: {raw_path}. Running {self.ocr_mode} OCR...")
        if frame is not None:
            self.capture_img = self._render_for_label(frame)
            self.capture_label.configure(image=self.capture_img, text="")

        threading.Thread(target=self._ocr_worker, args=(frame, ts, raw_path), daemon=True).start()

    def _ocr_worker(self, initial_frame, ts, raw_path):
        t0 = time.time()
        frame = initial_frame if initial_frame is not None else self.client.get_snapshot()
        if frame is None:
            self.root.after(0, lambda: self._on_ocr_failed("Capture failed. Check ESP32 stream/network."))
            return

        try:
            if self.ocr_mode == "ultra":
                merged, processed = run_ocr_ultra(self.reader, frame)
            elif self.ocr_mode == "advanced":
                merged, processed = run_ocr_advanced(self.reader, frame)
            else:
                merged, processed = run_ocr_basic(self.reader, frame)

            annotated, lines = draw_results(frame, merged)
            text = " | ".join(t for t, c in lines if c >= 0.28)
            ann_path, proc_path, txt_path = save_processed_outputs(self.save_dir, ts, annotated, processed, text)
            saved_paths = (raw_path, ann_path, proc_path, txt_path)
            elapsed = time.time() - t0

            self.root.after(0, lambda: self._on_ocr_done(frame, annotated, processed, text, lines, elapsed, saved_paths))
        except Exception as e:
            fallback_text = f"OCR failed: {e}"
            ann_path, proc_path, txt_path = save_processed_outputs(self.save_dir, ts, frame, frame, fallback_text)
            self.root.after(0, lambda: self._on_ocr_failed(f"OCR failed. Saved error to {txt_path}"))

    def _on_ocr_failed(self, message):
        self.ocr_running = False
        self.capture_btn.configure(state=self.tk.NORMAL)
        self.status_var.set(message)

    def _on_ocr_done(self, raw, annotated, processed, text, lines, elapsed, saved_paths):
        self.ocr_running = False
        self.capture_btn.configure(state=self.tk.NORMAL)

        self.last_raw = raw
        self.last_annotated = annotated
        self.last_processed = processed
        self.last_text = text
        self.last_saved_paths = saved_paths

        self.capture_img = self._render_for_label(annotated)
        self.capture_label.configure(image=self.capture_img, text="")

        shown = text if text else "Image unclear or no text detected - try again"

        self.text_box.delete("1.0", self.tk.END)
        self.text_box.insert(self.tk.END, shown + "\n")
        for line, conf in lines:
            if conf >= 0.15:
                self.text_box.insert(self.tk.END, f"- {line} ({conf:.0%})\n")
        self.text_box.insert(self.tk.END, "\nSaved files:\n")
        self.text_box.insert(self.tk.END, f"- Raw: {saved_paths[0]}\n")
        self.text_box.insert(self.tk.END, f"- Annotated: {saved_paths[1]}\n")
        self.text_box.insert(self.tk.END, f"- Processed: {saved_paths[2]}\n")
        self.text_box.insert(self.tk.END, f"- Text: {saved_paths[3]}\n")

        self.status_var.set(f"OCR done in {elapsed:.2f}s. Saved to {saved_paths[0]}")

    def save_last(self):
        if self.last_raw is None or self.last_annotated is None or self.last_processed is None:
            self.status_var.set("Nothing to save. Click Capture + Analyze first.")
            return

        paths = save_result(
            self.save_dir,
            self.last_raw,
            self.last_annotated,
            self.last_processed,
            self.last_text,
        )
        self.status_var.set(f"Saved: {paths[0]}")

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    args = parse_args()
    base_url = build_base_url(args.ip, args.port)
    snapshot_paths = [p.strip() for p in args.snapshot_paths.split(",") if p.strip()]

    cuda_available = detect_cuda()
    use_gpu = args.gpu and cuda_available

    if args.gpu and not cuda_available:
        print("[WARN] --gpu requested, but CUDA is not available. Falling back to CPU.")
    elif use_gpu:
        print("[INFO] CUDA detected. EasyOCR will run on GPU.")

    print("[OCR] Loading EasyOCR model...")
    languages = [s.strip() for s in args.lang.split(",") if s.strip()]
    reader = easyocr.Reader(languages, gpu=use_gpu)
    print(f"[OCR] Ready (languages={languages}, gpu={use_gpu}, mode={args.ocr_mode})")

    client = ESP32Client(
        base_url=base_url,
        stream_path=args.stream_path,
        snapshot_paths=snapshot_paths,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        mirror=args.mirror,
    )

    try:
        import tkinter as tk
        from PIL import Image, ImageTk  # noqa: F401
    except Exception as e:
        print(f"[ERROR] Tkinter/Pillow not available: {e}")
        return

    print(f"[INFO] Base URL: {base_url}")
    print(f"[INFO] Stream URL: {client.stream_url}")

    root = tk.Tk()
    TkinterOCRApp(root, client, reader, args.save_dir, args.refresh_ms, args.ocr_mode)
    root.mainloop()


if __name__ == "__main__":
    main()
