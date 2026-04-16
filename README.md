# ESP32-CAM Handwriting OCR

Tkinter-based OCR app for an ESP32-CAM handwriting capture workflow. The Python app captures frames from the ESP32-CAM, runs OCR, and saves the raw image, processed image, annotated output, and recognized text into the `captures/` folder.

## Requirements

- Python 3.12+
- ESP32-CAM running the Arduino sketch in `ESP32CAM_Handwriting.ino`
- A working virtual environment in this project folder

## Initial Setup

If you are starting from a fresh clone, run:

```bash
git clone https://github.com/AyushDhabsa/Word-recogniton-using-esp32-cam.git
cd Word-recogniton-using-esp32-cam
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python easyocr numpy requests pillow torch
```

If you are already inside the project folder, just activate the virtual environment:

```bash
cd /home/ayush/Desktop/SOFTCOMPUTING
source .venv/bin/activate
```

## Run the GUI

Use the ESP32-CAM IP address shown by your board:

```bash
.venv/bin/python handwriting.py --ip http://10.86.253.22 --stream-path /stream --gpu --ocr-mode ultra
```

If you want a lighter mode, use `--ocr-mode advanced` or `--ocr-mode basic`.

## Output Files

Each capture saves these files into `captures/`:

- `*_raw.jpg`
- `*_processed.jpg`
- `*_annotated.jpg`
- `*_text.txt`

## Notes

- `--gpu` will only work if CUDA is available.
- If OCR is unstable, try `--ocr-mode ultra` first for difficult handwriting, or `--ocr-mode advanced` if you want faster runs.
- The `captures/` folder is kept in the repo, but generated images and text files are ignored by Git.