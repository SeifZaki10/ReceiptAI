from pathlib import Path
import re

import cv2
import numpy as np
import torch
import streamlit as st

from ultralytics import YOLO
from paddleocr import TextRecognition
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_PATH = BASE_DIR / "models" / "best (3).pt"

T5_PATH = BASE_DIR / "models" / "final_t5_receipt_model_v2"


# =========================================================
# Load Models
# =========================================================

@st.cache_resource
def load_models():

    # YOLO text detector
    yolo_model = YOLO(str(YOLO_PATH))

    # PaddleOCR text recognizer
    recognizer = TextRecognition(
        model_name="en_PP-OCRv4_mobile_rec"
    )

    # T5 tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(
        str(T5_PATH)
    )

    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(T5_PATH)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    t5_model.to(device)
    t5_model.eval()

    return (
        yolo_model,
        recognizer,
        tokenizer,
        t5_model,
        device
    )


# Load once
yolo_model, recognizer, tokenizer, t5_model, device = load_models()


# =========================================================
# YOLO + PaddleOCR
# =========================================================

def get_prediction(image_path):

    # -----------------------------
    # YOLO Text Detection
    # -----------------------------

    result = yolo_model.predict(
        source=image_path,
        conf=0.4,
        imgsz=640,
        verbose=False
    )[0]

    image = result.orig_img

    boxes = result.boxes.xyxy.cpu().numpy()

    # -----------------------------
    # Sort boxes vertically
    # -----------------------------

    boxes = sorted(
        boxes,
        key=lambda box: (box[1] + box[3]) / 2
    )

    lines = []

    # -----------------------------
    # Group boxes into text lines
    # -----------------------------

    for box in boxes:

        x1, y1, x2, y2 = box

        cy = (y1 + y2) / 2
        height = y2 - y1

        placed = False

        for line in lines:

            if abs(cy - line["cy"]) < min(
                height,
                line["height"]
            ) * 0.5:

                line["boxes"].append(box)

                placed = True

                break

        if not placed:

            lines.append({
                "cy": cy,
                "height": height,
                "boxes": [box]
            })

    # -----------------------------
    # Sort lines top → bottom
    # -----------------------------

    lines.sort(
        key=lambda line: line["cy"]
    )

    predicted_lines = []

    # -----------------------------
    # PaddleOCR Recognition
    # -----------------------------

    for line in lines:

        # Sort words left → right
        line["boxes"].sort(
            key=lambda box: box[0]
        )

        line_texts = []

        for box in line["boxes"]:

            x1, y1, x2, y2 = map(int, box)

            # Keep crop inside image boundaries
            h, w = image.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            results = recognizer.predict(
                input=crop
            )

            for res in results:

                text = res["rec_text"]

                if text.strip():
                    line_texts.append(
                        text.strip()
                    )

        if line_texts:

            predicted_lines.append(
                " ".join(line_texts)
            )

    # Final OCR text
    predicted_text = "\n".join(
        predicted_lines
    )

    return predicted_text


# =========================================================
# Prepare T5 Input
# =========================================================

def prepare_input(text):

    tokens = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False
    )["input_ids"]

    # T5 maximum input length = 512
    # Reserve one token for EOS

    if len(tokens) <= 511:

        final_tokens = tokens

    else:

        # Keep beginning + end of receipt
        first_part = tokens[:350]

        last_part = tokens[-161:]

        final_tokens = (
            first_part + last_part
        )

    # Add EOS token
    final_tokens.append(
        tokenizer.eos_token_id
    )

    return final_tokens


# =========================================================
# T5 Information Extraction
# =========================================================

def extract_information(ocr_text):

    input_ids = prepare_input(
        ocr_text
    )

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long
    ).to(device)

    attention_mask = torch.ones_like(
        input_ids
    )

    with torch.no_grad():

        outputs = t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=96
        )

    output_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return output_text


# =========================================================
# Parse T5 Output
# =========================================================

def parse_output(text):

    fields = {
        "company": "",
        "address": "",
        "date": "",
        "total": ""
    }

    # Find:
    # company:
    # address:
    # date:
    # total:

    label_pattern = re.compile(
        r"(company|address|date|total)\s*:",
        re.IGNORECASE
    )

    matches = list(
        label_pattern.finditer(text)
    )

    for i, match in enumerate(matches):

        field_name = (
            match.group(1).lower()
        )

        value_start = match.end()

        if i + 1 < len(matches):

            value_end = (
                matches[i + 1].start()
            )

        else:

            value_end = len(text)

        value = text[
            value_start:value_end
        ].strip()

        fields[field_name] = value

    return fields


# =========================================================
# Complete Receipt Pipeline
# =========================================================

def process_receipt(image_path):

    # Step 1:
    # YOLO + PaddleOCR
    ocr_text = get_prediction(
        image_path
    )

    # Step 2:
    # T5 V2
    t5_output = extract_information(
        ocr_text
    )

    # Step 3:
    # Parse structured fields
    fields = parse_output(
        t5_output
    )

    # Step 4:
    # Return final result
    return {
        "company": fields["company"],
        "address": fields["address"],
        "date": fields["date"],
        "total": fields["total"],

        # Advanced details
        "ocr_text": ocr_text,
        "raw_t5_output": t5_output
    }