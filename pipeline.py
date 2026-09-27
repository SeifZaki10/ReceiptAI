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
T5_PATH = BASE_DIR / "models" / "final_t5_receipt_model"


# =========================================================
# Load Models
# =========================================================

@st.cache_resource
def load_models():

    yolo_model = YOLO(str(YOLO_PATH))

    recognizer = TextRecognition(
        model_name="en_PP-OCRv4_mobile_rec"
    )

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


(
    yolo_model,
    recognizer,
    tokenizer,
    t5_model,
    device
) = load_models()


# =========================================================
# YOLO + PaddleOCR
# =========================================================

def get_prediction(image_path):

    result = yolo_model.predict(
        source=image_path,
        conf=0.4,
        imgsz=640,
        verbose=False
    )[0]

    image = result.orig_img

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    # Sort boxes by vertical center
    boxes = sorted(
        boxes,
        key=lambda box: (box[1] + box[3]) / 2
    )

    lines = []

    for box in boxes:

        x1, y1, x2, y2 = box

        cy = (y1 + y2) / 2
        height = y2 - y1

        placed = False

        for line in lines:

            if (
                abs(cy - line["cy"])
                < min(height, line["height"]) * 0.5
            ):

                line["boxes"].append(box)

                placed = True

                break

        if not placed:

            lines.append({
                "cy": cy,
                "height": height,
                "boxes": [box]
            })

    # Sort lines from top to bottom
    lines.sort(
        key=lambda line: line["cy"]
    )

    predicted_lines = []

    h, w = image.shape[:2]

    for line in lines:

        # Sort boxes from left to right
        line["boxes"].sort(
            key=lambda box: box[0]
        )

        line_texts = []

        for box in line["boxes"]:

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # Keep coordinates inside image
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

    if len(tokens) <= 511:

        final_tokens = tokens

    else:

        first_part = tokens[:350]
        last_part = tokens[-161:]

        final_tokens = (
            first_part + last_part
        )

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

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result


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
# Rule-Based Total Extraction
# =========================================================

def extract_total_rule(ocr_text):

    patterns = [

        # Example:
        # GRAND TOTAL 678.30
        # GRAND TOTAL RM 678.30
        r"\bgrand\s*total\s*[:\-]?\s*"
        r"(?:RM|EGP|LE|\$)?\s*"
        r"(\d+(?:[.,]\d{2}))",

        # Example:
        # TOTAL 678.30
        # TOTAL: 678.30
        # TOTAL RM 678.30
        r"\btotal\s*[:\-]?\s*"
        r"(?:RM|EGP|LE|\$)?\s*"
        r"(\d+(?:[.,]\d{2}))"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            ocr_text,
            re.IGNORECASE
        )

        if match:

            total = match.group(1)

            total = total.replace(
                ",",
                "."
            )

            return total

    return ""


# =========================================================
# Rule-Based Date Extraction
# =========================================================

def extract_date_rule(ocr_text):

    patterns = [

        # DD/MM/YYYY or MM/DD/YYYY
        # Examples:
        # 25/12/2018
        # 09/24/2026
        r"\b(\d{1,2}/\d{1,2}/\d{4})",

        # DD-MM-YYYY or MM-DD-YYYY
        r"\b(\d{1,2}-\d{1,2}-\d{4})",

        # DD/MM/YY or MM/DD/YY
        r"\b(\d{1,2}/\d{1,2}/\d{2})",

        # DD-MM-YY or MM-DD-YY
        r"\b(\d{1,2}-\d{1,2}-\d{2})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            ocr_text,
            re.IGNORECASE
        )

        if match:

            return match.group(1)

    return ""


# =========================================================
# Complete Receipt Pipeline
# =========================================================

def process_receipt(image_path):

    # 1. YOLO + PaddleOCR
    ocr_text = get_prediction(
        image_path
    )

    # 2. T5 information extraction
    t5_output = extract_information(
        ocr_text
    )

    # 3. Parse T5 output
    fields = parse_output(
        t5_output
    )

    # -----------------------------------------------------
    # 4. Hybrid Rule-Based Fallbacks
    # -----------------------------------------------------

    # If T5 could not detect Total,
    # try extracting it directly from OCR text
    if not fields["total"]:

        fields["total"] = extract_total_rule(
            ocr_text
        )

    # If T5 could not detect Date,
    # try extracting it directly from OCR text
    if not fields["date"]:

        fields["date"] = extract_date_rule(
            ocr_text
        )

    # -----------------------------------------------------
    # 5. Return final results
    # -----------------------------------------------------

    return {

        "company": fields["company"],

        "address": fields["address"],

        "date": fields["date"],

        "total": fields["total"],

        "ocr_text": ocr_text,

        "raw_t5_output": t5_output
    }