import io
import os
import re
import hashlib
import base64
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    torch = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None


# =====================================================
# APP SETTINGS
# =====================================================
st.set_page_config(page_title="PDF Field Extraction Tool", layout="wide")
CACHE_VERSION = "unified_three_layouts_v2_ui"
TEXT_OCR_ZOOM = 2.0
HANDWRITING_ZOOM = 2.0
HEADER_LOGO_BASE64 = "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"

ALL_FIELDS = [
    "Document Type",
    "Ship From",
    "Ship To",
    "Notify Party",
    "Date",
    "Our Ref. No.",
    "Vessel/Truck",
    "Carrier",
    "Carrier Name",
    "SCAC",
    "Loading Port",
    "Discharge Port",
    "Location",
    "Origin/Destination Port",
    "Sales Order No.",
    "Shipment Number",
    "Forwarding Agent",
    "Shipping Line",
    "B/L or AWB No.",
    "Bill of Lading",
    "Arrival/Departure Date",
    "Free Time Exp.",
    "Delivery Type",
    "House No.",
    "Entry-B/L No.",
    "Customer Ref. No.",
    "PO/PRO Number",
    "VRET ID",
    "RA Number",
    "Deliver To",
    "Shipping Marks",
    "Delivery Numbers",
    "Item Numbers",
    "No. of Packages",
    "Description of Articles",
    "Bill of Lading Description",
    "Pallets",
    "Cartons",
    "Units",
    "Weight",
    "Volume",
    "Total Charges",
    "FOB Value",
    "Currency",
    "Total Bundles",
    "Total Pieces",
    "Total Net Weight",
    "Total Gross Weight",
    "Bill of Lading Instructions",
    "HS Code",
    "Container No.",
    "Container Numbers",
    "Container Size/Type",
    "Container Types",
    "Booking Number",
    "Trailer Number",
    "Seal No.",
    "Seal Numbers",
    "Quantity",
    "Total Containers",
    "Handwritten Text",
    "Length",
    "Width",
    "Height",
]

st.markdown(
    """
    <style>
    .stApp { background: #F4F7FB; }
    .block-container { max-width: 1500px; padding-top: 1rem; padding-bottom: 3rem; }
    header[data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 { color: #082A46 !important; }

    .app-header {
        background: #0B2540;
        border-radius: 16px;
        padding: 20px 28px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        box-shadow: 0 8px 25px rgba(3, 26, 46, 0.24);
    }
    .brand-area { display: flex; align-items: center; gap: 16px; }
    .brand-divider { width: 1px; height: 58px; background: rgba(255,255,255,0.32); margin: 0 4px; flex: 0 0 1px; }
    .brand-icon {
        width: 175px;
        height: auto;
        display: block;
        object-fit: contain;
        background: #0B2540;
        mix-blend-mode: normal;
    }
    .brand-title { font-size: 23px; line-height: 1.1; font-weight: 750; }
    .brand-subtitle { color: #C9DFEC; font-size: 12.5px; margin-top: 5px; }
    .header-status {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 11px; padding: 9px 15px; min-width: 205px;
    }
    .status-label { color: #AFC9D9; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; }
    .status-value { color: white; font-size: 14px; font-weight: 650; margin-top: 3px; }

    .step-navigation {
        background: white; border: 1px solid #DEE7EF; border-radius: 14px;
        display: flex; align-items: center; padding: 5px 10px;
        margin-bottom: 22px; overflow-x: auto;
        box-shadow: 0 2px 8px rgba(8, 42, 70, 0.04);
    }
    .step-item {
        padding: 13px 22px; color: #627384; font-size: 14px;
        font-weight: 600; white-space: nowrap; border-bottom: 3px solid transparent;
    }
    .step-item.active { color: #0878B9; border-bottom-color: #1598C6; }
    a.step-item { text-decoration: none !important; }
    a.step-item:hover { color: #0878B9; background: #F7FBFD; }
    .step-number {
        display: inline-flex; align-items: center; justify-content: center;
        width: 24px; height: 24px; margin-right: 7px; border-radius: 50%;
        background: #E8F5FA; color: #0878B9; font-size: 12px; font-weight: 750;
    }
    .step-item.active .step-number { color: white; background: #1598C6; }

    .section-title { color: #082A46; font-size: 20px; font-weight: 750; margin-bottom: 4px; }
    .section-description { color: #6A7B8C; font-size: 13px; margin-bottom: 18px; }
    .content-card {
        background: white; border: 1px solid #DEE7EF; border-radius: 15px;
        padding: 23px 24px; margin-bottom: 18px;
        box-shadow: 0 3px 12px rgba(8, 42, 70, 0.05);
    }
    [data-testid="stFileUploader"] {
        background: #F9FCFE; border: 2px dashed #55B4D5;
        border-radius: 13px; padding: 12px;
    }
    [data-testid="stFileUploader"]:hover { border-color: #0878B9; background: #F2FAFD; }
    div[data-baseweb="input"] > div { border-radius: 9px; border-color: #D5E1E9; }

    .stButton > button {
        background: linear-gradient(135deg, #0878B9, #149FC7);
        color: white; border: none; border-radius: 9px; min-height: 44px;
        padding: 0 22px; font-weight: 650;
        box-shadow: 0 4px 12px rgba(8, 120, 185, 0.20);
    }
    .stButton > button:hover {
        color: white; border: none;
        background: linear-gradient(135deg, #07699F, #0D89AD);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998E, #18B8A5);
        color: white; border: none; border-radius: 9px; min-height: 46px;
        padding: 0 22px; font-weight: 650;
        box-shadow: 0 4px 12px rgba(17, 153, 142, 0.20);
    }
    .stDownloadButton > button:hover {
        color: white; border: none;
        background: linear-gradient(135deg, #0D857C, #13A594);
    }
    div[data-testid="metric-container"] {
        background: white; border: 1px solid #DEE7EF; border-radius: 14px;
        padding: 17px 18px; box-shadow: 0 3px 10px rgba(8, 42, 70, 0.04);
    }
    div[data-testid="stMetricValue"] { color: #0878B9; font-weight: 750; }
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
        background: white; border: 1px solid #DEE7EF; border-radius: 12px; overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #EDF3F7; border-radius: 11px; padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 43px; padding: 0 16px; border-radius: 8px;
        color: #536779; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: white !important; color: #0878B9 !important;
        box-shadow: 0 2px 8px rgba(8, 42, 70, 0.08);
    }
    div[role="radiogroup"] { background: #F0F5F8; border-radius: 10px; padding: 6px 10px; }
    .preview-heading {
        display: flex; align-items: center; gap: 8px; color: #082A46;
        font-size: 18px; font-weight: 750; margin-bottom: 12px;
    }
    .app-footer {
        margin-top: 32px; padding: 16px; border-top: 1px solid #DCE5EC;
        color: #758695; text-align: center; font-size: 12px;
    }
    @media (max-width: 800px) {
        .app-header { align-items: flex-start; flex-direction: column; gap: 16px; }
        .header-status { width: 100%; }
        .step-item { padding: 11px 13px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="app-header">
        <div class="brand-area">
            <img class="brand-icon" src="data:image/png;base64,{HEADER_LOGO_BASE64}" alt="MAERSK">
            <div class="brand-divider"></div>
            <div>
                <div class="brand-title">SmartDoc Extractor</div>
                <div class="brand-subtitle">Intelligent shipping and logistics document extraction</div>
            </div>
        </div>
        <div class="header-status">
            <div class="status-label">Extraction Engine</div>
            <div class="status-value">Multi-layout Auto Detection</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# GENERAL HELPERS
# =====================================================
def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip(" :-|\t\r\n")


def multiline_clean(value):
    lines = [clean(line) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line)


def empty_fields():
    return {field: "" for field in ALL_FIELDS}


def normalize_label(value):
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def detect_document_type(text):
    upper = str(text or "").upper()
    if "SHIPPING INSTRUCTION / INVOICE" in upper:
        return "Shipping Instruction / Invoice"
    if "DELIVERY ORDER" in upper:
        return "Delivery Order"
    if "BILL OF LADING" in upper or "BILL OFLADING" in upper:
        return "Bill of Lading"
    if "INVOICE" in upper:
        return "Invoice"
    return "PDF Document"


def first_match(text, patterns, flags=re.IGNORECASE | re.MULTILINE):
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return clean(match.group(1))
    return ""


def block_between(text, start_pattern, stop_patterns):
    start = re.search(start_pattern, text, re.IGNORECASE)
    if not start:
        return ""
    remaining = text[start.end():]
    positions = []
    for pattern in stop_patterns:
        stop = re.search(pattern, remaining, re.IGNORECASE)
        if stop:
            positions.append(stop.start())
    end = min(positions) if positions else len(remaining)
    return multiline_clean(remaining[:end])


def render_page_rgb(page, zoom):
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False
    )
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    ).copy()


# =====================================================
# PRINTED OCR WITH POSITIONS
# =====================================================
@st.cache_resource(show_spinner="Loading printed-text OCR...")
def get_rapidocr():
    if RapidOCR is None:
        return None
    try:
        threads = max(1, min(4, os.cpu_count() or 2))
        return RapidOCR(intra_op_num_threads=threads)
    except Exception:
        return RapidOCR()


def run_rapidocr(page):
    engine = get_rapidocr()
    if engine is None:
        return "", []

    image = render_page_rgb(page, TEXT_OCR_ZOOM)
    result, _ = engine(image, use_cls=False)
    lines = []

    for item in result or []:
        try:
            box, text, confidence = item[0], clean(item[1]), float(item[2])
            if not text:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            lines.append(
                {
                    "x0": min(xs),
                    "y0": min(ys),
                    "x1": max(xs),
                    "y1": max(ys),
                    "cx": (min(xs) + max(xs)) / 2,
                    "cy": (min(ys) + max(ys)) / 2,
                    "text": text,
                    "confidence": confidence,
                    "page_width": image.shape[1],
                    "page_height": image.shape[0],
                }
            )
        except (TypeError, ValueError, IndexError):
            continue

    lines.sort(key=lambda line: (line["cy"], line["x0"]))
    return "\n".join(line["text"] for line in lines), lines


def find_positioned_label(lines, alternatives):
    targets = [normalize_label(value) for value in alternatives]
    for line in lines:
        compact = normalize_label(line["text"])
        if any(target in compact for target in targets):
            return line
    return None


def lines_in_region(lines, x0, y0, x1, y1):
    selected = [
        line for line in lines
        if x0 <= line["cx"] < x1 and y0 <= line["cy"] < y1
    ]
    selected.sort(key=lambda line: (line["cy"], line["x0"]))
    return selected


def region_text(lines, x0, y0, x1, y1, exclusions=()):
    output = []
    exclusion_targets = [normalize_label(value) for value in exclusions]
    for line in lines_in_region(lines, x0, y0, x1, y1):
        compact = normalize_label(line["text"])
        if any(target in compact for target in exclusion_targets):
            continue
        output.append(line["text"])
    return "\n".join(output)


def inline_value(text, labels):
    for label in labels:
        match = re.search(
            rf"(?im)^\s*{label}\s*[:#-]?\s*(.*?)\s*$",
            text,
        )
        if match and clean(match.group(1)):
            return clean(match.group(1))
    return ""


# =====================================================
# AMAZON / MAERSK SCANNED BILL OF LADING
# =====================================================
def extract_amazon_bol(text, lines):
    fields = empty_fields()
    fields["Document Type"] = "Bill of Lading"
    if not lines:
        return fields

    page_width = lines[0]["page_width"]
    page_height = lines[0]["page_height"]
    left_boundary = page_width * 0.52

    ship_from_label = find_positioned_label(lines, ["SHIP FROM", "SHIPFROM"])
    ship_to_label = find_positioned_label(lines, ["SHIP TO", "SHIPTO"])
    vret_label = find_positioned_label(lines, ["VRET ID", "VRETID"])

    if ship_from_label and ship_to_label:
        fields["Ship From"] = region_text(
            lines,
            0,
            ship_from_label["y1"] + 1,
            left_boundary,
            ship_to_label["y0"] - 1,
            exclusions=("Bill of Lading", "ID"),
        )

    if ship_to_label:
        ship_to_end = (
            vret_label["y0"] - 1
            if vret_label and vret_label["y0"] > ship_to_label["y1"]
            else ship_to_label["y1"] + page_height * 0.16
        )
        fields["Ship To"] = region_text(
            lines,
            0,
            ship_to_label["y1"] + 1,
            left_boundary,
            ship_to_end,
            exclusions=(
                "Carrier Name", "SCAC", "PRO Number", "Trailer Number",
                "Seal Number", "VRET ID", "RA Number",
            ),
        )

    fields["Carrier Name"] = inline_value(text, [r"CARRIER\s*NAME"])
    fields["Carrier"] = fields["Carrier Name"]
    fields["SCAC"] = inline_value(text, [r"SCAC"])
    fields["PO/PRO Number"] = inline_value(
        text, [r"PRO\s*NUMBER", r"PRC\s*NUMBER", r"PO\s*NUMBER"]
    )
    fields["Trailer Number"] = inline_value(text, [r"TRAILER\s*NUMBER"])
    fields["Seal No."] = inline_value(text, [r"SEAL\s*NUMBER"])
    fields["VRET ID"] = inline_value(text, [r"VRET\s*ID"])
    fields["RA Number"] = inline_value(text, [r"RA\s*NUMBER"])

    uuid = re.search(
        r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
        text,
        re.IGNORECASE,
    )
    if uuid:
        fields["Bill of Lading"] = uuid.group(0)

    # Preserve the source's printed date including time, but keep it as one clean value.
    date = re.search(
        r"DATE\s*:\s*((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
        r"\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+UTC\s+\d{4})",
        text,
        re.IGNORECASE,
    )
    if date:
        fields["Date"] = clean(date.group(1))

    weight = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*pounds\b", text, re.IGNORECASE)
    volume = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*cuft\b", text, re.IGNORECASE)
    fields["Weight"] = f"{weight.group(1)} pounds" if weight else ""
    fields["Volume"] = f"{volume.group(1)} cuft" if volume else ""

    # Numeric table values located between labels and the legal text.
    table_start = find_positioned_label(lines, ["STACKABLE"])
    legal_start = find_positioned_label(lines, ["If the shipment moves"])
    if table_start:
        max_y = legal_start["y0"] if legal_start else table_start["y1"] + page_height * 0.07
        numeric_lines = lines_in_region(lines, 0, table_start["y1"], page_width, max_y)
        numbers = [line["text"] for line in numeric_lines if re.fullmatch(r"[0-9]+", line["text"])]
        if len(numbers) >= 4:
            fields["Pallets"] = numbers[0]
            fields["Cartons"] = numbers[2]
            fields["Units"] = numbers[3]

    fields["Total Charges"] = "USD" if re.search(r"\bUSD\b", text) else ""
    return fields


# =====================================================
# FLEXPORT DELIVERY ORDER
# =====================================================
def pdf_words_in_rect(words, x0, y0, x1, y1):
    selected = []
    for word in words:
        wx0, wy0, wx1, wy1 = word[:4]
        cx = (wx0 + wx1) / 2
        cy = (wy0 + wy1) / 2
        if x0 <= cx < x1 and y0 <= cy < y1:
            selected.append(word)
    selected.sort(key=lambda word: (word[1], word[0]))
    return clean(" ".join(word[4] for word in selected))


def extract_flexport_delivery_order(words, page_width, page_height):
    fields = empty_fields()
    fields["Document Type"] = "Delivery Order"
    sx = page_width / 612.0
    sy = page_height / 792.0

    def box(x0, y0, x1, y1):
        return pdf_words_in_rect(words, x0*sx, y0*sy, x1*sx, y1*sy)

    fields["Ship From"] = box(190, 88, 455, 128)
    fields["Date"] = box(385, 139, 445, 160)
    fields["Our Ref. No."] = box(465, 139, 535, 160)
    fields["Carrier"] = box(35, 204, 255, 222)
    fields["Location"] = box(255, 204, 405, 222)
    fields["Origin/Destination Port"] = box(405, 204, 612, 222)
    fields["B/L or AWB No."] = box(35, 224, 170, 242)
    fields["Arrival/Departure Date"] = box(170, 224, 238, 242)
    fields["Free Time Exp."] = box(238, 224, 305, 242)
    fields["Delivery Type"] = box(320, 224, 545, 242)
    fields["House No."] = box(270, 244, 390, 262)
    fields["Entry-B/L No."] = box(390, 244, 490, 262)
    fields["Customer Ref. No."] = box(545, 244, 612, 262)
    fields["Deliver To"] = box(35, 268, 260, 310)
    fields["Ship To"] = fields["Deliver To"]
    fields["No. of Packages"] = box(35, 379, 108, 398)
    fields["Description of Articles"] = box(108, 379, 455, 398)
    fields["Weight"] = box(480, 379, 550, 398)
    fields["Container No."] = box(105, 409, 195, 430)
    fields["Container Size/Type"] = clean(
        f"{box(195, 409, 235, 430)} {box(235, 409, 300, 430)}"
    )
    fields["Quantity"] = box(350, 409, 410, 430)
    fields["Seal No."] = box(410, 409, 480, 430)
    return fields


# =====================================================
# EGA SHIPPING INSTRUCTION / INVOICE
# =====================================================
def extract_ega_page(text, page_number):
    fields = empty_fields()
    fields["Document Type"] = "Shipping Instruction / Invoice"

    fields["Ship From"] = block_between(
        text,
        r"\bSHIPPER\s*:",
        [r"\bCONSIGNEE1?\s*:", r"\bNOTIFY\s*:", r"\bShipping Information\b"],
    )
    fields["Ship To"] = block_between(
        text,
        r"\bCONSIGNEE1?\s*:",
        [r"\bNOTIFY\s*:", r"\bDetails\b", r"\bShipping Information\b"],
    )
    fields["Notify Party"] = block_between(
        text,
        r"\bNOTIFY\s*:",
        [r"\bDetails\b", r"\bShipping Marks\b"],
    )

    mapping = {
        "Date": [r"\bDate\s+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"],
        "Vessel/Truck": [r"\bVessel/Truck\s+([^\n]+)"],
        "Loading Port": [r"\bLoading Port\s+([^\n]+)"],
        "Discharge Port": [r"\bDischarge Port\s+([^\n]+)"],
        "Sales Order No.": [r"\bSales order no\s+([A-Z0-9-]+)"],
        "Shipment Number": [r"\bShipment number\s+([A-Z0-9-]+)"],
        "Forwarding Agent": [r"\bForwarding Agent\s+(.+?)(?:\n\s*Shipping line|$)"],
        "Shipping Line": [r"\bShipping line\s+([^\n]+)"],
        "HS Code": [r"H\.S\.\s*Code\s*No\s*:\s*([0-9]+)"],
    }
    for field, patterns in mapping.items():
        fields[field] = first_match(text, patterns, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    fields["Delivery Numbers"] = "; ".join(
        dict.fromkeys(re.findall(r"(?<!\d)(807\d{5})(?!\d)", text))
    )
    fields["Item Numbers"] = "; ".join(
        dict.fromkeys(re.findall(r"(?<!\d)(0000\d{2})(?!\d)", text))
    )
    fields["Bill of Lading Description"] = first_match(
        text, [r"Bill of Lading Description:\s*([^\n]+)"]
    )

    if re.search(r"Puerto Cortes\s+Made in United Arab Emirates", text, re.IGNORECASE):
        fields["Shipping Marks"] = "Puerto Cortes; Made in United Arab Emirates"

    fob = re.search(r"FOB VALUE:\s*([0-9,]+(?:\.[0-9]+)?)\s*([A-Z]{3})", text, re.IGNORECASE)
    if fob:
        fields["FOB Value"] = fob.group(1)
        fields["Currency"] = fob.group(2).upper()

    total = re.search(
        r"Total\s+(\d+)\s+Bundles\s+Containing\s+(\d+)\s+pieces\s+"
        r"([0-9.]+)\s*MT\s+([0-9.]+)\s*MT",
        text,
        re.IGNORECASE,
    )
    if total:
        fields["Total Bundles"] = total.group(1)
        fields["Total Pieces"] = total.group(2)
        fields["Total Net Weight"] = f"{total.group(3)} MT"
        fields["Total Gross Weight"] = f"{total.group(4)} MT"

    fields["Bill of Lading Instructions"] = block_between(
        text,
        r"\bBill of Lading Instructions\b",
        [r"\bH\.S\.Code\b", r"\bContainer details\b", r"\bPage:"],
    )

    containers = re.findall(r"\b([A-Z]{4}\d{7})\b", text)
    types = re.findall(r"\b(\d{2}FT)\b", text, re.IGNORECASE)
    seals = re.findall(r"\b(MLAE\d+)\b", text, re.IGNORECASE)
    fields["Container Numbers"] = "; ".join(dict.fromkeys(v.upper() for v in containers))
    fields["Container Types"] = "; ".join(dict.fromkeys(v.upper() for v in types))
    fields["Seal Numbers"] = "; ".join(dict.fromkeys(v.upper() for v in seals))
    total_containers = re.search(r"Total\s+(\d+)\s+Containers", text, re.IGNORECASE)
    if total_containers:
        fields["Total Containers"] = total_containers.group(1)
    return fields


def extract_ega_items(text, page_number):
    rows = []
    description = first_match(text, [r"Bill of Lading Description:\s*([^\n]+)"])
    pattern = re.compile(
        r"(?<!\d)(807\d{5})\s+(\d{6})\s+([0-9.]+)\s*MT\s+([0-9.]+)\s*MT",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "Page Number": page_number,
                "Delivery No.": match.group(1),
                "Item No.": match.group(2),
                "Bundles": 8,
                "Pieces": 40,
                "Net Weight": f"{match.group(3)} MT",
                "Gross Weight": f"{match.group(4)} MT",
                "Shipping Port": "Puerto Cortes",
                "Country of Origin": "United Arab Emirates",
                "Bill of Lading Description": description,
            }
        )
    return rows


def extract_ega_containers(text, page_number):
    rows = []
    pattern = re.compile(
        r"(?<!\d)(807\d{5})\s+([A-Z]{4}\d{7})\s+(\d{2}FT)\s+(\d+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([A-Z0-9]+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "Page Number": page_number,
                "Delivery No.": match.group(1),
                "Container No.": match.group(2).upper(),
                "Container Type": match.group(3).upper(),
                "Bundles": match.group(4),
                "Bundle Weight": f"{match.group(5)} MT",
                "Tare Weight": f"{match.group(6)} MT",
                "Gross Weight": f"{match.group(7)} MT",
                "Seal No.": match.group(8).upper(),
            }
        )
    return rows


# =====================================================
# OPTIONAL HANDWRITING OCR
# =====================================================
DIMENSION_PATTERN = re.compile(
    r"(?<!\d)(\d{1,4})\s*[xX×*]\s*(\d{1,4})\s*[xX×*]\s*(\d{1,4})(?!\d)"
)


def normalize_dimension(value):
    match = DIMENSION_PATTERN.search(str(value or "").upper())
    if not match:
        return ""
    values = tuple(int(value) for value in match.groups())
    if not all(1 <= value <= 500 for value in values):
        return ""
    return "X".join(str(value) for value in values)


@st.cache_resource(show_spinner="Loading handwriting OCR model...")
def get_trocr():
    if torch is None or TrOCRProcessor is None or VisionEncoderDecoderModel is None:
        return None, None, None
    local_model = Path(__file__).resolve().parent / "models" / "trocr-small-handwritten"
    model_name = str(local_model) if local_model.is_dir() else "microsoft/trocr-small-handwritten"
    local_only = local_model.is_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(model_name, local_files_only=local_only)
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name, local_files_only=local_only
    ).to(device).eval()
    return processor, model, device


def extract_handwriting(pdf_bytes, page_number):
    processor, model, device = get_trocr()
    if processor is None:
        return ""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        image = render_page_rgb(document[page_number - 1], HANDWRITING_ZOOM)
    finally:
        document.close()
    height, width = image.shape[:2]
    crop = image[int(height * 0.56):, int(width * 0.03):int(width * 0.97)]
    pixel_values = processor(
        images=Image.fromarray(crop).convert("RGB"), return_tensors="pt"
    ).pixel_values.to(device)
    with torch.inference_mode():
        generated = model.generate(pixel_values, max_new_tokens=24, num_beams=1)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return normalize_dimension(text)


# =====================================================
# PAGE AND PDF ANALYSIS
# =====================================================
def get_page_content(page):
    embedded_text = page.get_text("text", sort=True).strip()
    words = page.get_text("words", sort=True)
    if len(re.sub(r"\s+", "", embedded_text)) >= 30:
        return embedded_text, "Embedded text", [], words
    ocr_text, positioned = run_rapidocr(page)
    return ocr_text, "Printed OCR", positioned, words


def analyze_pdf(pdf_bytes, run_handwriting=False):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_rows = []
    item_rows = []
    container_rows = []

    try:
        for page_number, page in enumerate(document, start=1):
            text, source, positioned, words = get_page_content(page)
            document_type = detect_document_type(text)

            is_flexport = (
                source == "Embedded text"
                and document_type == "Delivery Order"
                and "FLEXPORT" in text.upper()
                and "ENTRY-B/L" in text.upper()
            )

            if is_flexport:
                fields = extract_flexport_delivery_order(
                    words, page.rect.width, page.rect.height
                )
            elif document_type == "Shipping Instruction / Invoice":
                fields = extract_ega_page(text, page_number)
                item_rows.extend(extract_ega_items(text, page_number))
                container_rows.extend(extract_ega_containers(text, page_number))
            elif document_type == "Bill of Lading" or source == "Printed OCR":
                fields = extract_amazon_bol(text, positioned)
            else:
                fields = empty_fields()
                fields["Document Type"] = document_type

            dimension = normalize_dimension(text)
            if not dimension and run_handwriting:
                try:
                    dimension = extract_handwriting(pdf_bytes, page_number)
                except Exception:
                    dimension = ""

            if dimension:
                length, width, height = dimension.split("X")
                fields["Handwritten Text"] = dimension
                fields["Length"] = length
                fields["Width"] = width
                fields["Height"] = height

            page_rows.append(
                {
                    "Page Number": page_number,
                    "Text Source": source,
                    **fields,
                }
            )
    finally:
        document.close()

    return page_rows, item_rows, container_rows


# =====================================================
# EXCEL OUTPUT
# =====================================================
def format_sheet(worksheet):
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cells in worksheet.columns:
        width = min(max(len(str(cell.value or "")) for cell in cells) + 2, 60)
        worksheet.column_dimensions[cells[0].column_letter].width = width


def create_excel(page_df, item_df, container_df, validation_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        page_df.to_excel(writer, sheet_name="All PDF Pages", index=False)
        item_df.to_excel(writer, sheet_name="Item Details", index=False)
        container_df.to_excel(writer, sheet_name="Container Details", index=False)
        validation_df.to_excel(writer, sheet_name="Validation", index=False)
        for worksheet in writer.sheets.values():
            format_sheet(worksheet)
    return output.getvalue()


# =====================================================
# CLICKABLE WORKFLOW NAVIGATION
# =====================================================
def render_workflow_navigation():
    requested_step = str(st.query_params.get("step", "1"))
    if requested_step not in {"1", "2", "3", "4"}:
        requested_step = "1"

    steps = [
        ("1", "Upload &amp; Extract"),
        ("2", "Review Fields"),
        ("3", "Results"),
        ("4", "Export"),
    ]

    links = []
    for number, label in steps:
        active_class = " active" if number == requested_step else ""
        links.append(
            f'<a class="step-item{active_class}" href="?step={number}" target="_self">'
            f'<span class="step-number">{number}</span>{label}</a>'
        )

    st.markdown(
        '<div class="step-navigation">' + "".join(links) + '</div>',
        unsafe_allow_html=True,
    )


# =====================================================
# STREAMLIT UI
# =====================================================
render_workflow_navigation()

st.markdown(
    """
    <div class="content-card">
        <div class="section-title">Upload Documents</div>
        <div class="section-description">
            Upload one or more shipping PDFs. The application automatically
            identifies the document layout and extracts supported fields.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, settings_column = st.columns([1.65, 1], gap="large")

with upload_column:
    st.markdown("#### 📄 Select PDF Files")
    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Upload Amazon/Maersk Bills of Lading, Flexport Delivery Orders, "
            "or EGA Shipping Instructions."
        ),
    )

with settings_column:
    st.markdown("#### ⚙️ Extraction Settings")
    folder_path = st.text_input(
        "Optional local folder path",
        placeholder=r"C:\Users\Name\Downloads\PDFs",
        help="Use this option only when running Streamlit locally.",
    )
    run_handwriting = st.checkbox(
        "Enable handwriting OCR",
        value=False,
        help=(
            "Runs TrOCR when a dimension such as L x W x H is not found "
            "in the printed or embedded text."
        ),
    )
    st.info(
        "Automatic layout detection is enabled for Bill of Lading, "
        "Delivery Order, and Shipping Instruction documents."
    )

pdf_items = []
seen_names = set()

for uploaded in uploaded_files or []:
    if uploaded.name not in seen_names:
        pdf_items.append({"name": uploaded.name, "bytes": uploaded.getvalue()})
        seen_names.add(uploaded.name)

if folder_path and os.path.isdir(folder_path):
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith(".pdf") and file_name not in seen_names:
            complete_path = os.path.join(folder_path, file_name)
            with open(complete_path, "rb") as file_handle:
                pdf_items.append({"name": file_name, "bytes": file_handle.read()})
            seen_names.add(file_name)
elif folder_path:
    st.warning(
        "The local folder path was not found. Uploaded files can still be processed."
    )

if "unified_pdf_cache" not in st.session_state:
    st.session_state.unified_pdf_cache = {}
if "show_extraction_results" not in st.session_state:
    st.session_state.show_extraction_results = False
if "last_pdf_signature" not in st.session_state:
    st.session_state.last_pdf_signature = ""

current_pdf_signature = "|".join(
    f"{item['name']}:{hashlib.md5(item['bytes']).hexdigest()}" for item in pdf_items
)

if current_pdf_signature != st.session_state.last_pdf_signature:
    st.session_state.show_extraction_results = False
    st.session_state.last_pdf_signature = current_pdf_signature

all_page_rows = []
all_item_rows = []
all_container_rows = []

if pdf_items:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Reading PDFs, detecting layouts, and extracting fields..."):
        progress_bar = st.progress(0)
        progress_message = st.empty()
        total_files = len(pdf_items)

        for file_index, item in enumerate(pdf_items, start=1):
            progress_message.caption(
                f"Processing {item['name']} ({file_index} of {total_files})"
            )
            cache_key = (
                f"{CACHE_VERSION}:{run_handwriting}:"
                f"{hashlib.md5(item['bytes']).hexdigest()}"
            )

            if cache_key not in st.session_state.unified_pdf_cache:
                st.session_state.unified_pdf_cache[cache_key] = analyze_pdf(
                    item["bytes"], run_handwriting=run_handwriting
                )

            pages, items, containers = st.session_state.unified_pdf_cache[cache_key]
            all_page_rows.extend(
                {"File Name": item["name"], **row} for row in pages
            )
            all_item_rows.extend(
                {"File Name": item["name"], **row} for row in items
            )
            all_container_rows.extend(
                {"File Name": item["name"], **row} for row in containers
            )
            progress_bar.progress(file_index / total_files)

        progress_bar.empty()
        progress_message.empty()

    page_df = pd.DataFrame(all_page_rows)
    item_df = pd.DataFrame(all_item_rows)
    container_df = pd.DataFrame(all_container_rows)

    ordered_columns = ["File Name", "Page Number", "Text Source"] + ALL_FIELDS
    for column in ordered_columns:
        if column not in page_df.columns:
            page_df[column] = ""
    page_df = page_df[ordered_columns]

    automatically_detected_fields = [
        field
        for field in ALL_FIELDS
        if page_df[field].fillna("").astype(str).str.strip().ne("").any()
    ]

    st.markdown(
        """
        <div class="section-title">Extraction Overview</div>
        <div class="section-description">
            Summary of documents and fields found during automatic analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("PDF Files", len(pdf_items))
    metric_2.metric("PDF Pages", len(page_df))
    metric_3.metric("Fields Detected", len(automatically_detected_fields))

    document_types = page_df["Document Type"].fillna("").astype(str).str.strip()
    detected_layout_count = document_types[document_types.ne("")].nunique()
    metric_4.metric("Document Layouts", detected_layout_count)

    st.markdown("<br>", unsafe_allow_html=True)
    preview_column, values_column = st.columns([1.05, 1], gap="large")

    with preview_column:
        st.markdown(
            '<div class="preview-heading">📑 Document Preview</div>',
            unsafe_allow_html=True,
        )
        preview_file_names = [item["name"] for item in pdf_items]
        selected_preview_name = st.selectbox(
            "Preview Document",
            preview_file_names,
            label_visibility="collapsed",
        )
        selected_preview_item = next(
            item for item in pdf_items if item["name"] == selected_preview_name
        )
        preview_doc = fitz.open(
            stream=selected_preview_item["bytes"], filetype="pdf"
        )
        try:
            preview_page_count = len(preview_doc)
            selected_preview_page = st.number_input(
                "Preview Page",
                min_value=1,
                max_value=preview_page_count,
                value=1,
                step=1,
            )
            preview_page = preview_doc[int(selected_preview_page) - 1]
            preview_pix = preview_page.get_pixmap(
                matrix=fitz.Matrix(1.1, 1.1),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            preview_image = Image.open(io.BytesIO(preview_pix.tobytes("png")))
            st.image(
                preview_image,
                caption=f"{selected_preview_name} - Page {selected_preview_page}",
                use_container_width=True,
            )
        finally:
            preview_doc.close()

    with values_column:
        st.markdown(
            '<div class="preview-heading">🔎 Extracted Information</div>',
            unsafe_allow_html=True,
        )
        selected_result_rows = page_df[
            page_df["File Name"] == selected_preview_name
        ]
        selected_result_row = selected_result_rows[
            selected_result_rows["Page Number"] == int(selected_preview_page)
        ]

        if selected_result_row.empty:
            st.info("No extracted values are available for this page.")
        else:
            first_row = selected_result_row.iloc[0]
            preview_records = [
                {"Field": field, "Extracted Value": first_row.get(field, "")}
                for field in ALL_FIELDS
                if clean(first_row.get(field, ""))
            ]
            preview_df = pd.DataFrame(preview_records)
            if preview_df.empty:
                st.warning("No supported field values were detected on this page.")
            else:
                st.dataframe(
                    preview_df,
                    use_container_width=True,
                    hide_index=True,
                    height=560,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">Review and Select Fields</div>
        <div class="section-description">
            Use automatic selection or choose the exact fields that should be
            shown and exported to Excel.
        </div>
        """,
        unsafe_allow_html=True,
    )

    field_mode = st.radio(
        "Choose how fields should be selected",
        ["Automatically detect fields", "Select fields manually"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if field_mode == "Automatically detect fields":
        st.success(
            f"{len(automatically_detected_fields)} fields containing values "
            "were detected automatically."
        )
        selected_fields = st.multiselect(
            "Detected Fields",
            ALL_FIELDS,
            default=automatically_detected_fields,
            help=(
                "Fields were detected automatically from all uploaded PDF pages. "
                "You can still add or remove fields."
            ),
        )
    else:
        selected_fields = st.multiselect(
            "Select Output Fields",
            ALL_FIELDS,
            default=automatically_detected_fields,
            help=(
                "Choose exactly which columns should appear in the results and "
                "Excel workbook."
            ),
        )

    display_columns = ["File Name", "Page Number", "Text Source"] + [
        field for field in ALL_FIELDS if field in selected_fields
    ]

    button_column, information_column = st.columns([0.32, 0.68])
    with button_column:
        extract_clicked = st.button(
            "🚀 Show Extraction Results",
            type="primary",
            use_container_width=True,
        )
    with information_column:
        st.caption(
            f"{len(selected_fields)} fields selected for {len(page_df)} page row(s)."
        )

    if extract_clicked:
        st.session_state.show_extraction_results = True

    if st.session_state.show_extraction_results:
        output_df = page_df[display_columns].copy()
        validation_rows = []

        for field in selected_fields:
            extracted_count = int(
                output_df[field]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            )
            if extracted_count == len(output_df):
                validation_status = "PASS"
            elif extracted_count > 0:
                validation_status = "PARTIAL"
            else:
                validation_status = "REVIEW"

            coverage = (extracted_count / len(output_df) * 100) if len(output_df) else 0
            validation_rows.append(
                {
                    "Field": field,
                    "Rows Extracted": extracted_count,
                    "Total Page Rows": len(output_df),
                    "Coverage": f"{coverage:.1f}%",
                    "Status": validation_status,
                }
            )

        validation_df = pd.DataFrame(validation_rows)
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(
            f"Extraction completed for {len(output_df)} page(s) from "
            f"{len(pdf_items)} PDF file(s)."
        )
        st.markdown(
            """
            <div class="section-title">Extraction Results</div>
            <div class="section-description">
                Review page-level data, EGA details, PDF page images, and field
                validation before downloading Excel.
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_tab, item_tab, container_tab, image_tab, validation_tab = st.tabs(
            [
                "📋 All PDF Pages",
                "📦 EGA Items",
                "🚢 EGA Containers",
                "🖼️ Page Images",
                "✅ Validation",
            ]
        )

        with page_tab:
            st.caption(
                "One row is created for every PDF page. You can edit extracted "
                "values before downloading the Excel file."
            )
            edited_output_df = st.data_editor(
                output_df,
                use_container_width=True,
                hide_index=True,
                height=600,
                num_rows="fixed",
                key="page_results_editor",
            )

        with item_tab:
            if item_df.empty:
                st.info("No EGA item-detail rows were found in the uploaded PDFs.")
            else:
                st.dataframe(
                    item_df, use_container_width=True, hide_index=True, height=500
                )

        with container_tab:
            if container_df.empty:
                st.info(
                    "No EGA container-detail rows were found in the uploaded PDFs."
                )
            else:
                st.dataframe(
                    container_df,
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )

        with image_tab:
            st.caption("Rendered images for every page in the selected PDF files.")
            for item in pdf_items:
                with st.expander(f"📄 {item['name']}", expanded=False):
                    document = fitz.open(stream=item["bytes"], filetype="pdf")
                    try:
                        for page_number, page in enumerate(document, start=1):
                            pix = page.get_pixmap(
                                matrix=fitz.Matrix(1.0, 1.0),
                                colorspace=fitz.csRGB,
                                alpha=False,
                            )
                            st.image(
                                Image.open(io.BytesIO(pix.tobytes("png"))),
                                caption=f"{item['name']} - Page {page_number}",
                                use_container_width=True,
                            )
                    finally:
                        document.close()

        with validation_tab:
            if validation_df.empty:
                st.info(
                    "Select at least one output field to generate validation information."
                )
            else:
                st.dataframe(
                    validation_df,
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )
                pass_count = int(validation_df["Status"].eq("PASS").sum())
                partial_count = int(validation_df["Status"].eq("PARTIAL").sum())
                review_count = int(validation_df["Status"].eq("REVIEW").sum())
                validation_metric_1, validation_metric_2, validation_metric_3 = (
                    st.columns(3)
                )
                validation_metric_1.metric("Passed", pass_count)
                validation_metric_2.metric("Partial", partial_count)
                validation_metric_3.metric("Needs Review", review_count)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">Export Results</div>
            <div class="section-description">
                Download page data, item details, container details, and validation
                results in one Excel workbook.
            </div>
            """,
            unsafe_allow_html=True,
        )

        download_column, export_info_column = st.columns([0.34, 0.66])
        with download_column:
            st.download_button(
                "📥 Download Complete Excel",
                data=create_excel(
                    edited_output_df,
                    item_df,
                    container_df,
                    validation_df,
                ),
                file_name="Complete_All_PDF_Extraction.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )
        with export_info_column:
            st.caption(
                "The Excel workbook contains All PDF Pages, Item Details, "
                "Container Details, and Validation sheets."
            )
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Upload one or more PDF files above to begin automatic extraction.")
    empty_1, empty_2, empty_3 = st.columns(3)

    with empty_1:
        st.markdown(
            """
            ### 📄 1. Upload
            Select one or more shipping PDF documents.
            """
        )
    with empty_2:
        st.markdown(
            """
            ### 🔎 2. Review
            Check automatically detected fields and values.
            """
        )
    with empty_3:
        st.markdown(
            """
            ### 📥 3. Export
            Review the final table and download Excel.
            """
        )

st.markdown(
    """
    <div class="app-footer">
        SmartDoc Extractor &nbsp;•&nbsp;
        Multi-layout document processing &nbsp;•&nbsp;
        One output row per PDF page &nbsp;•&nbsp;
        Optional handwriting OCR
    </div>
    """,
    unsafe_allow_html=True,
)
