from pathlib import Path

from pipeline import process_receipt


# =========================================================
# Test Image
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "test_receipt.jpg"


# =========================================================
# Check Image
# =========================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Test receipt not found: {IMAGE_PATH}"
    )


# =========================================================
# Run Complete Pipeline
# =========================================================

print("\nProcessing receipt...")

result = process_receipt(
    IMAGE_PATH
)


# =========================================================
# Display Results
# =========================================================

print("\n" + "=" * 50)
print("EXTRACTED INFORMATION")
print("=" * 50)

print(f"Company : {result['company']}")
print(f"Address : {result['address']}")
print(f"Date    : {result['date']}")
print(f"Total   : {result['total']}")


print("\n" + "=" * 50)
print("OCR TEXT")
print("=" * 50)

print(result["ocr_text"])


print("\n" + "=" * 50)
print("RAW T5 OUTPUT")
print("=" * 50)

print(result["raw_t5_output"])