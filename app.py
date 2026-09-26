from pathlib import Path
import tempfile

import streamlit as st

from pipeline import process_receipt


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="ReceiptAI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "styles.css"


# =========================================================
# Load CSS
# =========================================================

def load_css():

    if CSS_PATH.exists():

        with open(
            CSS_PATH,
            "r",
            encoding="utf-8"
        ) as css_file:

            css = css_file.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )


load_css()


# =========================================================
# Header
# =========================================================

st.title("🧾 ReceiptAI")

st.caption(
    "Intelligent Receipt Understanding"
)


# =========================================================
# Hero Section
# =========================================================

st.markdown(
    "### AI-POWERED DOCUMENT UNDERSTANDING"
)

st.header(
    "Turn receipts into structured data."
)

st.write(
    """
    Upload a receipt and automatically detect, recognize,
    and extract the **company, address, date, and total amount**
    using computer vision, OCR, and transformer-based
    information extraction.
    """
)

st.divider()


# =========================================================
# Upload Section
# =========================================================

st.subheader(
    "Upload a receipt"
)

st.caption(
    "Upload a JPG, JPEG, or PNG receipt image."
)


uploaded_file = st.file_uploader(
    "Choose a receipt image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# =========================================================
# Receipt Uploaded
# =========================================================

if uploaded_file is not None:

    st.write("")

    preview_column, result_column = st.columns(
        [1, 1],
        gap="large"
    )


    # =====================================================
    # Receipt Preview
    # =====================================================

    with preview_column:

        st.subheader(
            "Receipt Preview"
        )

        st.image(
            uploaded_file,
            use_container_width=True
        )


    # =====================================================
    # Extracted Information
    # =====================================================

    with result_column:

        st.subheader(
            "Extracted Information"
        )

        analyze_button = st.button(
            "✨ Analyze Receipt",
            type="primary",
            use_container_width=True
        )


        # =================================================
        # Analyze
        # =================================================

        if analyze_button:

            temp_path = None

            try:

                suffix = Path(
                    uploaded_file.name
                ).suffix


                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name


                with st.spinner(
                    "Analyzing receipt..."
                ):

                    result = process_receipt(
                        temp_path
                    )


                # Store result
                st.session_state[
                    "receipt_result"
                ] = result

                st.session_state[
                    "processed_file"
                ] = uploaded_file.name


            except Exception as error:

                st.error(
                    "The receipt could not be processed."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.exception(error)


            finally:

                if temp_path is not None:

                    temp_file_path = Path(
                        temp_path
                    )

                    if temp_file_path.exists():

                        temp_file_path.unlink()


        # =================================================
        # Show Results
        # =================================================

        result_exists = (
            "receipt_result"
            in st.session_state
        )

        same_file = (
            st.session_state.get(
                "processed_file"
            )
            == uploaded_file.name
        )


        if result_exists and same_file:

            result = st.session_state[
                "receipt_result"
            ]


            st.success(
                "Receipt analyzed successfully!"
            )


            # =============================================
            # Company
            # =============================================

            st.markdown(
                "##### 🏢 Company"
            )

            company = (
                result["company"]
                if result["company"]
                else "Not detected"
            )

            st.info(company)


            # =============================================
            # Address
            # =============================================

            st.markdown(
                "##### 📍 Address"
            )

            address = (
                result["address"]
                if result["address"]
                else "Not detected"
            )

            st.info(address)


            # =============================================
            # Date + Total
            # =============================================

            date_column, total_column = (
                st.columns(2)
            )


            with date_column:

                st.markdown(
                    "##### 📅 Date"
                )

                date = (
                    result["date"]
                    if result["date"]
                    else "Not detected"
                )

                st.info(date)


            with total_column:

                st.markdown(
                    "##### 💰 Total"
                )

                total = (
                    result["total"]
                    if result["total"]
                    else "Not detected"
                )

                st.success(total)


            # =============================================
            # Advanced Details
            # =============================================

            with st.expander(
                "🔍 Advanced Details"
            ):

                st.markdown(
                    "### Recognized OCR Text"
                )

                if result["ocr_text"]:

                    st.code(
                        result["ocr_text"],
                        language=None
                    )

                else:

                    st.warning(
                        "No OCR text detected."
                    )


                st.markdown(
                    "### T5 Model Output"
                )

                if result[
                    "raw_t5_output"
                ]:

                    st.code(
                        result[
                            "raw_t5_output"
                        ],
                        language=None
                    )

                else:

                    st.warning(
                        "No T5 output generated."
                    )


        # =================================================
        # Before Analysis
        # =================================================

        else:

            st.info(
                "Click **Analyze Receipt** to extract "
                "the receipt information."
            )


# =========================================================
# Divider
# =========================================================

st.divider()


# =========================================================
# How It Works
# =========================================================

st.header(
    "How ReceiptAI Works"
)

st.caption(
    "Three AI stages work together to understand each receipt."
)


step1, step2, step3 = st.columns(
    3,
    gap="large"
)


# =========================================================
# YOLO
# =========================================================

with step1:

    st.markdown(
        "### ① Text Detection"
    )

    st.write(
        """
        **YOLO11**

        Detects text regions across
        the uploaded receipt.
        """
    )


# =========================================================
# PaddleOCR
# =========================================================

with step2:

    st.markdown(
        "### ② Text Recognition"
    )

    st.write(
        """
        **PaddleOCR**

        Converts the detected text
        regions into readable text.
        """
    )


# =========================================================
# T5
# =========================================================

with step3:

    st.markdown(
        "### ③ Information Extraction"
    )

    st.write(
        """
        **T5 Transformer**

        Extracts the company, address,
        date, and total amount.
        """
    )


# =========================================================
# Project Pipeline
# =========================================================

st.write("")

st.markdown(
    """
**Receipt Image** → **YOLO11** → **PaddleOCR** → **T5** → **Structured Data**
"""
)


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "ReceiptAI • Computer Vision • OCR • Transformer-based Information Extraction"
)