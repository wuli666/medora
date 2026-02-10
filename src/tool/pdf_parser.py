import base64

import fitz  # PyMuPDF


def parse_pdf(pdf_path: str) -> dict:
    """Extract text and images from a PDF file.

    Returns dict with 'text' (str) and 'images' (list of base64 strings).
    """
    doc = fitz.open(pdf_path)
    text_parts = []
    images = []

    for page in doc:
        text_parts.append(page.get_text())
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            if base_image and base_image.get("image"):
                images.append(base64.b64encode(base_image["image"]).decode("utf-8"))

    doc.close()
    return {
        "text": "\n".join(text_parts).strip(),
        "images": images,
    }
