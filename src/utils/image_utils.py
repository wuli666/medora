import base64


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def resize_image_if_needed(image_bytes: bytes, max_size: int = 1024) -> bytes:
    try:
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(image_bytes))
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
            buf = BytesIO()
            img.save(buf, format=img.format or "PNG")
            return buf.getvalue()
    except ImportError:
        pass
    return image_bytes
