import base64
import re
import cv2
import torch
import traceback
import numpy as np
import nepali_datetime
import io
from PIL import Image

# def preprocess_image(img):
#     # # Convert image to grayscale
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # # Convert grayscale image to numpy array
#     # img = np.array(gray)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,2))
#     morphology_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

#     # Apply median blur
#     blur = cv2.GaussianBlur(morphology_img, (3,3),0)

#     # Apply thresholding
#     _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)


#     # Find the bounding box coordinates of the non-white pixels
#     coords = cv2.findNonZero(binary)
#     x, y, w, h = cv2.boundingRect(coords)

#     # Add extra white space to the bounding box coordinates
#     padding = 5  # Adjust the padding size as needed
#     x -= padding
#     y -= padding
#     w += 2 * padding
#     h += 2 * padding

#     # Make sure the coordinates are within the image boundaries
#     x = max(0, x)
#     y = max(0, y)
#     w = min(w, img.shape[1] - x)
#     h = min(h, img.shape[0] - y)

#     # Crop the image using the modified bounding box coordinates
#     cropped_image = blur[y:y + h, x:x + w]

#     # Add extra white space around the cropped image
#     extra_space = np.zeros((cropped_image.shape[0] + 2 * padding, cropped_image.shape[1] + 2 * padding), dtype=np.uint8) * 255
#     extra_space[padding:-padding, padding:-padding] = cropped_image

#     corrected = cv2.resize(extra_space,(330,175))
#     # Convert the numpy array back to PIL image
#     resized_image = Image.fromarray(corrected)
#     bin_image = Image.fromarray(binary)
#     bin_image.save("1000.jpg")

#     return resized_image


def process_image_as_tensor(img):
    # Convert bytes to OpenCV img
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1366, 768))
    # Convert to PyTorch tensor
    img_tensor = (
        torch.tensor(img * 255, dtype=torch.float32) / 255.0
    )  # Normalize to [0, 1]
    img_tensor = img_tensor.permute(2, 0, 1)
    return img_tensor


def preprocess_for_trocr(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    return img


def encode_image_for_api(img):
    return (
        "image.png",
        (cv2.imencode(".png", (np.array(img)))[1]).tobytes(),
        "image/png",
    )


# def encode_image_for_api(img, quality=80):  # Add a quality parameter to control compression
#     # Compress the image as JPEG with the specified quality
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]  # The second element is the quality (0-100)

#     # Encoding the image in JPEG format
#     result, encoded_img = cv2.imencode(".jpg", np.array(img), encode_param)

#     if result:
#         return (
#             "image.jpg",  # File name for the image
#             encoded_img.tobytes(),  # Encoded image bytes
#             "image/jpeg",  # MIME type
#         )
#     else:
#         raise Exception("Image encoding failed")


def is_bs_date(bs_date_str):
    """Determine if a given date string is in BS or AD format."""
    try:
        # Split the date string to extract the year
        # month, day, year = map(int, bs_date_str.split('-'))
        year, month, day = map(int, bs_date_str.split("-"))

        # Heuristic check: BS years are typically greater than 1943
        # (Bikram Sambat year 2000 is equivalent to 1943 AD)
        if year > 2070:
            return True
        else:
            return False
    except Exception as e:
        # If parsing fails, return False
        print(e)
        return False


def convert_bs_to_ad(bs_date_str):
    """Convert a BS date string to an AD date."""
    # Split the BS date string into month, day, and year

    # day, month, year = map(int, bs_date_str.split('-'))
    year, month, day = map(int, bs_date_str.split("-"))
    # Create a Nepali date object
    bs_date = nepali_datetime.date(year, month, day)
    # Convert BS date to AD date
    ad_date = bs_date.to_datetime_date()
    # ad_date_str = ad_date.strftime('%d/%m/%Y')
    ad_date_str = ad_date.strftime("%Y-%m-%d")
    return ad_date_str


def format_date(date):
    try:
        if not is_bs_date(date):
            return date
        return convert_bs_to_ad(date)
    except Exception as e:
        print(e)
        return date


def date_validation(result):
    nepali_to_english_digits = str.maketrans("०१२३४५६७८९", "0123456789")
    try:
        if "date" in result or "Date" in result:
            print(result["date"])
            result = result["date"].translate(nepali_to_english_digits)
            print(type(result))
            print(result)
            raw_date = re.sub(r"\D", "", result).zfill(8)

            # raw_date = str(int(re.sub(r"(?i)rs|[:*,/-]", "", result["date"]))).zfill(8)
            formatted_date = f"{raw_date[-4:]}-{raw_date[2:4]}-{raw_date[:2]}"
            return format_date(formatted_date)
        return None
    except Exception as e:
        print(f"Date validation failed: {e}")
        traceback.print_exc()
        return None


def account_number_validation(result):
    try:
        if "account_number" not in result:
            return None
        cleaned_amount = re.sub(r"(?i)[A-Za-z]|[:*.,/-]", "", result["account_number"])
        return cleaned_amount.strip().zfill(20)
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        return ""


def micr_validation(result):
    try:
        if "micr" not in result.keys():
            return None
        cleaned_micr = result["micr"].rstrip(".").replace(" ", "")
        return cleaned_micr
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        return ""


def account_name_validation(result):
    try:
        if "account_name" not in result.keys():
            return None
        cleaned_account_name = result["account_name"]
        return cleaned_account_name
    except Exception as e:
        print(traceback.print_exc())
        return None


def bfd_validation(result):
    try:
        if "payee" not in result.keys():
            return None
        cleaned_payee = result["payee"]
        return cleaned_payee
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        return None


def sum_validation(result):
    try:
        if "sum" not in result.keys():
            return None
        cleaned_sum = result["sum"]
        cleaned_sum = cleaned_sum.replace(".", "")
        cleaned_sum = cleaned_sum.replace("&", "AND")
        return cleaned_sum[:230]
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        return None


def amount_validation(result, nep=False):
    nepali_to_english_digits = str.maketrans("०१२३४५६७८९", "0123456789")

    try:
        if "amount" not in result.keys():
            return None
        try:
            result["amount"] = result["amount"].translate(nepali_to_english_digits)
            result["amount"] = re.sub(r"[\u0900-\u097F]", "", result["amount"])
            result["amoujt"] = result["amount"].replace("/", ".")
        except:
            result["amount"] = result["amount"]
        step1 = re.sub(r"[^\d.]+", "", result["amount"])
        cleaned_amount = str(re.sub(r"^\.*", "", step1)).strip()
        return cleaned_amount
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        return None


def decode_base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def decode_base64_to_np_array(base64_str):
    img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    return np.array(img)  # < ------ Added ----- >


if __name__ == "__main__":
    print(amount_validation({"amount": '"र१०,०००००/- "'}, nep=True))
