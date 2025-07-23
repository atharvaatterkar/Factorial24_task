import os
import cv2
import fitz  
import easyocr
import pandas as pd
import numpy as np

reader = easyocr.Reader(['en'])

def pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    image_folder = os.path.join(output_folder, f"{filename}_pages")
    os.makedirs(image_folder, exist_ok=True)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        image_path = os.path.join(image_folder, f"page_{page_num+1}.png")
        pix.save(image_path)
    return image_folder, filename

def classify_shape(contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        # check for square vs rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) > 5:
        shape = "circle"
    return shape


def detect_shapes_and_dimensions(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ocr_result = reader.readtext(image)

    shape_data = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 500:  # filter noise
            continue

        shape_type = classify_shape(cnt)


        closest_text = ""
        min_dist = float('inf')
        for (bbox, text, conf) in ocr_result:
            (tl, tr, br, bl) = bbox
            text_x = int((tl[0] + br[0]) / 2)
            text_y = int((tl[1] + br[1]) / 2)
            dist = np.sqrt((x - text_x)**2 + (y - text_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_text = text

        shape_data.append({
            "image": os.path.basename(image_path),
            "shape": shape_type,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "dimension_text": closest_text
        })

        # Annotate
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, closest_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return image, shape_data

def run_pipeline():
    pdf_folder = "engineering drawings"
    output_image_folder = "pdf_pages"
    base_annotated_folder = "annotated"
    base_excel_folder = "output"

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(base_annotated_folder, exist_ok=True)
    os.makedirs(base_excel_folder, exist_ok=True)

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            img_folder, filename = pdf_to_images(pdf_path, output_image_folder)

            annotated_folder = os.path.join(base_annotated_folder, filename)
            os.makedirs(annotated_folder, exist_ok=True)

            all_data = []

            for img_file in sorted(os.listdir(img_folder)):
                img_path = os.path.join(img_folder, img_file)
                annotated_img, shape_data = detect_shapes_and_dimensions(img_path)

                out_path = os.path.join(annotated_folder, img_file)
                cv2.imwrite(out_path, annotated_img)

                all_data.extend(shape_data)

            
            df = pd.DataFrame(all_data)
            excel_path = os.path.join(base_excel_folder, f"{filename}_summary.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"Processed: {file} → {excel_path}")


if __name__ == "__main__":
    run_pipeline()
