import cv2
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

VIDEO_PATH = "video.mp4"
OUTPUT_PATH = "result_cell_6.mp4"
CONF_THRESH = 0.7

CELL_INDEX = 6
GRID_ROWS = 3
GRID_COLS = 3

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def crop_grid_cell(frame, cell_index, rows=3, cols=3):

    h, w = frame.shape[:2]
    idx = cell_index - 1
    row = idx // cols
    col = idx % cols

    cell_w = w // cols
    cell_h = h // rows

    x1 = col * cell_w
    x2 = (col + 1) * cell_w
    y1 = row * cell_h
    y2 = (row + 1) * cell_h

    return frame[y1:y2, x1:x2].copy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
model.eval()

cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tile = crop_grid_cell(frame, CELL_INDEX, GRID_ROWS, GRID_COLS)
    
    if out is None:
        h_t, w_t = tile.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w_t, h_t))

    image_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    img_tensor = to_tensor(image_rgb).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()
    labels = outputs["labels"].cpu()

    keep = scores >= CONF_THRESH

    for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
        label_id = int(label.item())
        if not (0 <= label_id < len(COCO_INSTANCE_CATEGORY_NAMES)):
            continue
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
        x1, y1, x2, y2 = box.int().tolist()
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        text = f"{class_name} {score:.2f}"

        cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            tile,
            (x1, y1 - th - baseline),
            (x1 + tw, y1),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            tile,
            text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imshow("Detections", tile)

    if out is not None:
        out.write(tile)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()