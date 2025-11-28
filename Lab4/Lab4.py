import cv2
import numpy as np

SCALE = 0.3


def create_tracker(tracker_name: str):
    tracker_name = tracker_name.upper()

    if tracker_name == "KCF":
        return cv2.legacy.TrackerKCF_create()

    if tracker_name == "CSRT":
        return cv2.legacy.TrackerCSRT_create()

def run_opencv_tracker(cap, tracker_type: str):
    ok, frame = cap.read()

    if SCALE != 1.0:
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    if bbox[2] == 0 or bbox[3] == 0:
        print("ROI not selected")
        return

    tracker = create_tracker(tracker_type)
    tracker.init(frame, bbox)

    while True:
        ok, frame = cap.read()

        if SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

        ok, bbox = tracker.update(frame)

        if ok:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{tracker_type} OK", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"{tracker_type} LOST", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def run_diff_tracker(cap):
    ok, frame = cap.read()

    if SCALE != 1.0:
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    if w == 0 or h == 0:
        print("ROI not selected")
        return

    track_window = (int(x), int(y), int(w), int(h))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_area = max(300, (w * h) // 30)

    while True:
        ok, frame = cap.read()

        if SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray, prev_gray)
        prev_gray = gray.copy()

        diff_blur = cv2.GaussianBlur(diff, (11, 11), 9)
        noise_level = float(np.median(diff_blur))
        thr_val = 3

        _, thresh = cv2.threshold(diff_blur, thr_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        x, y, w, h = track_window
        h_frame, w_frame = thresh.shape[:2]

        margin = 50
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w_frame, x + w + margin)
        y1 = min(h_frame, y + h + margin)

        local = thresh[y0:y1, x0:x1]

        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False
        best_bbox = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            if area > best_area:
                best_area = area
                best_bbox = (x_c, y_c, w_c, h_c)
                found = True

        if found and best_bbox is not None:
            bx, by, bw, bh = best_bbox
            x_new = x0 + bx
            y_new = y0 + by
            track_window = (x_new, y_new, bw, bh)

            cv2.rectangle(frame, (x_new, y_new), (x_new + bw, y_new + bh), (0, 255, 0), 2)
            status_text = f"DIFF OK area={best_area:.0f} thr={thr_val:.0f} noise={noise_level:.0f}"
            color = (0, 255, 0)
        else:
            x, y, w, h = track_window
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            status_text = f"DIFF LOST thr={thr_val:.0f} noise={noise_level:.0f}"
            color = (0, 0, 255)

        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Choose metod of Object Tracking:")
    print("  1 - DIFF")
    print("  2 - KCF")
    print("  3 - CSRT")

    choice = input("Type 1/2/3 and press Enter: ").strip()

    if choice == "1":
        method = "DIFF"
    elif choice == "2":
        method = "KCF"
    elif choice == "3":
        method = "CSRT"

    video_source = "ya.mp4"

    cap = cv2.VideoCapture(video_source)

    if method == "DIFF":
        run_diff_tracker(cap)
    else:
        run_opencv_tracker(cap, method)


if __name__ == "__main__":
    main()