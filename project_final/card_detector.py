# Luke Jansen
# 7/24/26
# Read the YOLO11 Nano OBB pytorch model trained on pokemon TCG cards
# Outputs Oriented Bounding Box (OBB) for each card on a live webcam feed

import sys

import cv2
from ultralytics import YOLO

def order_quad_corners(pts):
    """
    Orders a list of 4 (x, y) points as [top-left, top-right, bottom-right, bottom-left].
    Sorting is done by y (top pair first), then by x within each pair.
    Input: pts - list of 4 points, each a tuple or list (x, y) or np.ndarray.
    Output: pts reordered in-place as [TL, TR, BR, BL]
    """
    if len(pts) != 4:
        return

    # Convert points to list of tuples if using np.ndarray
    pts = [tuple(p) for p in pts]

    # Sort by y, then by x within each pair
    pts_sorted = sorted(pts, key=lambda p: (round(p[1], 1), p[0]))

    # First 2 are the top pair (smaller y), sort them by x (left to right)
    top_pair = sorted(pts_sorted[:2], key=lambda p: p[0])
    bottom_pair = sorted(pts_sorted[2:], key=lambda p: p[0])

    tl = top_pair[0]
    tr = top_pair[1]
    bl = bottom_pair[0]
    br = bottom_pair[1]

    ordered = [tl, tr, br, bl]
    # Mutate original list in-place if it's a python list
    for i in range(4):
        pts[i] = ordered[i]

    return pts

def main(argv):
    weights = argv[1] if len(argv) > 1 else "data/yolo/ptcg-detector.pt"
    device = argv[2] if len(argv) > 2 else "/dev/video0"

    # Ultralytics checkpoint: weights + architecture (not a project5 state_dict)
    # This model contains weights, architecture metadata, training config, etc.
    # No need to build model like in project5
    model = YOLO(weights)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Unable to open video device: {device}")
        return 1

    cv2.namedWindow("Card Detector", cv2.WINDOW_NORMAL)
    print("Controls: q = quit")

    bridge_path = "bridge.txt"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("frame is empty")
            break

        # Predict on this BGR frame; verbose=False keeps the terminal quiet
        results = model.predict(frame, verbose=False)

        # Draw OBB / boxes onto a copy of the frame
        annotated = results[0].plot()

        cv2.imshow("Card Detector", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        r = results[0]

        if r.obb is not None and len(r.obb) > 0:
            i = int(r.obb.conf.argmax())
            # (4, 2) float corners of best detection
            pts = r.obb.xyxyxyxy.cpu().numpy()[i].tolist()
            corners = order_quad_corners(pts)  # [TL, TR, BR, BL]
            # one line, overwrite — C++ reads latest
            with open(bridge_path, "w") as f:
                f.write(
                    f"{corners[0][0]},{corners[0][1]},"
                    f"{corners[1][0]},{corners[1][1]},"
                    f"{corners[2][0]},{corners[2][1]},"
                    f"{corners[3][0]},{corners[3][1]}\n"
                )
        else:
            with open(bridge_path, "w") as f:
                f.write("none\n")

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
