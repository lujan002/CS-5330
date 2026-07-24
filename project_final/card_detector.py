# Luke Jansen
# 7/24/2026
# main script for AR pokemon card-model visualization
# YOLO11 Nano OBB for card corners (replaces contour detect / bridge file)

import sys

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# TODO (later): Assimp FBX loader
#   - Load XY .FBX (mesh + bones + skin weights + animation clips)
#   - Do NOT convert to .obj if you want idle animation (obj drops skeleton)
#   - libassimp-dev is already available on this machine
#
# TODO (later): Skinned OpenGL renderer (like project4 gl_renderer)
#   - Bone palette in the vertex shader
#   - Advance idle animation time each frame
#   - Place model on the card plane using rvec/tvec from solvePnP below
#
# TODO (later): OCR on card name region -> pick which Pokemon FBX to load
# TODO (later): Scrape Models Resource ZIP URLs to download models
# ---------------------------------------------------------------------------


def order_quad_corners(pts):
    if len(pts) != 4:
        return pts

    pts = [tuple(p) for p in pts]

    # sort by y (top pair first), then by x within each pair
    pts_sorted = sorted(pts, key=lambda p: (round(p[1], 1), p[0]))

    top_pair = sorted(pts_sorted[:2], key=lambda p: p[0])
    bottom_pair = sorted(pts_sorted[2:], key=lambda p: p[0])

    tl = top_pair[0]
    tr = top_pair[1]
    bl = bottom_pair[0]
    br = bottom_pair[1]

    return [tl, tr, br, bl]


def load_intrinsics(path):
    fs = cv2.FileStorage(path, cv2.FileStorage_READ)
    if not fs.isOpened():
        return None, None
    camera_mat = fs.getNode("camera_mat").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    if camera_mat is None or camera_mat.size == 0:
        return None, None
    return camera_mat, dist_coeffs


def main(argv):
    weights = argv[1] if len(argv) > 1 else "data/yolo/ptcg-detector.pt"
    device = argv[2] if len(argv) > 2 else "/dev/video0"

    model = YOLO(weights)

    capdev = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capdev.isOpened():
        print("Unable to open video device")
        return -1
    cv2.namedWindow("Video", 1)

    img_size = (640, 480)

    # Card plane in world units (Z = 0). Top-left corner is origin.
    # Aspect ~ Pokemon TCG card (2.5x3.5 in. or 63.5x88 mm).
    card_w = 2.5
    card_h = 3.5
    # point_set equivalent: TL, TR, BR, BL
    card_object_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [card_w, 0.0, 0.0],
            [card_w, card_h, 0.0],
            [0.0, card_h, 0.0],
        ],
        dtype=np.float32,
    )

    camera_mat = np.eye(3, dtype=np.float64)
    camera_mat[0, 2] = img_size[0] / 2.0
    camera_mat[1, 2] = img_size[1] / 2.0
    dist_coeffs = np.zeros((1, 5), dtype=np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    intrinsics_loaded = False
    pose_mode = False
    model_mode = False

    intrinsic_paths = [
        "camera_intrinsics.yaml",
        "../project4/build/camera_intrinsics.yaml",
        "../../project4/build/camera_intrinsics.yaml",
    ]

    print("Controls: q=quit  p=toggle pose/axes  m=model  s=print corners")

    while True:
        ok, frame = capdev.read()
        if not ok or frame is None or frame.size == 0:
            print("frame is empty")
            break

        frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find rough corner locations (OBB corners == findChessboardCorners output)
        results = model.predict(frame, verbose=False)
        r = results[0]

        corner_set = []
        patternfound = False
        if r.obb is not None and len(r.obb) > 0:
            i = int(r.obb.conf.argmax())
            pts = r.obb.xyxyxyxy.cpu().numpy()[i].tolist()
            corner_set = order_quad_corners(pts)
            patternfound = len(corner_set) == 4

        if patternfound:
            h, w = grey.shape[:2]
            margin = 11
            clamped = []
            for x, y in corner_set:
                x = float(np.clip(x, margin, w - margin - 1))
                y = float(np.clip(y, margin, h - margin - 1))
                clamped.append((x, y))
            corner_set = clamped

            # Refines the corner locations
            corners_np = np.array(corner_set, dtype=np.float32).reshape(-1, 1, 2)
            cv2.cornerSubPix(
                grey,
                corners_np,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
            )
            corner_set = [tuple(pt[0]) for pt in corners_np]

        # draw corner circles on frame (same role as drawChessboardCorners)
        if patternfound:
            for i in range(4):
                pt_i = (int(corner_set[i][0]), int(corner_set[i][1]))
                pt_j = (
                    int(corner_set[(i + 1) % 4][0]),
                    int(corner_set[(i + 1) % 4][1]),
                )
                cv2.line(frame, pt_i, pt_j, (0, 255, 0), 2)
                cv2.circle(frame, pt_i, 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(i),
                    (pt_i[0] + 6, pt_i[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            if patternfound:
                print("corners (TL TR BR BL):")
                for i in range(4):
                    print(f"  {i}: ({corner_set[i][0]:.2f}, {corner_set[i][1]:.2f})")
            else:
                print("no card quad found")
        if key == ord("p"):
            if not intrinsics_loaded:
                ok_load = False
                for path in intrinsic_paths:
                    cam, dist = load_intrinsics(path)
                    if cam is not None:
                        camera_mat = cam
                        dist_coeffs = dist
                        print(f"Loaded intrinsics from {path}")
                        ok_load = True
                        break
                if not ok_load:
                    print(
                        "Could not open camera_intrinsics.yaml — "
                        "calibrate in project4 first, or copy yaml next to the binary"
                    )
                else:
                    intrinsics_loaded = True
                    pose_mode = True
            else:
                pose_mode = not pose_mode
                print(f"pose_mode = {'on' if pose_mode else 'off'}")
        if key == ord("m"):
            if not intrinsics_loaded:
                ok_load = False
                for path in intrinsic_paths:
                    cam, dist = load_intrinsics(path)
                    if cam is not None:
                        camera_mat = cam
                        dist_coeffs = dist
                        print(f"Loaded intrinsics from {path}")
                        ok_load = True
                        break
                if not ok_load:
                    print(
                        "Could not open camera_intrinsics.yaml — "
                        "calibrate in project4 first, or copy yaml next to the binary"
                    )
                else:
                    intrinsics_loaded = True
                    model_mode = True
            else:
                model_mode = True

        # Pose: extrinsics from 4 card corners, then project object points to image
        if pose_mode and patternfound and intrinsics_loaded:
            corners_np = np.array(corner_set, dtype=np.float32)
            solved, rvec, tvec = cv2.solvePnP(
                card_object_pts,
                corners_np,
                camera_mat,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if solved:
                R, _ = cv2.Rodrigues(rvec)
                t_cam = -R.T @ tvec
                R_cam = R.T
                print(
                    f"\r t_cam: {t_cam.T.ravel()}  R_cam: {R_cam.reshape(-1).tolist()}",
                    end="",
                    flush=True,
                )

                # Project card plane corners onto the image plane
                proj_corner_set, _ = cv2.projectPoints(
                    card_object_pts, rvec, tvec, camera_mat, dist_coeffs
                )
                for pt in proj_corner_set:
                    x = float(pt[0][0])
                    y = float(pt[0][1])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    if abs(x) > 1e5 or abs(y) > 1e5:
                        continue
                    cv2.circle(frame, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)

                cv2.drawFrameAxes(frame, camera_mat, dist_coeffs, rvec, tvec, 1.0)

        # Model: simple pyramid (project4 default when use_nidoking_model=false)
        # Coordinates are in card frame (TL origin, Z out of / into plane)
        if model_mode and patternfound and intrinsics_loaded:
            model_point_set = np.array(
                [
                    [0.0, 0.0, -1.0],
                    [2.0, 0.0, -1.0],
                    [2.0, 2.0, -1.0],
                    [0.0, 2.0, -1.0],
                    [1.0, 1.0, -3.0],
                ],
                dtype=np.float32,
            )
            thickness = 3

            corners_np = np.array(corner_set, dtype=np.float32)
            solved, rvec, tvec = cv2.solvePnP(
                card_object_pts,
                corners_np,
                camera_mat,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if solved:
                proj_model_corner_set, _ = cv2.projectPoints(
                    model_point_set, rvec, tvec, camera_mat, dist_coeffs
                )
                proj_pts = []
                for i in range(len(proj_model_corner_set)):
                    x = float(proj_model_corner_set[i][0][0])
                    y = float(proj_model_corner_set[i][0][1])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        proj_pts.append(None)
                        continue
                    # skip wild projections (bad pose / lost tracking)
                    if abs(x) > 1e5 or abs(y) > 1e5:
                        proj_pts.append(None)
                        continue
                    proj_pts.append((int(round(x)), int(round(y))))

                for i in range(len(proj_pts)):
                    pi = proj_pts[i]
                    if pi is None:
                        continue
                    cv2.circle(frame, pi, 5, (0, 0, 255), -1)
                    for j in range(i + 1, len(proj_pts)):
                        pj = proj_pts[j]
                        if pj is None:
                            continue
                        color = (
                            int((37 * i + 17 * j) % 256),
                            int((59 * i + 23 * j) % 256),
                            int((97 * i + 31 * j) % 256),
                        )
                        cv2.line(frame, pi, pj, color, thickness)

                # TODO (later): replace pyramid with Roserade FBX / skinned mesh

        cv2.imshow("Video", frame)

    capdev.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
