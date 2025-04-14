import cv2
import numpy as np
import os

def fix_border(frame, scale=1.04):
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    return cv2.warpAffine(frame, T, (w, h))

def mesh_flow(input_path, output_filename, mesh_size=16, smoothing_radius=50, scale=1.04):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output path
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Read the first frame
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Store mesh displacements
    mesh_flows = []

    for i in range(1, n_frames):
        success, curr = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        dx = cv2.resize(flow[..., 0], (w // mesh_size, h // mesh_size), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(flow[..., 1], (w // mesh_size, h // mesh_size), interpolation=cv2.INTER_LINEAR)
        mesh_flows.append((dx, dy))

        prev_gray = curr_gray

    # Smooth mesh flows over time
    mesh_flows = np.array(mesh_flows)
    smoothed_flows = np.copy(mesh_flows)
    for t in range(2):  # 0 for dx, 1 for dy
        for y in range(mesh_flows.shape[1]):
            for x in range(mesh_flows.shape[2]):
                smoothed_flows[:, y, x, t] = np.convolve(mesh_flows[:, y, x, t],
                                                         np.ones(smoothing_radius) / smoothing_radius,
                                                         mode='same')

    # Reset video and apply stabilization
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()
    out.write(frame)

    for i in range(1, n_frames):
        success, frame = cap.read()
        if not success:
            break

        dx = cv2.resize(smoothed_flows[i-1, ..., 0], (w, h), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(smoothed_flows[i-1, ..., 1], (w, h), interpolation=cv2.INTER_LINEAR)

        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x - dx).astype(np.float32)
        map_y = (map_y - dy).astype(np.float32)

        stabilized = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        stabilized = fix_border(stabilized, scale)

        out.write(stabilized)

    cap.release()
    out.release()

    return output_path
