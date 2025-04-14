# import cv2
# import numpy as np
# import os

# def fix_border(frame, scale=1.04):
#     h, w = frame.shape[:2]
#     T = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
#     return cv2.warpAffine(frame, T, (w, h))

# def mesh_flow(input_path, output_filename, mesh_size=16, smoothing_radius=50, scale=1.04):
#     cap = cv2.VideoCapture(input_path)
#     n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Output path
#     output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, output_filename)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

#     # Read the first frame
#     _, prev = cap.read()
#     prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

#     # Store mesh displacements
#     mesh_flows = []

#     for i in range(1, n_frames):
#         success, curr = cap.read()
#         if not success:
#             break

#         curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

#         # Optical flow
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
#                                             pyr_scale=0.5, levels=3, winsize=15,
#                                             iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

#         dx = cv2.resize(flow[..., 0], (w // mesh_size, h // mesh_size), interpolation=cv2.INTER_LINEAR)
#         dy = cv2.resize(flow[..., 1], (w // mesh_size, h // mesh_size), interpolation=cv2.INTER_LINEAR)
#         mesh_flows.append((dx, dy))

#         prev_gray = curr_gray

#     # Smooth mesh flows over time
#     mesh_flows = np.array(mesh_flows)
#     smoothed_flows = np.copy(mesh_flows)
#     for t in range(2):  # 0 for dx, 1 for dy
#         for y in range(mesh_flows.shape[1]):
#             for x in range(mesh_flows.shape[2]):
#                 smoothed_flows[:, y, x, t] = np.convolve(mesh_flows[:, y, x, t],
#                                                          np.ones(smoothing_radius) / smoothing_radius,
#                                                          mode='same')

#     # Reset video and apply stabilization
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     _, frame = cap.read()
#     out.write(frame)

#     for i in range(1, n_frames):
#         success, frame = cap.read()
#         if not success:
#             break

#         dx = cv2.resize(smoothed_flows[i-1, ..., 0], (w, h), interpolation=cv2.INTER_LINEAR)
#         dy = cv2.resize(smoothed_flows[i-1, ..., 1], (w, h), interpolation=cv2.INTER_LINEAR)

#         map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
#         map_x = (map_x - dx).astype(np.float32)
#         map_y = (map_y - dy).astype(np.float32)

#         stabilized = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
#         stabilized = fix_border(stabilized, scale)

#         out.write(stabilized)

#     cap.release()
#     out.release()

#     return output_path

import cv2
import numpy as np

def mesh_flow(input_video, output_filename, mesh_size=16, smoothing_radius=50, scale=1.05, debug=False):
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read video frame.")
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize transformations list
    transforms = []

    # Mesh grid creation
    def create_mesh_grid(frame_shape, mesh_size):
        h, w = frame_shape[:2]  # Only take the height and width
        mesh = []
        for y in range(0, h, mesh_size):
            for x in range(0, w, mesh_size):
                mesh.append((x, y))
        return np.array(mesh)

    # Warp frame based on motion vectors
    def warp_frame(frame, transformation, mesh_size):
        h, w = frame.shape[:2]
        mesh = create_mesh_grid(frame.shape, mesh_size)
        grid_transforms = np.array(transformation).reshape((-1, 2))
        
        # Warp the mesh
        new_frame = frame.copy()
        for i, (pt_x, pt_y) in enumerate(mesh):
            transform = grid_transforms[i]
            new_x = int(pt_x + transform[0])
            new_y = int(pt_y + transform[1])

            if 0 <= new_x < w and 0 <= new_y < h:
                new_frame[new_y, new_x] = frame[pt_y, pt_x]

        return new_frame

    # Process all frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (Farneback method)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate mesh grid and motion
        mesh_grid = create_mesh_grid(frame.shape, mesh_size)
        motion_vectors = []

        for pt in mesh_grid:
            fx, fy = flow[pt[1], pt[0]]  # Flow vectors at the mesh grid points
            motion_vectors.append((fx, fy))

        # Append the transformation (motion) for the current frame
        transforms.append(motion_vectors)

        # Apply smoothing on the transformations (motion vectors)
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = smooth(trajectory, radius=smoothing_radius)
        correction = smoothed_trajectory - trajectory
        smoothed_vectors = correction[-1]

        # Warp the frame using smoothed motion vectors
        stabilized_frame = warp_frame(frame, smoothed_vectors, mesh_size)

        # Write the stabilized frame to the output video
        out.write(stabilized_frame)

        # Show the frame if debugging is enabled
        if debug:
            cv2.imshow("Stabilized Frame", stabilized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Update the previous frame for the next iteration
        prev_gray = frame_gray

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_filename

# Smoothing function for trajectories
def smooth(trajectory, radius=50):
    smoothed = np.copy(trajectory)
    for i in range(trajectory.shape[0]):
        if i > 0 and i < trajectory.shape[0] - 1:
            smoothed[i] = np.mean(trajectory[max(i - radius, 0):min(i + radius, trajectory.shape[0])], axis=0)
    return smoothed
