import streamlit as st
import time
import os
import tempfile
from utils.codec_conversion import ensure_h264_compliance
from classical.optical_flow.optical_flow import optical_flow
from classical.block_matching.block_matching import block_matching
from classical.bitplane_matching.bitplane_matching import bitplane_matching
from classical.l1_optimal_paths.l1_optimal_paths import l1_optimal_stabilization
from classical.mesh_flow import mesh_flow

st.title("Stable Vision")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

# Sidebar: Method selection and file upload
st.sidebar.header("Settings")
method = st.sidebar.selectbox(
    "Select stabilization method:",
    ("Optical Flow", "Block Matching", "Bitplane Matching", "L1 Optimal Paths", "Mesh Flow")
)

# Show sliders for method-specific arguments
with st.sidebar.expander("Options",expanded=False):
    if method == "Optical Flow":
        # st.subheader("Optical Flow Methods")
        optical_flow_method = st.selectbox(
            "Select optical flow method:",
            ("Lucas-Kanade", "Horn-Schunck", "Farneback")
        )
        feature_tracker = None
        if optical_flow_method == "Lucas-Kanade":
            feature_tracker = st.selectbox(
                "Select feature tracking method:",
                ("GFTT (Shi Tomasi)", "HARRIS", "FAST")
            )

        elif optical_flow_method == "Farneback":
            feature_tracker = st.selectbox(
                "Select feature tracking method:",
                ("GFTT (Shi Tomasi)","HARRIS", "SIFT")
            )
        
        use_kalman = st.checkbox("Use Kalman Filter", value=False)
        smoothing_radius = st.slider("Smoothing Radius", min_value=10, max_value=100, value=30, step=5)
        max_corners = st.slider("Max Corners", min_value=50, max_value=500, value=200, step=10)
        quality_level = st.slider("Quality Level", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
        min_distance = st.slider("Min Distance", min_value=5, max_value=50, value=30, step=5)
        block_size = st.slider("Block Size", min_value=3, max_value=15, value=3, step=2)

    elif method == "Bitplane Matching":
        # st.subheader("Bitplane Matching Parameters")
        smoothing_radius = st.slider("Smoothing Radius", min_value=10, max_value=100, value=50, step=5)
        scale = st.slider("Scale", min_value=1.0, max_value=2.0, value=1.04, step=0.01)

    elif method == "Block Matching":
        # st.subheader("Block Matching Parameters")
        use_kalman = st.checkbox("Use Kalman Filter", value=False)
        smoothing_radius = st.slider("Smoothing Radius", min_value=10, max_value=100, value=30, step=5)
        block_size = st.slider("Block Size", min_value=8, max_value=64, value=16, step=8)
        search_area = st.slider("Search Area", min_value=8, max_value=64, value=16, step=8)

    elif method == "L1 Optimal Paths":
        crop_ratio = st.slider("Crop Ratio", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
    elif method == "Mesh Flow":
    mesh_size = st.slider("Mesh Size", min_value=4, max_value=64, value=16, step=4)
    smoothing_radius = st.slider("Smoothing Radius", min_value=5, max_value=100, value=50, step=5)
    scale = st.slider("Scale", min_value=1.0, max_value=2.0, value=1.04, step=0.01)

        
if uploaded_file is not None:
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name
        input_path = ensure_h264_compliance(input_path)

    # Generate output file name from uploaded file name
    base_name = os.path.splitext(uploaded_file.name)[0]
    output_filename = f"{base_name}_stabilized.mp4"

    with st.spinner(f"Stabilizing video using {method}"):
        # Call the appropriate method
        if method == "Optical Flow":
            method_mapping = {
                "Lucas-Kanade": "lk",
                "Horn-Schunck": "horn-schunck",
                "Farneback": "farneback"
            }
            selected_method = method_mapping[optical_flow_method]
            if feature_tracker is not None:
                st.write(f"Using {selected_method} with {feature_tracker}")
            else:
                st.write(f"Using {selected_method}")
            start_time = time.time()
            output_path = optical_flow(
                input_path, 
                output_filename, 
                smoothing_radius=smoothing_radius, 
                use_kalman=use_kalman, 
                method = selected_method,
                feature_tracker = feature_tracker,
                maxCorners=max_corners, 
                qualityLevel=quality_level, 
                minDistance=min_distance, 
                blockSize=block_size
            )

        elif method == "Block Matching":
            start_time = time.time()
            output_path = block_matching(
                input_path,
                output_filename,
                smoothing_radius=smoothing_radius,
                block_size=block_size,
                search_area=search_area,
                use_kalman=use_kalman
            )

        elif method == "Bitplane Matching":
            start_time = time.time()
            output_path = bitplane_matching(
                input_path, 
                output_filename, 
                smoothing_radius=smoothing_radius, 
                scale=scale
            )
        elif method == "Mesh Flow":
            start_time = time.time()
            output_path = mesh_flow(
                input_path,
                output_filename,
                mesh_size=mesh_size,
                smoothing_radius=smoothing_radius,
                scale=scale)

        elif method == "L1 Optimal Paths":
            start_time = time.time()
            output_path = l1_optimal_stabilization(input_path,
                                                    output_filename,
                                                    crop_ratio=crop_ratio)

        print("Output path:", output_path)
        print("File exists:", os.path.exists(output_path))
        print("File size:", os.path.getsize(output_path))

    st.success(f"Video stabilization complete! Process completed in {time.time()-start_time:.2f} seconds")

    with st.spinner("Finalizing the output... Please wait"):
        time.sleep(1)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Video**")
        with open(input_path, "rb") as f:
            st.video(f.read())

    with col2:
        st.markdown(f"**Stabilized Video via {method}**")
        output_path = ensure_h264_compliance(output_path)
        with open(output_path, "rb") as f:
            st.video(f.read())

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Stabilized Video",
            data=f.read(),
            file_name=output_filename,
            mime="video/mp4"
        )
