import os
import tempfile
import uuid

import streamlit as st

from database import (
    clear_video_data,
    create_connection,
    delete_video,
    init_db,
    register_video,
    update_video_output_path,
)
from interpolation import interpolate_results
from processing import process_video
from visualization import visualize_results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


@st.cache_resource
def get_db_connection():
    conn = create_connection()
    init_db(conn)
    return conn


def main():
    st.title("Automatic License Plate Recognition")
    st.write("Upload a video to detect license plates.")

    if 'download_button_visible' not in st.session_state:
        st.session_state.download_button_visible = False
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.session_state.download_button_visible = False
        st.session_state.video_processed = False
        st.video(uploaded_file)
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                conn = get_db_connection()
                video_id = uuid.uuid4().hex

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                tfile.flush()
                tfile.close()
                video_path = tfile.name

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.info("⏳ Starting video processing...")
                register_video(conn, video_id, uploaded_file.name)
                clear_video_data(conn, video_id)

                max_frames, detection_count = process_video(
                    video_path,
                    progress_bar,
                    conn,
                    video_id,
                )

                if detection_count == 0:
                    delete_video(conn, video_id)
                    status_text.warning("⚠️ No license plates detected.")
                    st.warning("No license plates detected.")
                    return

                os.makedirs(OUTPUT_DIR, exist_ok=True)

                status_text.info("✅ Video processing complete! Interpolating for smoother tracking...")
                interpolate_results(conn, video_id)

                status_text.info("✅ Interpolation complete! Generating output video...")
                output_video_filename = f"{video_id}_output.mp4"
                output_video_path = os.path.join(OUTPUT_DIR, output_video_filename)
                visualize_results(
                    video_path,
                    conn,
                    video_id,
                    output_video_path,
                    max_frames,
                )

                update_video_output_path(conn, video_id, output_video_path)

                status_text.success("✅ Output video generated!")
                
                st.session_state.output_video_path = output_video_path
                st.session_state.video_id = video_id
                st.session_state.video_processed = True
                st.session_state.download_button_visible = True

                progress_bar.progress(1.0, text="Processing complete!")

    if st.session_state.download_button_visible:
        with open(st.session_state.output_video_path, 'rb') as f:
            video_bytes = f.read()
        download_name = "output.mp4"
        if st.session_state.video_id:
            download_name = f"{st.session_state.video_id}_output.mp4"

        st.download_button(
            label="Download Video",
            data=video_bytes,
            file_name=download_name,
            mime="video/mp4"
        )

if __name__ == "__main__":
    main()
