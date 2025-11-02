import streamlit as st
import pandas as pd
import re
from io import StringIO
import plotly.graph_objects as go
from scipy.signal import find_peaks
import plotly.io as pio
import plotly.express as px
import os  # Import os for filename manipulation
import base64
import tempfile
import pymzml.run
import numpy as np  # Import numpy for new parser
import time  # Import time for file deletion logic

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(page_title="Peak Peek 🧪", layout="wide")

# --- Security Check ---
# Encoded strings (base64)
encoded_title = "VmVyaWZ5IHlvdSBhcmUgYSBodW1hbg=="
encoded_question = "V2hhdCdzIDMwIGRpdmlkZWQgYnkgMj8="
encoded_options = ["MTU=", "MTg="]  # base64 of "15" and "18"

# Decode at runtime
title = base64.b64decode(encoded_title).decode("utf-8")
question = base64.b64decode(encoded_question).decode("utf-8")
options = [base64.b64decode(o).decode("utf-8") for o in encoded_options]

st.subheader(title)

# Radio with no pre-selection
answer = st.radio(question, options, index=None, horizontal=True)

# Handle no selection
if answer is None:
    st.info("Please answer.")
    st.stop()

# Correct answer is first option (15)
correct_answer = options[0]

if answer != correct_answer:
    st.error("Unbelievably moronic. 😤")
    st.stop()
else:
    st.success("Wow - applause. 🤛🤛🤛")
# --- End Security Check ---


st.title("👀 Peak Peek")


# --- Parsers ---

def parse_blocks(file_content):
    """
    Parses an ASCII chromatogram file into a dictionary of DataFrames.
    Supports MS (TIC), CAD (AD2), and PDA (wavelengths) with unique naming.
    """
    try:
        text = file_content.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error decoding file: {e}")
        return {}

    blocks = re.split(r"(?=\[)", text)
    parsed = {}

    for block in blocks:
        if not block.strip() or not block.startswith("["):
            continue

        header_match = re.search(r"\[(.*?)\]", block)
        if not header_match:
            continue

        original_header = header_match.group(1).strip()  # e.g., "MS Chromatogram"
        label = original_header  # Default label

        # --- Customize Label Based on Block Type ---

        # PDA: "PDA Multi Chromatogram(Ch2)"
        if "PDA Multi Chromatogram" in original_header:
            wl_match = re.search(r"Wavelength\(nm\)\s+(\d+)", block)
            if wl_match:
                wavelength = wl_match.group(1)
                ch_match = re.search(r"\(Ch(\d+)\)", original_header)
                if ch_match:
                    label = f"PDA Ch{ch_match.group(1)} ({wavelength} nm)"
                else:
                    label = f"PDA ({wavelength} nm)"
            # If no wavelength, label remains original_header

        # MS TIC: "MS Chromatogram"
        elif "MS Chromatogram" in original_header:
            # Try to get the specific m/z description line
            mz_match = re.search(r"m/z\s+(.*?)\n", block, re.IGNORECASE)
            if mz_match:
                label = mz_match.group(1).strip()  # e.g., "1-1MS(E+) TIC"
            else:
                # Fallback to simple polarity
                if "(E+)" in block:
                    label = "MS TIC (Positive)"
                elif "(E-)" in block:
                    label = "MS TIC (Negative)"
                # else, label remains "MS Chromatogram"

        # CAD: "LC Chromatogram(AD2)"
        elif "LC Chromatogram" in original_header:
            if "AD2" in original_header:
                label = "CAD (AD2)"
            # else, label remains original_header (e.g., "LC Chromatogram(AD2)")

        # --- Find numeric data region ---
        data_start_match = re.search(
            r"^(.*?R\.?Time.*?(?:Intensity|Counts|Absolute Intensity).*?)\n",
            block, re.MULTILINE | re.IGNORECASE
        )

        if not data_start_match:
            continue

        data_header_line = data_start_match.group(1).strip()
        data_str = block[data_start_match.end():].strip()

        if not data_str or not re.search(r"\d", data_str):
            continue

        # FIX 1: Clean header line to remove parentheticals like (min)
        data_header_line_clean = re.sub(r"\s*\([^)]*\)", "", data_header_line)

        # Prepend header for read_csv
        data_with_header = data_header_line_clean + "\n" + data_str

        try:
            df = pd.read_csv(
                StringIO(data_with_header),
                sep=r"\s+",
                engine="python",
                comment="#",
                on_bad_lines="skip"
            )
        except Exception as e:
            st.warning(f"Failed to parse data block for '{label}': {e}")
            continue

        # Identify Time and Intensity columns dynamically
        time_col = next((c for c in df.columns if "time" in c.lower()), None)

        # FIX 2: Prioritize "Absolute" or "Counts" over plain "Intensity"
        intensity_col = None
        absolute_col = next((c for c in df.columns if "absolute" in c.lower()), None)
        counts_col = next((c for c in df.columns if "counts" in c.lower()), None)
        plain_intensity_col = next((c for c in df.columns if c.lower() == "intensity"), None)  # Exact match

        if absolute_col:
            intensity_col = absolute_col
        elif counts_col:
            intensity_col = counts_col
        elif plain_intensity_col:
            intensity_col = plain_intensity_col
        else:
            # Fallback to any column containing "intensity"
            intensity_col = next((c for c in df.columns if "intensity" in c.lower()), None)

        if not time_col or not intensity_col:
            st.warning(f"Could not find Time/Intensity columns for '{label}'. Skipping.")
            continue

        try:
            df_clean = df[[time_col, intensity_col]].copy()
            df_clean.columns = ["Time", "Intensity"]

            # Coerce to numeric and drop bad lines
            df_clean = df_clean.apply(pd.to_numeric, errors="coerce").dropna()

            # For CAD (AD2), drop negative or huge RTs
            if "LC Chromatogram" in original_header or "AD2" in original_header:
                df_clean = df_clean[df_clean["Time"] >= 0]

        except Exception as e:
            st.warning(f"Error cleaning data for '{label}': {e}")
            continue

        if df_clean.empty:
            continue

        # --- Handle Label Duplicates before adding to dict ---
        final_label = label
        i = 2
        while final_label in parsed:
            final_label = f"{label} ({i})"
            i += 1

        parsed[final_label] = df_clean

    return parsed


@st.cache_data
def detect_mzml_polarities(list_of_file_bytes):
    """
    Reads a list of mzML file bytes, saves each to a temp file,
    and checks for positive/negative scan modes.
    """
    pos_found, neg_found = False, False

    for file_bytes in list_of_file_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mzML")
        tmp.write(file_bytes)
        tmp_path = tmp.name
        tmp.close()

        run = None
        try:
            run = pymzml.run.Reader(tmp_path)
            for spectrum in run:
                if spectrum.ms_level == 1:
                    if spectrum.get("positive scan"):
                        pos_found = True
                    elif spectrum.get("negative scan"):
                        neg_found = True
                if pos_found and neg_found:
                    break  # Found both, no need to check this file anymore

        except Exception as e:
            st.warning(f"Failed to read temp mzML for polarity detection: {e}")

        finally:
            if run:
                try:
                    run.close()
                except Exception:
                    pass
            try:
                os.remove(tmp_path)
            except PermissionError:
                pass  # Optional fallback

        if pos_found and neg_found:
            break  # Found both, no need to check other files

    options = []
    if pos_found:
        options.append("Positive")
    if neg_found:
        options.append("Negative")
    return options


# --- NEW, FIXED mzML Extraction Function ---
# Note: Not cached with @st.cache_data because file_bytes is not hashable
def extract_mzml_chromatogram(file_name, file_bytes, mz_target, mz_tolerance, polarity_filter):
    """
    Extracts a chromatogram (TIC or EIC) from mzML file bytes.
    Uses robust peak parsing and filters by polarity and m/z.

    :param file_name: Name of the file (for error logging)
    :param file_bytes: Raw bytes of the mzML file
    :param mz_target: None for TIC, or a float m/z for EIC
    :param mz_tolerance: Float tolerance for EIC
    :param polarity_filter: "Positive", "Negative", or None (for Both)
    :return: A pandas DataFrame with "Time" and "Intensity"
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mzML")
    tmp.write(file_bytes)
    tmp_path = tmp.name
    tmp.close()

    data = []
    run = None
    try:
        run = pymzml.run.Reader(tmp_path)

        for spec in run:
            # 1. Check MS Level
            ms_level = getattr(spec, "ms_level", np.nan)
            if ms_level != 1:
                continue

            # 2. Check Polarity
            if polarity_filter == "Positive" and not spec.get("positive scan"):
                continue
            if polarity_filter == "Negative" and not spec.get("negative scan"):
                continue

            rt = spec.scan_time_in_minutes() if hasattr(spec, "scan_time_in_minutes") else np.nan
            if pd.isna(rt):
                continue

            # 3. Get Peaks (Robustly, from new script)
            peaks = None
            for peak_type in ["raw", "centroided", None]:  # Try different peak types
                try:
                    peaks = spec.peaks(peak_type) if peak_type else spec.peaks()
                    break
                except Exception:
                    continue

            if peaks is None:
                peaks = np.array([])
            else:
                # Ensure it's a numpy array for filtering
                peaks = np.array(peaks)

            # 4. Calculate Intensity (TIC or EIC)
            intensity = 0.0
            if peaks.size > 0:
                if mz_target is None:
                    # This is a TIC request
                    intensity = float(np.sum(peaks[:, 1]))
                else:
                    # This is an EIC request
                    min_mz = mz_target - mz_tolerance
                    max_mz = mz_target + mz_tolerance
                    # Filter peaks within the m/z range
                    eic_peaks = peaks[(peaks[:, 0] >= min_mz) & (peaks[:, 0] <= max_mz)]
                    if eic_peaks.size > 0:
                        intensity = float(np.sum(eic_peaks[:, 1]))

            data.append((rt, intensity))

    except Exception as e:
        st.warning(f"mzML parsing failed for {file_name}: {e}")

    finally:
        if run is not None:
            try:
                run.close()
            except Exception:
                pass
        # Robust file deletion
        try:
            os.remove(tmp_path)
        except PermissionError:
            time.sleep(0.5)
            try:
                os.remove(tmp_path)
            except Exception as e:
                st.warning(f"Could not delete temp file {tmp_path}: {e}")

    if data:
        return pd.DataFrame(data, columns=["Time", "Intensity"])
    else:
        # Return empty df on failure
        return pd.DataFrame(columns=["Time", "Intensity"])


# --- Plotting Function 1: Single File Overview ---
def create_single_file_figure(file_data, selected_chromatograms, file_name,
                              rt_min, rt_max, smooth_window,
                              stacked, normalize, line_width, color_palette, single_color,
                              rt_marks_input, stack_step):
    """
    Creates the Plotly figure for a SINGLE file, showing multiple selected chromatograms.
    """
    fig = go.Figure()
    offsets = {}
    step = stack_step

    rt_marks = []
    rt_input_lower = rt_marks_input.strip().lower()  # Check once
    mark_max_peak = rt_input_lower == "max"
    mark_all_peaks = rt_input_lower == "all"  # New check

    if not mark_max_peak and not mark_all_peaks and rt_marks_input:  # Modified check
        try:
            rt_marks = [float(x.strip()) for x in rt_marks_input.split(",") if x.strip()]
        except ValueError:
            st.warning("Could not parse 'Mark specific RTs'. Please use comma-separated numbers.")

    # --- Loop iterates over selected chromatograms ---
    for chrom_name in selected_chromatograms:
        df_original = file_data.get(chrom_name)
        if df_original is None or df_original.empty:
            st.warning(f"'{chrom_name}' data is empty or not found.")
            continue

        df = df_original.copy()
        df = df[(df["Time"] >= rt_min) & (df["Time"] <= rt_max)]
        if df.empty:
            continue

        # Apply smoothing
        if smooth_window > 1 and len(df) > smooth_window:
            df["Intensity_smooth"] = df["Intensity"].rolling(window=smooth_window, center=True).mean()
            df["Intensity_smooth"].fillna(df["Intensity"], inplace=True)
        else:
            df["Intensity_smooth"] = df["Intensity"]

        # --- Peak finding ---
        if mark_all_peaks:  # Check for "all"
            if not df["Intensity_smooth"].empty and df["Intensity_smooth"].max() > 0:
                max_intensity = df["Intensity_smooth"].max()
                min_prominence = max_intensity * 0.05
                try:
                    peaks, _ = find_peaks(df["Intensity_smooth"], prominence=min_prominence, distance=5)
                    for p in peaks:
                        rt = df["Time"].iloc[p]
                        rt_marks.append(rt)
                except Exception as e:
                    st.warning(f"Peak finding failed for {chrom_name}: {e}")

        if mark_max_peak:  # "max" input for highest peak
            if not df["Intensity_smooth"].empty:
                try:
                    max_idx = df["Intensity_smooth"].idxmax()
                    rt_marks.append(df["Time"].loc[max_idx])
                except Exception as e:
                    st.warning(f"Failed to find max peak for {chrom_name}: {e}")

        # --- Apply stacking/normalization ---
        y_data = df["Intensity_smooth"]
        if normalize:
            min_val = y_data.min()
            max_val = y_data.max()
            if max_val > min_val:
                y_data = (y_data - min_val) / (max_val - min_val)
            else:
                y_data = pd.Series([0.5] * len(y_data), index=y_data.index)

        offset_value = 0
        if stacked:
            offset_value = len(offsets) * step

        offsets[chrom_name] = offset_value
        y_values = y_data + offset_value

        trace_color = None
        if color_palette == "Single Color":
            trace_color = single_color

        fig.add_trace(go.Scatter(
            x=df["Time"],
            y=y_values,
            mode="lines",
            name=chrom_name,  # Legend shows the chromatogram name
            line=dict(width=line_width, color=trace_color)
        ))

    # --- Add RT Markers ---
    rt_marks = sorted(set(round(rt, 2) for rt in rt_marks))
    for rt in rt_marks:
        fig.add_vline(x=rt, line_dash="dot", line_color="black", line_width=1, opacity=0.8,
                      layer="above")
        fig.add_annotation(
            x=rt, y=1.02, yref="paper", text=f"{rt:.2f}",
            showarrow=False, textangle=-90, font=dict(size=9, color="grey"), yanchor="bottom"
        )

    # --- Layout Updates ---
    yaxis_title = "Intensity"
    if normalize and stacked:
        yaxis_title = "Normalized Intensity (Stacked)"
    elif normalize:
        yaxis_title = "Normalized Intensity (Overlaid)"
    elif stacked:
        yaxis_title = "Intensity (Stacked, No-Norm)"

    if color_palette != "Default" and color_palette != "Single Color":
        palette_map = {
            "Vivid (Set1)": px.colors.qualitative.Set1,
            "Muted (Plotly)": px.colors.qualitative.Plotly,
            "Dark (Set2)": px.colors.qualitative.Set2,
            "Colorblind (T10)": px.colors.qualitative.T10,
            "Pastel": px.colors.qualitative.Pastel,
            "Bold": px.colors.qualitative.Bold,
        }
        fig.update_layout(colorway=palette_map.get(color_palette))

    fig.update_layout(
        xaxis_title="Retention Time (min)",
        yaxis_title=yaxis_title,
        template="simple_white",
        title=f"File: {file_name}",  # Title is the file name
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=50, b=50),
    )
    if stacked:
        fig.update_layout(yaxis=dict(showticklabels=False, zeroline=False))

    return fig


# --- Plotting Function 2: Multi-File Comparison ---
def create_comparison_figure(master_data, selected_files, target_chromatogram,
                             rt_min, rt_max, smooth_window,
                             stacked, normalize, line_width, color_palette, single_color,
                             rt_marks_input, stack_step):
    """
    Creates the Plotly figure for MULTIPLE files, showing one selected chromatogram type.
    """
    fig = go.Figure()
    offsets = {}
    step = stack_step

    rt_marks = []
    rt_input_lower = rt_marks_input.strip().lower()  # Check once
    mark_max_peak = rt_input_lower == "max"
    mark_all_peaks = rt_input_lower == "all"  # New check

    if not mark_max_peak and not mark_all_peaks and rt_marks_input:  # Modified check
        try:
            rt_marks = [float(x.strip()) for x in rt_marks_input.split(",") if x.strip()]
        except ValueError:
            st.warning("Could not parse 'Mark specific RTs'. Please use comma-separated numbers.")

    # --- Loop iterates over selected files ---
    for file_name in selected_files:
        file_data = master_data.get(file_name)
        if not file_data:
            continue

        df_original = file_data.get(target_chromatogram)
        if df_original is None or df_original.empty:
            st.warning(f"'{target_chromatogram}' not found in file: {file_name}")
            continue

        df = df_original.copy()
        df = df[(df["Time"] >= rt_min) & (df["Time"] <= rt_max)]
        if df.empty:
            continue

        # Apply smoothing
        if smooth_window > 1 and len(df) > smooth_window:
            df["Intensity_smooth"] = df["Intensity"].rolling(window=smooth_window, center=True).mean()
            df["Intensity_smooth"].fillna(df["Intensity"], inplace=True)
        else:
            df["Intensity_smooth"] = df["Intensity"]

        # --- Peak finding ---
        if mark_all_peaks:  # Check for "all"
            if not df["Intensity_smooth"].empty and df["Intensity_smooth"].max() > 0:
                max_intensity = df["Intensity_smooth"].max()
                min_prominence = max_intensity * 0.05
                try:
                    peaks, _ = find_peaks(df["Intensity_smooth"], prominence=min_prominence, distance=5)
                    for p in peaks:
                        rt = df["Time"].iloc[p]
                        rt_marks.append(rt)
                except Exception as e:
                    st.warning(f"Peak finding failed for {file_name}: {e}")

        if mark_max_peak:  # "max" input for highest peak
            if not df["Intensity_smooth"].empty:
                try:
                    max_idx = df["Intensity_smooth"].idxmax()
                    rt_marks.append(df["Time"].loc[max_idx])
                except Exception as e:
                    st.warning(f"Failed to find max peak for {file_name}: {e}")

        # --- Apply stacking/normalization ---
        y_data = df["Intensity_smooth"]
        if normalize:
            min_val = y_data.min()
            max_val = y_data.max()
            if max_val > min_val:
                y_data = (y_data - min_val) / (max_val - min_val)
            else:
                y_data = pd.Series([0.5] * len(y_data), index=y_data.index)

        offset_value = 0
        if stacked:
            offset_value = len(offsets) * step

        offsets[file_name] = offset_value
        y_values = y_data + offset_value

        trace_color = None
        if color_palette == "Single Color":
            trace_color = single_color

        fig.add_trace(go.Scatter(
            x=df["Time"],
            y=y_values,
            mode="lines",
            name=file_name,  # Legend shows the filename
            line=dict(width=line_width, color=trace_color)
        ))

    # --- Add RT Markers ---
    rt_marks = sorted(set(round(rt, 2) for rt in rt_marks))
    for rt in rt_marks:
        fig.add_vline(x=rt, line_dash="dot", line_color="black", line_width=1, opacity=0.8,
                      layer="above")
        fig.add_annotation(
            x=rt, y=1.02, yref="paper", text=f"{rt:.2f}",
            showarrow=False, textangle=-90, font=dict(size=9, color="grey"), yanchor="bottom"
        )

    # --- Layout Updates ---
    yaxis_title = target_chromatogram
    if normalize and stacked:
        yaxis_title += " (Normalized, Stacked)"
    elif normalize:
        yaxis_title += " (Normalized, Overlaid)"
    elif stacked:
        yaxis_title += " (Stacked, No-Norm)"

    # --- Color Palette Logic (FIXED) ---
    if color_palette != "Default" and color_palette != "Single Color":
        palette_map = {
            "Vivid (Set1)": px.colors.qualitative.Set1,
            "Muted (Plotly)": px.colors.qualitative.Plotly,
            "Dark (Set2)": px.colors.qualitative.Set2,
            "Colorblind (T10)": px.colors.qualitative.T10,
            "Pastel": px.colors.qualitative.Pastel,
            "Bold": px.colors.qualitative.Bold,
        }
        fig.update_layout(colorway=palette_map.get(color_palette))

    fig.update_layout(
        xaxis_title="Retention Time (min)",
        yaxis_title=yaxis_title,  # Set dynamic title
        template="simple_white",
        title=f"Comparison: {target_chromatogram}",  # Set dynamic title
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=50, b=50),
    )

    if stacked:
        fig.update_layout(
            yaxis=dict(
                showticklabels=False,
                zeroline=False
            )
        )

    return fig


# --- Sidebar: Parameters ---
st.sidebar.header("📈 General Plot Settings")

rt_min = st.sidebar.number_input("Min RT (min)", 0.0, 100.0, 0.0, 0.1)
rt_max = st.sidebar.number_input("Max RT (min)", 0.0, 100.0, 30.0, 0.1)
smooth_window = st.sidebar.slider("Smoothing window (odd integer)", 1, 101, 7, 2)
if smooth_window % 2 == 0:  # Enforce odd number
    smooth_window += 1

st.sidebar.header("🖥️ Display Mode")
normalize = st.sidebar.checkbox("Normalize chromatograms", True)
stacked = st.sidebar.checkbox("Stack chromatograms vertically", True)

stack_step = 1.0  # Default value
if stacked:
    stack_step = st.sidebar.slider("Stack offset step", 0.0, 5.0, 1.0, 0.2)

rt_marks_input = st.sidebar.text_input("Mark specific RTs (comma-separated, 'max', or 'all')", "")
# show_peaks = st.sidebar.checkbox("Mark all peaks", False) # Removed this line

st.sidebar.header("✨ Plot Style")
line_width = st.sidebar.slider("Line width", 1, 10, 2, 1)
color_palette = st.sidebar.selectbox("Color Palette",
                                     ["Default", "Single Color", "Vivid (Set1)", "Muted (Plotly)", "Dark (Set2)",
                                      "Colorblind (T10)", "Pastel", "Bold"])

single_color = None
if color_palette == "Single Color":
    single_color = st.sidebar.color_picker("Pick a color", "#FF0000")

st.sidebar.header("📨 Export Options")
svg_transparent = st.sidebar.checkbox("SVG transparent background", False)

# --- File Uploader ---
uploaded_files = st.file_uploader("Upload ASCII (.txt, .asc, .dat) or mzML files", type=["txt", "asc", "dat", "mzml"],
                                  accept_multiple_files=True)

# --- mzML Sidebar Section (Dynamic) ---
st.sidebar.header("🔍 EIC Extraction (mzML)")

# Store file bytes in a list to avoid re-reading
file_bytes_list = []
mzml_files_present = False
if uploaded_files:
    for f in uploaded_files:
        file_bytes = f.read()
        f.seek(0)  # Reset file pointer
        file_bytes_list.append((f.name, file_bytes))
        if f.name.lower().endswith(".mzml"):
            mzml_files_present = True

# --- DYNAMIC Polarity Detection (FIXED) ---
if mzml_files_present:
    # Get all file bytes for mzML files
    mzml_byte_content = [b for name, b in file_bytes_list if name.lower().endswith(".mzml")]
    # Call the fixed detection function
    available_polarities = detect_mzml_polarities(mzml_byte_content)

    polarity_options = ["Both"] + available_polarities
    if not available_polarities:
        polarity_options = ["Both", "Positive", "Negative"]  # Fallback if none found

    polarity_selection = st.sidebar.radio(
        "Ion Mode (mzML)",
        polarity_options,
        index=0,
        help="Select ion mode. 'Both' will combine data from all scans."
    )
else:
    # Show disabled controls if no mzML files are present
    polarity_selection = st.sidebar.radio(
        "Ion Mode (mzML)",
        ["Both", "Positive", "Negative"],
        index=0,
        disabled=True
    )

extract_tic = st.sidebar.checkbox("Extract TIC from mzML files", disabled=not mzml_files_present)
mz_input_str = st.sidebar.text_input("m/z values (comma-separated)", "", disabled=not mzml_files_present)
tolerance = st.sidebar.number_input("m/z Tolerance (Da)", 0.0, 5.0, 0.5, 0.1, disabled=not mzml_files_present)

# --- Mood Selector ---
st.sidebar.header("🧠 Mood Check")
mood = st.sidebar.selectbox(
    "How are your results?",
    ["Neutral", "Amazing", "Shitty"],
    index=0
)

# Define public image URLs (works anywhere)
background_images = {
    "Amazing": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdTU4MzV3dGhhOHJkOWo2dms1ZTV2cnptZ2U0bXEyOGVrajlqZ2psNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/glvyCVWYJ21fq/giphy.gif",  # Bright sunrise
    "Shitty": "https://i.imgur.com/7fiow2I.gif",   # Moody rain
}

# --- MAIN APP LOGIC ---
if uploaded_files:

    # --- Data Parsing ---
    master_data = {}
    all_chrom_names = set()

    # Get polarity string for processing
    polarity_str = None
    if polarity_selection != "Both":
        polarity_str = polarity_selection  # "Positive" or "Negative"

    for file_name, file_bytes in file_bytes_list:

        # 1. Parse ASCII files
        if file_name.lower().endswith((".txt", ".asc", ".dat")):
            try:
                file_data = parse_blocks(file_bytes)
                if file_data:
                    master_data[file_name] = file_data
                    all_chrom_names.update(file_data.keys())
            except Exception as e:
                st.error(f"Failed to read ASCII file {file_name}: {e}")

        # 2. Parse mzML files (FIXED)
        elif file_name.lower().endswith(".mzml"):
            file_chroms = {}

            # 2a. Extract TIC if requested
            if extract_tic:
                # Call fixed function: mz_target=None for TIC
                tic_df = extract_mzml_chromatogram(file_name, file_bytes, None, 0, polarity_str)
                if not tic_df.empty:
                    tic_label = f"TIC ({polarity_str or 'Both'})"
                    file_chroms[tic_label] = tic_df

            # 2b. Extract EICs if m/z values are provided
            if mz_input_str:
                try:
                    mz_values = [float(mz.strip()) for mz in mz_input_str.split(",") if mz.strip()]
                    for mz in mz_values:
                        # Call fixed function: pass mz_target
                        eic_df = extract_mzml_chromatogram(file_name, file_bytes, mz, tolerance, polarity_str)
                        if not eic_df.empty:
                            eic_label = f"EIC {mz} +/- {tolerance} Da ({polarity_str or 'Both'})"
                            # Handle duplicate EIC labels (e.g., if user enters '100, 100')
                            i = 2
                            final_label = eic_label
                            while final_label in file_chroms:
                                final_label = f"{eic_label} ({i})"
                                i += 1
                            file_chroms[final_label] = eic_df

                except ValueError:
                    st.error("Invalid m/z value. Please enter comma-separated numbers.")
                except Exception as e:
                    st.error(f"Failed to extract EIC from {file_name}: {e}")

            if file_chroms:
                if file_name not in master_data:
                    master_data[file_name] = {}
                master_data[file_name].update(file_chroms)
                all_chrom_names.update(file_chroms.keys())

    if not master_data:
        st.error("⚠️ Select m/z values for EICs in the side bar.")
        st.stop()

    # --- CONDITIONAL UI: Switch based on file count ---
    fig = None
    export_filename_base = "chromatogram"
    sorted_chrom_list = sorted(list(all_chrom_names))

    # --- Behavior 1: Single File Uploaded (Overview Mode) ---
    if len(uploaded_files) == 1:
        file_name = uploaded_files[0].name
        # Check if we actually have data for this file (it might have been an mzML with no EIC requested)
        if file_name not in master_data or not master_data[file_name]:
            st.warning("No data extracted. Please select 'Extract TIC' or enter m/z values for mzML files.")
            st.stop()

        file_data = master_data[file_name]
        file_chrom_list = sorted(file_data.keys())

        st.subheader("Single File Overview Mode")
        selected_chromatograms = st.multiselect(
            "Select chromatograms to display:",
            file_chrom_list,
            default=file_chrom_list[:min(len(file_chrom_list), 5)]  # Default to first 5
        )

        if not selected_chromatograms:
            st.info("Please select at least one chromatogram to display.")
            st.stop()

        fig = create_single_file_figure(
            file_data, selected_chromatograms, file_name,
            rt_min, rt_max, smooth_window,
            stacked, normalize, line_width, color_palette, single_color,
            rt_marks_input, stack_step
        )
        export_filename_base = f"{os.path.splitext(file_name)[0]}_overview"

    # --- Behavior 2: Multiple Files Uploaded (Comparison Mode) ---
    else:
        st.subheader("Multi-File Comparison Mode")

        if not sorted_chrom_list:
            st.warning("No chromatograms available to compare. Check mzML extraction settings.")
            st.stop()

        target_chromatogram = st.selectbox("Select chromatogram to compare:", sorted_chrom_list)

        sorted_file_list = sorted(master_data.keys())
        selected_files = st.multiselect(
            "Select files to display:",
            sorted_file_list,
            default=sorted_file_list[:min(len(sorted_file_list), 5)]  # Default to first 5
        )

        if not selected_files or not target_chromatogram:
            st.info("Please select at least one file and one chromatogram to compare.")
            st.stop()

        fig = create_comparison_figure(
            master_data, selected_files, target_chromatogram,
            rt_min, rt_max, smooth_window,
            stacked, normalize, line_width, color_palette, single_color,
            rt_marks_input, stack_step
        )
        # Clean target_chromatogram name for the filename
        clean_chrom_name = re.sub(r'[\s/():+~-]+', '_', target_chromatogram).strip('_')
        export_filename_base = f"comparison_of_{clean_chrom_name}"

    # --- Common Plot and Export Logic ---
    if fig:
        if mood == "Amazing":
            fig.add_layout_image(
                dict(
                    source=background_images["Amazing"],
                    xref="paper", yref="paper",
                    x=0, y=1,
                    sizex=1, sizey=1,
                    xanchor="left", yanchor="top",
                    sizing="stretch",
                    opacity=0.2,
                    layer="below"
                )
            )
        elif mood == "Shitty":
            fig.add_layout_image(
                dict(
                    source=background_images["Shitty"],
                    xref="paper", yref="paper",
                    x=0, y=1,
                    sizex=1, sizey=1,
                    xanchor="left", yanchor="top",
                    sizing="stretch",
                    opacity=0.2,
                    layer="below"
                )
            )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💾 Export Plot")
        col1, col2 = st.columns(2)

        with col1:
            try:
                html_bytes = fig.to_html().encode("utf-8")
                st.download_button(
                    "Download as HTML",
                    data=html_bytes,
                    file_name=f"{export_filename_base}.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"HTML Export Error: {e}")

        with col2:
            try:
                fig_export = go.Figure(fig)
                if svg_transparent:
                    fig_export.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                             paper_bgcolor='rgba(0,0,0,0)')

                svg_bytes = pio.to_image(fig_export, format="svg")
                st.download_button(
                    "Download as SVG",
                    data=svg_bytes,
                    file_name=f"{export_filename_base}.svg",
                    mime="image/svg+xml"
                )
            except Exception as e:
                st.error(f"SVG Export Error: {e}")

else:
    st.info("⬆️ Upload one or more ASCII (.txt, .asc, .dat) or .mzML files to get started.")

