"""Streamlit app for exploring Mars habitability datasets.

This app reproduces the scoring methodology from `untitled0 (1).py` and lets
users tune parameter weights and a top-percentage threshold to surface the most
promising locations under custom priorities.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from pyproj import Proj
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency path
    gpd = None
    Point = None
    Proj = None
    plt = None


st.set_page_config(
    page_title="Habitability Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🪐 Mars Habitability Explorer")
st.caption(
    "The app scans your monthly exports automatically. Adjust weights, filters, and surface the top sites."
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

PARAM_COLUMNS = [
    "lattitude",
    "longitude",
    "atm pressure",
    "atm density",
    "temperture",
    "zonal wind",
    "meridional wind",
] + [f"extvar_{i}" for i in range(79)]


def to_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_score(series: pd.Series) -> pd.Series:
    min_s = series.min()
    max_s = series.max()
    if pd.isna(min_s) or pd.isna(max_s) or np.isclose(max_s, min_s):
        return pd.Series(np.zeros_like(series), index=series.index)
    return 10 * (series - min_s) / (max_s - min_s)


def score_temperature(temperature: float, temp_min: float, temp_max: float) -> float:
    temp_c = temperature - 273.15
    ideal = 24.0
    max_distance = max(
        abs(temp_min - 273.15 - ideal),
        abs(temp_max - 273.15 - ideal),
        1e-9,
    )
    distance = abs(temp_c - ideal)
    score = 10 * (1 - distance / max_distance)
    return max(0.0, score)


def score_windspeed(windspeed: float, wind_min: float, wind_max: float) -> float:
    ideal = 5.0
    max_distance = max(abs(wind_min - ideal), abs(wind_max - ideal), 1e-9)
    distance = abs(windspeed - ideal)
    score = 10 * (1 - distance / max_distance)
    return max(0.0, score)


def score_pressure(pressure_pa: float, pres_min: float, pres_max: float) -> float:
    ideal = 101_325.0
    max_distance = max(abs(pres_min - ideal), abs(pres_max - ideal), 1e-9)
    distance = abs(pressure_pa - ideal)
    score = 10 * (1 - distance / max_distance)
    return max(0.0, score)


def score_energy(energy: float, energy_min: float, energy_max: float) -> float:
    span = max(energy_max - energy_min, 1e-9)
    score = 10 * (energy - energy_min) / span
    return max(0.0, min(10.0, score))


def score_water_access(water: float, water_min: float, water_max: float) -> float:
    span = max(water_max - water_min, 1e-9)
    score = 10 * (water - water_min) / span
    return max(0.0, min(10.0, score))


def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that the uploaded dataframe matches the expected schema.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "Uploaded file is empty. Please upload a file with data."
    
    # Check column count first (most important)
    expected_col_count = len(PARAM_COLUMNS)
    actual_col_count = len(df.columns)
    
    if actual_col_count != expected_col_count:
        return False, f"Column count mismatch: Expected {expected_col_count} columns, found {actual_col_count} columns."
    
    # Try to match column names (case-insensitive, allow some flexibility)
    # Convert to lowercase for comparison
    expected_cols_lower = [col.lower().strip() for col in PARAM_COLUMNS]
    actual_cols_lower = [str(col).lower().strip() for col in df.columns]
    
    # Check if column names match (allowing for some header variations)
    mismatches = []
    for i, (expected, actual) in enumerate(zip(expected_cols_lower, actual_cols_lower)):
        # Allow exact match or if actual column is numeric (unlabeled)
        if expected != actual and not str(df.columns[i]).isdigit():
            # Check if it's close enough (handles typos like "lattitude" vs "latitude")
            if expected not in actual and actual not in expected:
                mismatches.append(f"Column {i+1}: expected '{PARAM_COLUMNS[i]}', found '{df.columns[i]}'")
    
    if mismatches and len(mismatches) > 5:  # If too many mismatches, likely wrong schema
        return False, f"Schema mismatch: {len(mismatches)} column name mismatches. First few: {', '.join(mismatches[:3])}"
    
    return True, ""


def load_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";", names=PARAM_COLUMNS, skiprows=[0])
    df = to_numeric(df, PARAM_COLUMNS)
    df["longitude"] = df["longitude"].apply(lambda x: (x - 360) if x > 180 else x)
    df["windspeed"] = np.sqrt(df["zonal wind"] ** 2 + df["meridional wind"] ** 2)
    df["latitude"] = df["lattitude"]
    df["source_file"] = file_path.name
    return df


def load_datasets(file_paths: Iterable[Path]) -> pd.DataFrame:
    frames = [load_dataset(path) for path in file_paths]
    return pd.concat(frames, ignore_index=True)


def to_geodataframe(df: pd.DataFrame):
    if gpd is None or Point is None:
        raise ImportError("geopandas and shapely are required for map visualization.")
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326").to_crs("ESRI:54009")


def plot_with_best(
    gdf,
    column: str,
    title: str,
    best_places=None,
    best_label: str = "Best 1%",
):
    if Proj is None or plt is None:
        raise ImportError("pyproj and matplotlib are required for map visualization.")

    moll = Proj(proj="moll", lon_0=0)

    lon_labels = range(-180, 181, 60)
    lat_labels = range(-90, 91, 30)

    lon_ticks = [moll(lon, 0)[0] for lon in lon_labels]
    lat_ticks = [moll(0, lat)[1] for lat in lat_labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    gdf.plot(ax=ax, column=column, cmap="coolwarm", markersize=5, legend=True)

    if best_places is not None and not best_places.empty:
        best_places.plot(ax=ax, color="black", markersize=10, label=best_label)

    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([f"{lon}°" for lon in lon_labels])
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([f"{lat}°" for lat in lat_labels])

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if best_places is not None and not best_places.empty:
        plt.legend()

    return fig


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    temp_min, temp_max = df["temperture"].min(), df["temperture"].max()
    wind_min, wind_max = df["windspeed"].min(), df["windspeed"].max()
    pres_min, pres_max = df["atm pressure"].min(), df["atm pressure"].max()
    energy_min, energy_max = df["extvar_32"].min(), df["extvar_32"].max()
    water_min, water_max = df["extvar_41"].min(), df["extvar_41"].max()

    df = df.copy()
    df["temperature_habitability"] = df["temperture"].apply(
        lambda t: score_temperature(t, temp_min, temp_max)
    )
    df["windspeed_habitability"] = df["windspeed"].apply(
        lambda w: score_windspeed(w, wind_min, wind_max)
    )
    df["pressure_habitability"] = df["atm pressure"].apply(
        lambda p: score_pressure(p, pres_min, pres_max)
    )
    df["energy_habitability"] = df["extvar_32"].apply(
        lambda e: score_energy(e, energy_min, energy_max)
    )
    df["water_access_habitability"] = df["extvar_41"].apply(
        lambda w: score_water_access(w, water_min, water_max)
    )

    for col in [
        "temperature_habitability",
        "windspeed_habitability",
        "pressure_habitability",
        "energy_habitability",
        "water_access_habitability",
    ]:
        df[col] = normalize_score(df[col])

    return df


def apply_weights(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    weight_series = pd.Series(weights, dtype=float)
    if weight_series.sum() == 0:
        weight_series = pd.Series({k: 1 for k in weights}, dtype=float)
    normalized = weight_series / weight_series.sum()

    df = df.copy()
    df["overall_habitability"] = 0.0
    for column, weight in normalized.items():
        df["overall_habitability"] += df[column] * weight

    df["overall_habitability"] = normalize_score(df["overall_habitability"])
    return df, normalized.to_dict()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Data & Preferences")
    
    # File upload section
    st.subheader("Upload Data")
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files with the required schema. Files should have columns: " + ", ".join(PARAM_COLUMNS[:7]) + "..."
    )
    
    uploaded_dataframes: List[pd.DataFrame] = []
    
    # Handle uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # First, read a sample to check actual schema (read first row after skip)
                df_temp = pd.read_csv(uploaded_file, sep=";", skiprows=[0], nrows=1, header=None)
                
                # Reset file pointer again for full read
                uploaded_file.seek(0)
                
                # Validate schema before processing
                is_valid, error_msg = validate_schema(df_temp)
                if not is_valid:
                    st.error(f"❌ Schema validation failed for '{uploaded_file.name}': {error_msg}")
                    st.error(f"Expected {len(PARAM_COLUMNS)} columns: {', '.join(PARAM_COLUMNS[:7])}... (and {len(PARAM_COLUMNS)-7} more)")
                    st.error(f"Found {len(df_temp.columns)} columns in the file")
                    st.stop()
                
                # Now read the full file with proper column names
                df = pd.read_csv(uploaded_file, sep=";", names=PARAM_COLUMNS, skiprows=[0])
                
                # Final validation: check if data is actually there
                if df.empty:
                    st.error(f"❌ '{uploaded_file.name}' contains no data rows after processing.")
                    st.stop()
                
                # Process the dataframe
                df = to_numeric(df, PARAM_COLUMNS)
                df["longitude"] = df["longitude"].apply(lambda x: (x - 360) if x > 180 else x)
                df["windspeed"] = np.sqrt(df["zonal wind"] ** 2 + df["meridional wind"] ** 2)
                df["latitude"] = df["lattitude"]
                df["source_file"] = uploaded_file.name
                uploaded_dataframes.append(df)
                st.success(f"✅ '{uploaded_file.name}' loaded successfully ({len(df)} rows)")
            except Exception as e:
                st.error(f"❌ Error loading '{uploaded_file.name}': {str(e)}")
                import traceback
                st.error(f"Details: {traceback.format_exc()}")
                st.stop()
    
    # Check if we have any data source
    if not uploaded_dataframes:
        st.warning("⚠️ No data available. Please upload CSV files.")
        st.info("Expected schema columns: " + ", ".join(PARAM_COLUMNS[:10]) + "... (and extvar_0 to extvar_78)")
        st.stop()

    st.subheader("Parameter Weights")
    temp_weight = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    pressure_weight = st.slider(
        "Atmospheric Pressure", min_value=0.0, max_value=1.0, value=0.25, step=0.05
    )
    water_weight = st.slider(
        "Surface Water Access", min_value=0.0, max_value=1.0, value=0.25, step=0.05
    )
    energy_weight = st.slider("Solar Energy", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
    wind_weight = st.slider("Windspeed", min_value=0.0, max_value=1.0, value=0.10, step=0.05)

    st.markdown("Weighted values are normalized automatically so their sum equals 1.0.")

    top_percentage = st.slider(
        "Show top percentage of sites",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="For example, 1% reveals only the best-scoring 1% of rows.",
    )

    with st.expander("Minimum required score (0-10)"):
        min_temp = st.slider("Temperature score ≥", 0.0, 10.0, 0.0, 0.5)
        min_pressure = st.slider("Pressure score ≥", 0.0, 10.0, 0.0, 0.5)
        min_water = st.slider("Water access score ≥", 0.0, 10.0, 0.0, 0.5)
        min_energy = st.slider("Energy score ≥", 0.0, 10.0, 0.0, 0.5)
        min_wind = st.slider("Windspeed score ≥", 0.0, 10.0, 0.0, 0.5)

    filters = {
        "temperature_habitability": min_temp,
        "pressure_habitability": min_pressure,
        "water_access_habitability": min_water,
        "energy_habitability": min_energy,
        "windspeed_habitability": min_wind,
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

calculate = st.button("Calculate Best Locations", type="primary")

if not calculate:
    if gpd is None:
        st.warning(
            "Install optional plotting libraries for maps with `pip3 install geopandas shapely pyproj matplotlib`."
        )
    st.info("Adjust settings and press **Calculate Best Locations** to generate results.")
    st.stop()


with st.spinner("Processing datasets..."):
    # Check if we have uploaded data
    if not uploaded_dataframes:
        st.error("No data available to process. Please upload CSV files.")
        st.stop()
    
    # Combine all uploaded dataframes
    raw_df = pd.concat(uploaded_dataframes, ignore_index=True)
    
    # Final check for empty data
    if raw_df.empty:
        st.error("No data available after processing. Please check your uploaded files.")
        st.stop()
    
    scored_df = compute_scores(raw_df)

weights_input = {
    "temperature_habitability": temp_weight,
    "pressure_habitability": pressure_weight,
    "water_access_habitability": water_weight,
    "energy_habitability": energy_weight,
    "windspeed_habitability": wind_weight,
}

scored_df, normalized_weights = apply_weights(scored_df, weights_input)

st.subheader("Normalized weights")
weight_cols = (
    pd.Series(normalized_weights, name="weight")
    .rename_axis("parameter")
    .reset_index()
    .sort_values("weight", ascending=False)
)
st.dataframe(weight_cols, use_container_width=True)

# Filter by minimum scores
filtered_df = scored_df.copy()
for column, threshold in filters.items():
    if threshold > 0:
        filtered_df = filtered_df[filtered_df[column] >= threshold]

if filtered_df.empty:
    st.warning("No rows satisfy the minimum score filters. Try relaxing the thresholds.")
    st.stop()


threshold_value = filtered_df["overall_habitability"].quantile(1 - top_percentage / 100)
top_candidates = filtered_df[filtered_df["overall_habitability"] >= threshold_value]

best_row = top_candidates.sort_values("overall_habitability", ascending=False).iloc[0]

st.subheader("Best location under current settings")
col1, col2, col3 = st.columns(3)
col1.metric("Latitude", f"{best_row['latitude']:.2f}°")
col2.metric("Longitude", f"{best_row['longitude']:.2f}°")
col3.metric("Habitability score", f"{best_row['overall_habitability']:.2f}/10")

st.markdown("### Map view")

if gpd is None:
    st.warning(
        "Install optional dependencies for map plots: `pip3 install geopandas shapely pyproj matplotlib`."
    )
else:
    try:
        map_gdf = to_geodataframe(filtered_df)
        highlight_gdf = map_gdf.loc[top_candidates.index]
        if best_row.name in map_gdf.index:
            best_gdf = map_gdf.loc[[best_row.name]]
        elif not highlight_gdf.empty:
            best_gdf = highlight_gdf.iloc[[0]]
        else:
            best_gdf = map_gdf.iloc[0:0]

        fig = plot_with_best(
            map_gdf,
            "overall_habitability",
            f"Habitability Scores — Top {top_percentage:.1f}% Highlighted",
            highlight_gdf,
            f"Top {top_percentage:.1f}%",
        )
        st.pyplot(fig)

        if best_gdf is not None and not best_gdf.empty:
            best_fig = plot_with_best(
                map_gdf,
                "overall_habitability",
                "Best Location Highlight",
                best_gdf,
                "Best Site",
            )
            st.pyplot(best_fig)
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Unable to render map: {exc}")

st.markdown("### Top candidates")

display_cols = [
    "latitude",
    "longitude",
    "overall_habitability",
    "temperature_habitability",
    "pressure_habitability",
    "water_access_habitability",
    "energy_habitability",
    "windspeed_habitability",
    "temperture",
    "atm pressure",
    "extvar_41",
    "extvar_32",
    "windspeed",
]

with st.expander("Preview top results"):
    st.dataframe(top_candidates[display_cols].reset_index(drop=True), use_container_width=True)

csv_buffer = top_candidates[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered results as CSV",
    data=csv_buffer,
    file_name="habitability_top_candidates.csv",
    mime="text/csv",
)

st.caption(
    "Scores are normalized per column after applying the original scoring formulas."
    " Adjust weights and thresholds to explore alternative mission profiles."
)

