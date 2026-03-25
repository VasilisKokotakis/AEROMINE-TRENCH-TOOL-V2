# AEROMINE-TRENCH-TOOL-V2

A comprehensive application for processing and analyzing trench point cloud data from LAS/LAZ files. Extract cross-sections, calculate wall distances and depths, and generate detailed statistical analysis reports. Available as both a desktop GUI application and a web-based interface.

## Features

- **Desktop GUI Application**: Modern Tkinter interface with easy file selection and parameter configuration
- **Web Interface**: Alternative browser-based interface for remote processing
- **Automatic Axis Detection**: Uses PCA to automatically detect trench orientation
- **Cross-Section Generation**: Creates individual cross-section plots with length labels for each section
- **Statistical Analysis**: Calculates wall distances, depths, total trench length, and variability metrics
- **Depth Criterion**: Configurable minimum depth requirement with PASS/FAIL status tracking
- **Flexible Processing**: Configurable parameters for different trench types
- **Batch Analysis**: Process multiple sections with comprehensive reporting
- **Complete Analysis ZIP**: One-click generation of plots, reports, and statistics packaged in a ZIP file

## Requirements

- Python 3.8+
- LAS/LAZ file support via `laspy` library
- Web browser for interface access

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VasilisKokotakis/AEROMINE-TRENCHE-TOOL-V2.git
   cd AEROMINE-TRENCHE-TOOL-V2
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Desktop GUI Application (Recommended)

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Run the application:**
   ```bash
   python main_app.py
   ```

3. **Use the interface:**
   - Click "Browse..." to select your LAS/LAZ file
   - Select an output directory for results
   - Configure processing parameters:
     - **Spacing**: Distance between cross-sections (meters)
     - **Prefilter half-width**: Initial filtering width (meters)
     - **Edge-lock**: Margin for edge locking (meters)
     - **Half-width**: Trench half-width for clipping (meters)
     - **Right trim**: Asymmetric clipping from right side (meters)
     - **Slope threshold**: Edge detection sensitivity (for auto mode)
     - **Depth min**: Minimum acceptable depth criterion (meters)
     - **Auto-axis**: Enable PCA-based axis detection
     - **Clip mode**: Choose "fixed" or "auto" edge detection
   
4. **Process and analyze:**
   - Click "Run" to generate CSV files
   - Click "Analyze & Create ZIP" to generate complete analysis package
   - Click "Open Output Folder" to view results

### Web Application (Alternative)

1. **Start the server:**
   ```bash
   python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open your browser:**
   - Navigate to `http://localhost:8000`
   - Upload your LAS/LAZ file
   - Configure processing parameters
   - Click "Run" to process
   - Click "🔍 Analyze and Download Results" to generate analysis ZIP

3. **Download results:**
   - `sections.csv`: Raw point cloud data with section assignments
   - `sections_summary.csv`: Statistical summary with depth status for each section
   - `complete_analysis.zip`: Comprehensive analysis package (plots + report)

## Processing Parameters

| Parameter | Description | Default | Unit |
|-----------|-------------|---------|------|
| `spacing` | Distance between cross-sections | 0.1 | meters |
| `prefilter_half_width` | Initial trench width filter | 2.0 | meters |
| `edge_lock_margin` | Edge locking margin | 0.05 | meters |
| `half_width` | Final clipping width | 0.7 | meters |
| `right_trim` | Asymmetric right trimming | 0.0 | meters |
| `slope_threshold` | Edge detection sensitivity (auto mode) | 1.5 | - |
| `depth_min` | Minimum acceptable depth criterion | 0.0 | meters |
| `clip_mode` | Edge detection mode (fixed/auto) | fixed | - |
| `autoaxis` | Automatic axis detection via PCA | true | - |

## Project Structure

```
AEROMINE-TRENCHE-TOOL-V2/
├── main_app.py              # Desktop GUI application (Tkinter)
├── app.py                   # FastAPI web application
├── index.html               # Web interface
├── sections/                # Processing modules
│   ├── __init__.py
│   ├── processing.py        # Core sectioning logic
│   ├── io.py                # LAS file I/O
│   └── visualization.py     # Plotting functions
├── requirements.txt         # Python dependencies
├── runs/                    # Processed files and outputs
└── .venv/                   # Virtual environment
```

## Output Files

### CSV Files
- **sections.csv**: Complete point cloud data with section IDs and distances
- **sections_summary.csv**: Per-section statistics including:
  - Section ID, point count
  - Elevation range (z_min, z_max)
  - Wall distance (x_min, x_max)
  - Calculated depth and wall distance
  - Depth status (PASS/FAIL based on criterion)
  - Spacing and depth_min values

### Analysis ZIP Package
Generated by clicking "Analyze & Create ZIP" button, contains:
- **Cross-section plots**: Individual PNG for each section showing:
  - Distance from trench axis vs elevation
  - Section length label (e.g., "Cross-section 5 (Length: 5.00 m)")
- **Trench profile analysis**: Wall distance and depth profiles along length
- **Combined profile analysis**: Dual-axis view of wall distance and depth
- **Statistics summary**: Histograms and correlation plots
- **complete_analysis_report.txt**: Detailed text report with:
  - Total sections and length
  - Spacing and depth criterion
  - Wall distance and depth statistics (mean, min, max, std)
  - PASS/FAIL counts
  - Per-section details table

## Processing Pipeline

1. **File Selection**: LAS/LAZ file selected via GUI or uploaded via web interface
2. **Point Loading**: Points extracted with coordinates (X, Y, Z)
3. **Axis Detection**: PCA-based automatic trench orientation detection (if enabled)
4. **Sectioning**: Points divided into cross-sections based on spacing parameter
5. **Filtering**: Points filtered by distance from trench axis (prefilter stage)
6. **Edge Detection**: Automatic (slope-based) or fixed-width edge detection
7. **Statistics**: Wall distance, depth, and total length calculations per section
8. **Depth Criterion**: PASS/FAIL status assigned based on minimum depth requirement
9. **Output Generation**: CSV files generated with section data and summary
10. **Analysis (Optional)**: Complete analysis package with plots and detailed report

## Analysis Metrics

- **Wall Distance**: Maximum lateral extent of trench walls per section
- **Depth**: Vertical extent from highest to lowest point per section
- **Total Trench Length**: Calculated from section range and spacing
- **Depth Status**: PASS/FAIL based on configurable minimum depth criterion
- **Coefficient of Variation**: Relative variability measures for wall distance and depth
- **Section Count**: Number of processed cross-sections with sufficient point density
- **Point Density**: Points per section statistics

## Tips

- Use the **Desktop GUI** for easier file management and local processing
- Set **depth_min** to enforce minimum depth requirements and track compliance
- **Auto-axis** mode (PCA) works best for linear trenches
- **Clip mode: auto** uses slope detection to find trench edges automatically
- **Clip mode: fixed** uses the half-width parameter for consistent clipping
- Adjust **spacing** based on your required cross-section density
- Use **right_trim** for asymmetric trenches with different wall geometries

## Troubleshooting

- **"No module named 'numpy'"**: Activate the virtual environment first
  ```bash
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```
- **Empty CSV files**: Check that LAS file has sufficient point density
- **Analysis fails**: Ensure "Run" has completed successfully before clicking "Analyze"
- **GUI not starting**: Make sure tkinter is installed (usually included with Python)

## Support

For questions or issues:
- Check the processing parameters match your trench characteristics
- Verify LAS file format compatibility (tested with LAS 1.2-1.4)
- Ensure sufficient point density for reliable analysis
- Review the log output for detailed processing information

---

**Built with:** Python, Tkinter, FastAPI, pandas, matplotlib, scikit-learn, laspy
