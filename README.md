# AEROMINE-TRENCHE-TOOL-V2

A comprehensive web-based application for processing and analyzing trench point cloud data from LAS/LAZ files. Extract cross-sections, calculate wall distances and depths, and generate detailed statistical analysis reports.

##  Features

- **Web Interface**: Upload LAS files through a clean web interface
- **Automatic Axis Detection**: Uses PCA to automatically detect trench orientation
- **Cross-Section Generation**: Creates individual cross-section plots for each section
- **Statistical Analysis**: Calculates wall distances, depths, and variability metrics
- **Flexible Processing**: Configurable parameters for different trench types
- **Batch Analysis**: Process multiple sections with comprehensive reporting
- **Negative Section IDs**: Supports sections before the axis start point

## 📋 Requirements

- Python 3.8+
- LAS/LAZ file support via `laspy` library
- Web browser for interface access

## 🛠️ Installation

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

##  Usage

### Web Application

1. **Start the server:**
   ```bash
   python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open your browser:**
   - Navigate to `http://localhost:8000`
   - Upload your LAS/LAZ file
   - Configure processing parameters:
     - **Spacing**: Distance between cross-sections (meters)
     - **Half-width**: Trench half-width for filtering (meters)
     - **Right trim**: Asymmetric clipping from right side (meters)
     - **Slope threshold**: Edge detection sensitivity
     - **Edge lock margin**: Margin for edge locking (meters)

3. **Download results:**
   - `sections.csv`: Raw point cloud data with section assignments
   - `sections_summary.csv`: Statistical summary for each section

### Analysis Scripts

#### Complete Analysis (`complete_analysis.py`)
Generates comprehensive analysis including plots and statistics:

```bash
python complete_analysis.py
```

**Outputs:**
- Individual cross-section plots (section_XXXXX.jpg)
- Trench profile analysis (wall distance & depth profiles)
- Combined profile analysis (dual-axis view)
- Statistics summary (histograms & correlations)
- Complete analysis report (detailed text report)

#### Summary Analysis (`analyze_summary.py`)
Analyzes existing summary CSV files:

```bash
python analyze_summary.py
# Enter path to sections_summary.csv when prompted
```

##  Processing Parameters

| Parameter | Description | Default | Unit |
|-----------|-------------|---------|------|
| `spacing` | Distance between cross-sections | 0.1 | meters |
| `prefilter_half_width` | Trench width filter | 2.0 | meters |
| `half_width` | Final clipping width | 1.0 | meters |
| `right_trim` | Asymmetric right trimming | 0.0 | meters |
| `slope_threshold` | Edge detection sensitivity | 0.1 | - |
| `edge_lock_margin` | Edge locking margin | 0.05 | meters |

##  Project Structure

```
AEROMINE-TRENCHE-TOOL-V2/
├── app.py                    # FastAPI web application
├── index.html               # Web interface
├── complete_analysis.py     # Complete analysis workflow
├── analyze_summary.py       # Summary statistics analysis
├── plot_csv.py             # CSV plotting utilities
├── sections/               # Processing modules
│   ├── __init__.py
│   ├── processing.py       # Core sectioning logic
│   ├── io.py              # LAS file I/O
│   └── visualization.py   # Plotting functions
├── requirements.txt        # Python dependencies
├── results/               # Generated results (not in repo)
├── runs/                  # Processed files (not in repo)
└── Images/               # Generated plots (not in repo)
```

## 🔧 Processing Pipeline

1. **File Upload**: LAS/LAZ file uploaded via web interface
2. **Point Loading**: Points extracted with coordinates (X, Y, Z)
3. **Axis Detection**: PCA-based automatic trench orientation detection
4. **Sectioning**: Points divided into cross-sections based on spacing
5. **Filtering**: Points filtered by distance from trench axis
6. **Edge Detection**: Automatic or fixed-width edge detection
7. **Statistics**: Wall distance and depth calculations per section
8. **Output**: CSV files and analysis reports generated

##  Analysis Metrics

- **Wall Distance**: Maximum lateral extent of trench walls
- **Depth**: Vertical extent from highest to lowest point
- **Coefficient of Variation**: Relative variability measures
- **Section Count**: Number of processed cross-sections
- **Point Density**: Points per section statistics

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Check the processing parameters
- Verify LAS file format compatibility
- Ensure sufficient point density for analysis

---

**Built with:** FastAPI, pandas, matplotlib, scikit-learn, laspy
