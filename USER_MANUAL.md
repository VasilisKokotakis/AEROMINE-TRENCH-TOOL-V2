# Aeromine Trench Tool - User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Understanding the Parameters](#understanding-the-parameters)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Understanding the Outputs](#understanding-the-outputs)
6. [Tips and Best Practices](#tips-and-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

The Aeromine Trench Tool is designed to analyze trench point cloud data from LAS/LAZ files. It automatically extracts cross-sections along the trench, calculates wall distances and depths, and generates comprehensive analysis reports.

**What it does:**
- Processes 3D point cloud data from trenches
- Divides the trench into cross-sections at regular intervals
- Measures wall-to-wall distance and depth for each section
- Checks if sections meet minimum depth requirements
- Creates visual plots and detailed reports
- Packages everything in an easy-to-use ZIP file

---

## Getting Started

### Launching the Application

1. Open a terminal in the application folder
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Run the application:
   ```bash
   python main_app.py
   ```

The graphical interface will open with three main sections:
- **File Selection**: Choose your input LAS/LAZ file
- **Output Directory**: Choose where to save results
- **Parameters**: Configure processing settings

---

## Understanding the Parameters

### 1. Spacing (m)
**Default: 0.10 meters**

**What it does:** Defines the distance between consecutive cross-sections along the trench.

**Example:**
- If you set spacing to 1.0 meter, the tool will create cross-sections at 0m, 1m, 2m, 3m, etc. along the trench
- Smaller values (e.g., 0.1m) = More detailed analysis, more cross-sections
- Larger values (e.g., 2.0m) = Faster processing, fewer cross-sections

**When to adjust:**
- Use **smaller spacing** (0.05-0.2m) for detailed analysis or irregular trenches
- Use **larger spacing** (0.5-2.0m) for long, uniform trenches or quick overviews

**Visual representation:**
```
Trench:  |-------|-------|-------|-------|
Spacing:    1m      1m      1m      1m
Sections:   0       1       2       3       4
```

---

### 2. Prefilter Half-Width (m)
**Default: 2.0 meters**

**What it does:** Initial filtering step that removes points too far from the estimated trench centerline.

**Example:**
- With prefilter half-width of 2.0m, only points within ±2.0m of the trench axis are kept for processing
- This eliminates ground surface points, vegetation, and other objects far from the trench

**When to adjust:**
- **Increase** (3.0-5.0m) if you have a very wide trench or want to include surrounding terrain
- **Decrease** (1.0-1.5m) if you have a narrow trench or lots of nearby objects to exclude

**Visual representation:**
```
Top View:
                [2m prefilter zone]
    × × × ×  |==================|  × × × ×
    × × × ×  |  Trench Points   |  × × × ×
    × × × ×  |==================|  × × × ×
              ↑                  ↑
           -2m axis           +2m
    
    × = Points excluded by prefilter
```

---

### 3. Edge-Lock (m)
**Default: 0.05 meters**

**What it does:** Stabilizes edge detection by preventing small fluctuations in detected edges between adjacent sections.

**Example:**
- If section 10 has an edge at 0.8m and section 11 tries to detect an edge at 0.75m, the edge-lock margin keeps it close to 0.8m instead
- This prevents "jittery" edges in the analysis

**When to adjust:**
- **Increase** (0.1-0.2m) if sections are very similar and you want smooth, consistent edges
- **Decrease** (0.01-0.03m) if the trench width varies significantly and you need to capture real changes

**Recommendation:** Usually leave at default unless you see unrealistic edge jumps in results.

---

### 4. Half-Width (m)
**Default: 0.7 meters**

**What it does:** Defines the final clipping boundary for trench points in "fixed" mode, or serves as a fallback in "auto" mode.

**Example:**
- In **fixed mode**: Only keeps points within ±0.7m of the trench axis
- In **auto mode**: Uses this value when automatic edge detection fails for a section

**When to adjust:**
- Measure or estimate your actual trench width
- Set half-width to approximately half of your trench's wall-to-wall distance
- Example: For a 1.4m wide trench, use half-width = 0.7m

**Visual representation:**
```
Cross-section View:
    
    ↑                           ↑
    |                           |
    |     ╱                 ╲   |
    |    ╱                   ╲  |
    |   ╱                     ╲ |
    |  ╱                       ╲|
    |---|--------------------|---|
   -0.7m  Trench Floor      +0.7m
    
    Only points between -0.7m and +0.7m are analyzed
```

---

### 5. Right Trim (m)
**Default: 0.0 meters**

**What it does:** Applies asymmetric clipping by reducing the width on the right side of the trench.

**Example:**
- If half-width = 0.7m and right-trim = 0.1m:
  - Left side: -0.7m to 0m ✓ (full width)
  - Right side: 0m to +0.6m ✓ (reduced by 0.1m)

**When to use:**
- Your trench has one side that's damaged or has debris
- One wall is not vertical or has collapsed
- You want to exclude one side from the analysis

**Common scenario:** 
After excavation, one side of the trench collapsed slightly. Set right-trim to 0.15m to ignore that problematic wall in the measurements.

---

### 6. Slope Threshold
**Default: 1.5**

**What it does:** Used in "auto" clip mode to detect where the trench walls begin based on slope changes.

**How it works:**
- The tool looks at the elevation profile across the trench
- Where the slope (steepness) exceeds the threshold, it identifies a wall edge
- Higher threshold = more conservative, only detects very steep walls
- Lower threshold = more sensitive, detects gentler slopes as walls

**When to adjust:**
- **Increase** (2.0-3.0) for trenches with very steep, clearly defined walls
- **Decrease** (0.8-1.2) for trenches with gradual wall slopes or irregular shapes

**Only applies when Clip Mode = "auto"**

---

### 7. Depth Min (m)
**Default: 0.0 meters**

**What it does:** Sets a minimum acceptable depth criterion. Sections are marked as PASS or FAIL based on whether they meet this requirement.

**Example:**
- Set depth-min = 1.5m
- Section 5 has depth = 1.8m → **PASS** ✓
- Section 12 has depth = 1.2m → **FAIL** ✗

**When to use:**
- You have specifications or requirements for minimum trench depth
- You need to verify compliance with excavation standards
- Quality control for construction projects

**Output:**
- Summary CSV includes "depth_status" column (PASS/FAIL/N/A)
- Analysis report shows total PASS and FAIL counts
- Set to 0.0 to disable this check (all sections show "N/A")

---

### 8. Auto-Axis (Checkbox)
**Default: Enabled ✓**

**What it does:** Automatically detects the trench orientation using Principal Component Analysis (PCA).

**How it works:**
- Analyzes the overall point cloud distribution
- Finds the main direction the trench follows
- Works well for linear trenches

**When to disable:**
- Your trench has curves or multiple directions
- Manual axis definition is preferred
- Testing different orientations

**Recommendation:** Keep enabled for most trenches. The automatic detection is usually very accurate.

---

### 9. Clip Mode (Dropdown)
**Options: "fixed" or "auto"**
**Default: fixed**

**Fixed Mode:**
- Uses the Half-Width parameter to clip points
- Simple, predictable, consistent across all sections
- Best for uniform trenches with regular geometry

**Auto Mode:**
- Automatically detects wall edges for each section using slope analysis
- Adapts to varying trench widths
- Uses Half-Width as a fallback if detection fails
- Best for irregular trenches or varying widths

**Choosing between them:**
- Start with **"fixed"** for most projects
- Use **"auto"** if your trench width varies significantly or has irregular walls

---

## Step-by-Step Workflow

### Step 1: Select Input File
1. Click **"Browse..."** next to "1. Select LAS/LAZ File"
2. Navigate to your point cloud file
3. Select the .las or .laz file
4. The file path will appear in the field

### Step 2: Select Output Directory
1. Click **"Browse..."** next to "2. Select Output Directory"
2. Choose a folder where results will be saved
3. A subfolder will be created automatically for this analysis

### Step 3: Configure Parameters
1. Review the default parameters
2. Adjust based on your trench characteristics (see parameter descriptions above)
3. Common starting points:
   - **Small trench** (< 1m wide): half-width = 0.4-0.5m, spacing = 0.1m
   - **Medium trench** (1-2m wide): half-width = 0.7-1.0m, spacing = 0.2m
   - **Large trench** (> 2m wide): half-width = 1.0-1.5m, spacing = 0.5m

### Step 4: Run Processing
1. Click the **"Run"** button
2. Watch the log window for progress messages:
   - File loading
   - Axis detection
   - Section computation
   - Clipping operations
   - CSV file generation
3. Wait for "Processing complete" message
4. **Two CSV files are now created** in your output folder

### Step 5: Generate Analysis (Optional)
1. Click **"Analyze & Create ZIP"** button
2. Wait for analysis to complete (may take 1-5 minutes depending on section count)
3. A message box will confirm when the ZIP is ready
4. The ZIP file contains all plots and a detailed report

### Step 6: View Results
1. Click **"Open Output Folder"** to browse results
2. Review the CSV files in Excel or other spreadsheet software
3. Extract the analysis ZIP file to view plots and reports

---

## Understanding the Outputs

### sections.csv
**What it contains:** Every individual point from the point cloud with its section assignment.

**Columns:**
- `x`, `y`, `z`: Original 3D coordinates of each point
- `section_id`: Which cross-section this point belongs to (0, 1, 2, ...)
- `dist_along`: Distance along the trench axis
- `dist_off`: Distance perpendicular to the axis (positive = right, negative = left)

**Size:** Can be very large (millions of rows for dense point clouds)

**Use case:** 
- Detailed analysis
- Importing into other 3D software
- Creating custom visualizations

---

### sections_summary.csv
**What it contains:** Summary statistics for each cross-section.

**Columns:**
- `section_id`: Section number
- `count`: Number of points in this section
- `z_min`, `z_max`: Minimum and maximum elevation
- `x_min`, `x_max`: Minimum and maximum perpendicular distance
- `wall_distance`: Measured wall-to-wall distance (x_max - x_min)
- `depth`: Measured depth (z_max - z_min)
- `spacing`: The spacing parameter used
- `depth_min`: The minimum depth criterion used
- `depth_status`: PASS/FAIL/N/A based on depth criterion

**Size:** Small (one row per section)

**Use case:**
- Quick overview of trench characteristics
- Quality control checks
- Identifying problem sections
- Statistical analysis

---

### complete_analysis.zip
**What it contains:** Comprehensive analysis package with plots and reports.

**Contents:**

1. **Cross-section plots** (cross_section_XX.png)
   - One plot per section showing point distribution
   - X-axis: Distance from trench axis
   - Y-axis: Elevation
   - Title includes section length (e.g., "Cross-section 5 (Length: 5.00 m)")

2. **trench_profile_analysis.png**
   - Two graphs showing trends along the trench:
     - Top: Wall distance profile (how width varies)
     - Bottom: Depth profile (how depth varies)
   - Includes mean reference lines

3. **combined_profile_analysis.png**
   - Dual-axis plot showing both wall distance and depth on one graph
   - Useful for seeing correlations between width and depth

4. **statistics_summary.png**
   - Four-panel statistical overview:
     - Wall distance histogram (frequency distribution)
     - Depth histogram (frequency distribution)
     - Wall distance vs Depth scatter plot (correlation)
     - Points per section bar chart (data quality check)

5. **complete_analysis_report.txt**
   - Detailed text report with:
     - Summary statistics (total sections, length, spacing)
     - Depth criterion and PASS/FAIL counts
     - Wall distance statistics (mean, min, max, standard deviation)
     - Depth statistics (mean, min, max, standard deviation)
     - Coefficient of variation (measure of variability)
     - Complete section-by-section details table

---

## Tips and Best Practices

### Before Processing

1. **Inspect your LAS file:**
   - Open it in CloudCompare or similar software
   - Verify it contains the trench area
   - Check for excessive noise or outliers

2. **Estimate trench dimensions:**
   - Measure or estimate typical width
   - Measure or estimate typical depth
   - This helps set appropriate parameters

3. **Start with defaults:**
   - Run once with default parameters
   - Review results
   - Adjust parameters if needed and re-run

### During Processing

1. **Watch the log window:**
   - Check point count is reasonable (should see thousands to millions)
   - Verify section count makes sense for your trench length
   - Look for warning messages

2. **Processing time:**
   - Small files (< 1 million points): seconds
   - Medium files (1-10 million points): 10-60 seconds
   - Large files (> 10 million points): 1-5 minutes
   - Analysis ZIP generation: adds 1-5 minutes

### After Processing

1. **Check sections_summary.csv first:**
   - Look for sections with very low point counts (< 1000) - may be unreliable
   - Check if depth values are reasonable
   - Look for sudden jumps in wall_distance (may indicate problems)

2. **Review the plots:**
   - Do the cross-sections look like actual trench profiles?
   - Are the wall distance and depth trends smooth or erratic?
   - Do any sections look suspicious?

3. **Quality indicators:**
   - **Good:** Smooth trends, consistent measurements, 1000+ points per section
   - **Suspicious:** Sudden jumps, very low point counts, extreme outliers
   - **Bad:** Empty sections, negative depths, unrealistic dimensions

### Parameter Adjustment Strategy

If results aren't satisfactory, try adjusting in this order:

1. **First:** Check Clip Mode
   - Try switching between "fixed" and "auto"

2. **Second:** Adjust Half-Width
   - If measurements are too small: increase half-width
   - If including too much noise: decrease half-width

3. **Third:** Adjust Spacing
   - If trends are too coarse: decrease spacing
   - If too much detail/noise: increase spacing

4. **Fourth:** Fine-tune advanced parameters
   - Prefilter, edge-lock, slope threshold, right-trim

---

## Troubleshooting

### Problem: "No points loaded" or very small point count
**Solution:**
- Verify the LAS file is not corrupted
- Check if the file contains coordinate data
- Try opening the file in CloudCompare to verify

### Problem: All sections have FAIL depth status
**Solution:**
- Your depth_min criterion may be too strict
- Check actual depths in sections_summary.csv
- Either adjust depth_min or investigate why trench is shallow

### Problem: Wall distances seem too large or too small
**Solution:**
- Adjust half-width parameter to match actual trench width
- Try switching between "fixed" and "auto" clip mode
- Check if Auto-Axis detection worked correctly (view cross-section plots)

### Problem: Empty or missing sections
**Solution:**
- Point cloud may have gaps
- Increase prefilter_half_width to capture more points
- Decrease spacing to create fewer, better-populated sections

### Problem: Cross-section plots show diagonal lines instead of trench profile
**Solution:**
- Auto-Axis detection may have found wrong orientation
- Try disabling Auto-Axis checkbox
- Or rotate your point cloud in CloudCompare before processing

### Problem: Analysis ZIP generation fails
**Solution:**
- Ensure you clicked "Run" first and it completed successfully
- Check that sections.csv and sections_summary.csv exist in output folder
- Verify you have write permissions to the output directory

### Problem: Very slow processing
**Solution:**
- Point cloud may be extremely large (> 50 million points)
- Increase spacing to reduce section count
- Increase prefilter_half_width to filter out more distant points
- Consider decimating the point cloud in CloudCompare first

---

## Example Scenarios

### Scenario 1: Standard Utility Trench
**Characteristics:** 1.2m wide, 1.5m deep, 50m long, uniform shape

**Recommended parameters:**
- Spacing: 0.2m (250 sections)
- Half-width: 0.6m
- Prefilter: 2.0m
- Clip mode: fixed
- Depth min: 1.4m (to ensure compliance)

### Scenario 2: Irregular Archaeological Trench
**Characteristics:** Varies 0.8-1.5m wide, 0.5-2.0m deep, curved

**Recommended parameters:**
- Spacing: 0.1m (more detail for irregular shape)
- Half-width: 1.0m (generous to capture variations)
- Clip mode: auto (adapts to width changes)
- Slope threshold: 1.2 (gentler detection for varied walls)
- Depth min: 0.0 (no requirement)

### Scenario 3: Narrow Cable Trench
**Characteristics:** 0.4m wide, 0.8m deep, 200m long, very uniform

**Recommended parameters:**
- Spacing: 0.5m (less detail needed for uniform trench)
- Half-width: 0.25m
- Prefilter: 1.0m (narrow focus)
- Clip mode: fixed
- Depth min: 0.75m

---

## Contact and Support

For questions, issues, or feature requests:
- Check this manual first
- Review the log output for error messages
- Consult the main README.md file
- Check the GitHub repository for updates

**Remember:** Most processing issues can be resolved by adjusting parameters. Don't hesitate to experiment with different settings!

---

*Last updated: January 2026*
