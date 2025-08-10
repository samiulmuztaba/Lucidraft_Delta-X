# Lucidraft Delta-X - The Paper Plane Model Analyzer & Version Control

## Project Overview

This project is a comprehensive CLI tool designed to support iterative engineering, analysis, and version control of paper plane designs. It enables users to log new models, analyze flight performance from video data, visualize flight trajectories, and compare design iterations—all in an organized, research-friendly workflow.

---

## Key Features

- **Model Logging:**  
  Log new paper plane models with name, version, and design notes. Each model and version is tracked separately for clear version control.

- **Flight Video Analysis:**  
  - Detects the paper plane in each frame of uploaded flight videos.
  - Extracts and saves trajectory coordinates to CSV files.
  - Calculates real-world flight metrics: distance, airtime, speed, and stability (with user input).
  - Computes and saves average trajectory points for each model/version.
  - Visualizes all flight trajectories (cyan lines) and the average trajectory (bright red/blue line) in a plot.

- **Version Control & Data Organization:**  
  - All outputs (metadata, stats, trajectories, visualizations) are saved and organized per model/version.
  - Enables easy retrieval and reporting for research or presentation.

- **Performance Comparison:**  
  - After analysis, compares the current version’s metrics to previous versions.
  - Clearly indicates if the new design has improved, stayed the same, or regressed, with concise CLI output.

- **Reporting:**  
  - Generates comprehensive reports summarizing all flights, metrics, averages, and trends (“tendencies”) for each model/version.
  - Supports multiple flight videos per model/version, with statistical analysis and visualization.

---

## Workflow

1. **Log a New Model:**  
   Start by logging a new paper plane model with its name, version, and design notes.

2. **Upload Flight Videos:**  
   Upload or select flight videos for the model/version to be analyzed.

3. **Analyze & Visualize:**  
   The tool processes each video, detects the plane, extracts trajectory data, calculates metrics, and generates visualizations.

4. **Compare Versions:**  
   Automatically compares the current version’s performance to previous iterations, highlighting improvements or regressions.

5. **Generate Reports:**  
   Produces organized reports for each design iteration, supporting research, documentation, and presentation.

6. **Iterate:**  
   Repeat the process for each new design, enabling continuous improvement and data-driven engineering.

---

## Ideal Use Cases

- Iterative engineering and optimization of paper plane designs
- Educational research and classroom experiments
- Hobbyist competitions and performance tracking
- Scientific reporting and presentation of flight data

---

## Output Organization

- **Metadata:** JSON files per model/version
- **Trajectories:** CSV files for each flight and averaged data
- **Stats:** CSV files with all metrics and averages
- **Visualizations:** PNG plots of trajectories
- **Reports:** Markdown or PDF summaries with tables, charts, and trend analysis

---

## Professional Notes

This tool is designed for accessibility and extensibility. It supports non-technical users through clear CLI commands and organized outputs, while providing robust data management and analysis for advanced users and researchers. The modular structure allows for future integration with machine learning models, advanced video analysis, or web-based interfaces.

