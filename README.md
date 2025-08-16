# Lucid Raft Delta-X

![Lucid Raft Delta-X Banner](https://via.placeholder.com/800x200.png?text=Lucid+Raft+Delta-X)

**Lucid Raft Delta-X** is an advanced command-line tool for paper plane enthusiasts and engineers to log, analyze, and compare flight performance of paper aircraft models. It processes flight videos to extract metrics like distance, airtime, speed, and stability, generating detailed reports and visualizations for research and optimization. Built with Python, it leverages OpenCV for video analysis, Matplotlib for plotting, and Pandas for data handling, all wrapped in a colorful CLI interface powered by Colorama.

## Features

- **Add New Models**: Log new paper plane designs with images, flight videos, and design notes.
- **Update Models**: Create new versions of existing models and compare performance.
- **View Models**: Display a tabulated overview of all models and their versions with key metrics.
- **Compare Models**: Compare two models or versions based on distance, speed, and stability.
- **Generate Reports**: Produce detailed Markdown reports and visualizations (e.g., trajectory graphs, bar charts) for individual models and overall fleet performance.
- **Video Analysis**: Process flight videos to track plane trajectories, calculate metrics, and visualize paths with custom overlays.
- **Social Sharing**: Share findings on X with the hashtag `#LucidraftDeltaX`.

## Prerequisites

- **Python 3.8+**
- **Dependencies**:
  - `opencv-python` (for video processing)
  - `colorama` (for colorful CLI output)
  - `numpy` (for numerical operations)
  - `matplotlib` (for plotting trajectories and charts)
  - `pandas` (for data handling)
  - `tabulate` (for ASCII table generation)

## Installation

1. **Clone the Repository** (if hosted on a git platform):
   ```bash
   git clone https://github.com/yourusername/lucidraft-delta-x.git
   cd lucidraft-delta-x
   ```

2. **Install Dependencies**:
   Create a virtual environment (optional but recommended) and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install opencv-python colorama numpy matplotlib pandas tabulate
   ```

3. **Prepare Input Files**:
   - Ensure you have flight videos (e.g., `.mp4`) and model images (e.g., `.jpg`) ready for analysis.
   - Videos should capture the paper plane in flight with a clear, contrasting background for accurate tracking.

## Usage

1. **Run the Program**:
   ```bash
   python lucidraft_delta_x.py
   ```

2. **Navigate the CLI**:
   The program presents a menu with six options:
   - **1. Add New Model**: Enter a model name, image path, flight video paths (comma-separated), design notes, and a stability score (1–10).
   - **2. Update Existing Model**: Update an existing model with a new version, incrementing the version number by 0.1.
   - **3. View Models**: Display a table of all models, versions, and their average metrics (distance, speed, stability).
   - **4. Compare Models**: Compare two models/versions by distance, speed, and stability.
   - **5. Generate Overall Report**: Create a comprehensive report summarizing all models, including a CSV, bar chart, and Markdown file.
   - **6. Exit**: Close the program.

3. **Output Structure**:
   - Outputs are saved in the `outputs/<model_name>/<version>/` directory, containing:
     - `model_picture.jpg`: Model image.
     - `Flight Videos/`: Copied flight videos.
     - `Flight Coordinates/`: CSV files with trajectory coordinates.
     - `Flight Metrics/`: CSV files with per-video metrics (distance, airtime, speed, stability).
     - `avg_metrics.csv`: Average metrics for the model/version.
     - `avg_coordinates.csv`: Average trajectory coordinates.
     - `trajectory_graph.png`: Plot of flight trajectories.
     - `report.md`: Detailed Markdown report.
   - Overall reports are saved in `outputs/` as `overall_report.md`, `overall_report.csv`, and `overall_report.png`.

4. **Example**:
   - Add a new model:
     ```
     Select an option (1-6): 1
     ✈ Model Name: Falcon
     🖼 Model's Picture: path/to/falcon.jpg
     🎥 Flight Videos (comma separated): path/to/video1.mp4, path/to/video2.mp4
     📝 Design Notes: Optimized wing shape for longer glide
     📝 Enter stability score for Falcon (1-10): 8
     ```
   - Outputs will be saved in `outputs/Falcon/1.0/`.

## Video Processing Notes

- **Trajectory Tracking**: The script uses OpenCV to detect the paper plane in videos based on HSV color range (`[35, 50, 102]` to `[179, 255, 255]`) and contour area (>500 pixels). Adjust these values in `log_new_model()` if your plane’s color or video conditions differ.
- **Metrics Calculation**: Distance (pixels), airtime (seconds), speed (pixels/second), and user-provided stability scores are computed and saved.
- **Visualization**: Trajectories are plotted with Matplotlib, showing individual flights (cyan) and average trajectory (red).

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and test thoroughly.
4. Commit your changes (`git commit -m "Add your feature"`).
5. Push to your branch (`git push origin feature/your-feature`).
6. Open a pull request with a detailed description.

Please ensure your code follows the existing style (e.g., consistent use of Colorama for CLI output) and includes error handling.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python and libraries: OpenCV, Colorama, NumPy, Matplotlib, Pandas, and Tabulate.
- Inspired by the joy of paper plane engineering and the pursuit of flight perfection.
- Share your paper plane designs and results on X with `#LucidraftDeltaX`!

## Contact

For questions, suggestions, or to share your awesome paper planes, reach out on X with `#LucidraftDeltaX` or open an issue on the repository.

✈️ Fly high, engineers!