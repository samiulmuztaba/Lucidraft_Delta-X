

# Lucidraft


**Lucidraft** is a command-line tool for logging, analyzing, and comparing the flight performance of paper aircraft models using your own flight videos and images. It is menu-driven and interactive, but you must provide your own files and follow the prompts.

---

## Requirements

- Python 3.8 or newer
- Install dependencies:
  ```bash
  pip install opencv-python colorama numpy matplotlib rich pandas seaborn
  ```

---

## How to Use

1. **Open a terminal** (Bash, Zsh, or any modern terminal recommended for best color support).
2. **Navigate to the folder** containing `lucidraft.py`.
3. **Run the program:**
   ```bash
   python lucidraft.py
   ```

---

## Main Menu Options

You will see a menu like this:

```
 [1] ➕ Add New Model
 [2] 📂 View Models
 [3] ✨ Update Existing Model
 [4] 🗑 Delete Existing Model
 [5] 🔍 Compare Models
 [6] 📊 Generate Overall Report
 [7] 🚪 Exit
```

### 1. Add New Model
- Enter a model name (letters, numbers, dashes, underscores).
- Enter design notes (optional, but recommended).
- Enter the path to a `.jpg` image of your plane.
- Enter the path(s) to one or more `.mp4` flight videos (comma-separated).
- For each video, you will:
  - Select the plane in the first frame (ROI selection window will appear).
  - Enter the real-world flight distance in meters.
  - Accept or override the calculated airtime (in seconds).
  - Enter a stability score (0–10).
- All data and outputs are saved in `outputs/<model_name>/<version>/`.

**Demo Mode:** Type `demo` at any prompt to use built-in demo data for quick testing.

### 2. View Models
- Shows a table of all models and their versions with average distance, speed, and stability.

### 3. Update Existing Model
- Enter the model name to update.
- Enter update notes (what you changed).
- The tool will create a new version (increments by 0.1) and prompt you to log new flights as in option 1.
- After logging, it will compare the new version to the previous one.

### 4. Delete Existing Model
- Enter the model name (and optionally version) to delete.

### 5. Compare Models
- Enter two model names and versions to compare side by side.

### 6. Generate Overall Report
- Produces a summary table, CSV, chart, and Markdown report for all models in `outputs/`.

### 7. Exit
- Quits the program.

---

## Output Files

Each model/version directory contains:
- `model_picture.jpg` — your plane's image
- `Flight Videos/` — your flight videos
- `Flight Coordinates/` — CSVs of flight paths
- `Flight Metrics/` — CSVs of per-video metrics
- `avg_metrics.csv` — average metrics for the model/version
- `avg_coordinates.csv` — average trajectory
- `trajectory_graph.png` — flight path plot
- `report.md` — Markdown report

Overall reports are in `outputs/` as `overall_report.md`, `overall_report.csv`, and `overall_report.png`.

---

## Troubleshooting

- If you get errors, check that all file paths are correct and all dependencies are installed.
- Use a modern terminal for best color and emoji support.
- If you get stuck, read the error message and follow the prompt.

---

## License

MIT License. See LICENSE file.
