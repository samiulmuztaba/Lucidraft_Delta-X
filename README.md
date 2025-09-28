

# Lucidraft
![./Lucidraft Banner.png](https://raw.githubusercontent.com/samiulmuztaba/Lucidraft_Delta-X/refs/heads/main/Lucidraft%20Banner.png)

**Lucidraft** is a command-line tool for logging, analyzing, and comparing the flight performance of paper aircraft models using your own flight videos and images. It is menu-driven and interactive, but you must provide your own files and follow the prompts.

---

## Requirements
- Python 3.8 or newer
- Install dependencies:
  ```bash
  pip install opencv-python colorama numpy matplotlib rich pandas seaborn
  ```
### P.S: These requirements are only IF you wanted to run the source code. However, if you download the zip file from Github Releases page (which i recommend if you just wanted to test this project)
---

## How to Use

1. **Open a terminal** (Bash, Zsh, or any modern terminal recommended for best color support).
2. **Navigate to the folder** containing `lucidraft.py`.
3. **Run the program:**
   ```bash
   python lucidraft.py
   ```
BUT, if you just wanted to test:
1. Go to the latest release
2. Download the zip file.
3. Unzip it, it should contain the exe file and the demo files you can use without any need to literally make a plane and record the flight video. The software also requires the photo of the plane for report generating purpose 😳. Yeah....i think that should be removed but i am kind of overwhelmed for a bit to work on this again 🤷‍♂️.
4. Type 'demo' for literally any input and you should be good to go
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
  - See a window pop up asking you to select the plane.
  - You'll see a crosshair type of thing but before you hit enter and it pops up another window you won't be able to selct the plane
  - Draw a box around the plane and it should start the video and track the plane and draw a line of it's trajectory
  - You'll be asked to enter the distance and airtime. For distance, you can enter the distance (😳) or you can type 'demo' if you were just testing this project. Same goes for the airtime but you'll be asked if you want to use the calculated airtime which uses mainly the video length so it's not that accurate so it's highly recommended to use your own calculated airtime if you are using it for what it was made for.
- All data and outputs are saved in `outputs/<model_name>/<version>/`.

**Demo Mode:** Type `demo` at any prompt to use built-in demo data for quick testing.

### 2. View Models
- Shows a table of all models and their versions with average distance, speed, and stability.

  <img src="https://cdn.gamma.app/1u916ozw98awqos/3e3476558faa4f65b369aa40c62f1b90/original/image.png"/>

### 3. Update Existing Model
- Enter the model name to update.
- Enter update notes (what you changed).
<img width="668" height="139" alt="Presentation_update0" src="https://github.com/user-attachments/assets/93e310b7-85fa-425f-b536-f0ce8f649dfc" />
<img width="1289" height="225" alt="Presentation_update1" src="https://github.com/user-attachments/assets/25078723-1995-4048-b8de-78e9206579ef" /> (updates will be saved in `metada.txt`. Though i just input 'Nothing has changed' because it was just for demo 💩.

- The tool will create a new version (increments by 0.1) and prompt you to log new flights as in option 1.
  <img width="607" height="142" alt="image" src="https://github.com/user-attachments/assets/440afc6c-cb14-4b11-beca-f29b7e4cddda" />

- After logging, it will compare the new version to the previous one.
<img width="557" height="245" alt="Presentation_update2" src="https://github.com/user-attachments/assets/8de7d45f-aad4-4a1f-9f17-761854354840" />


### 4. Delete Existing Model
- Enter the model name (and optionally version) to delete.

### 5. Compare Models
- Enter two model names and versions to compare side by side.
<img width="745" height="365" alt="Presentation_compare" src="https://github.com/user-attachments/assets/a545e015-f94b-43fc-a5a3-a4a33cabee13" />

### 6. Generate Overall Report
- Produces a summary table, CSV, chart, and Markdown report for all models in `outputs/`.

### 7. Exit
- Quits the program.

---

## Output Files
```bash
outputs/
├── ModelName/
    ├── 1.0/
    │   ├── metadata.txt
    │   ├── model_picture.jpg
    │   ├── Flight Videos/
    │   ├── Flight Coordinates/
    │   ├── Flight Metrics/
    │   ├── avg_metrics.csv
    │   ├── avg_coordinates.csv
    │   └── report.md
    └── 1.1/
        └── (same structure)
        
```
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

## PLEASE NOTE
- This project made to help those who want to create good performing airplanes where the main focus is distance and stability. Our team actually made a full framework for that and this project is meant to be built on top of that. If you are interested, you can take a look to the [research paper](https://github.com/user-attachments/files/22580547/RAJ_54.docx) and we do agree the research paper might not make any snese and we are planning to make a video of the presentation soon. In the research we theoretically proved that this framework is better than trial-and-error and is much more cost and time effective. However, we didn't validate that with real information so we planned to run the validation using real users where some will use trial-and-error methods or just how people would normally approach this if they wanted a good performing paper airplane and others will use our framework. If you are interested, please DM me at discord (username: samiul.muztaba). I might be offline because whether i will be outside or sleeping and except that i should be online.

- I consider that this CLI is really hard to navigate through but believe me i tried my best to make it user-friendly. I planned that a webiste and app version can be made for the sake of simplicity BUT i am kinda exhausted to work on this again so. . . that might take some good amount of time. Would really appreciate your patience <3
---

## License

MIT License. See LICENSE file.
