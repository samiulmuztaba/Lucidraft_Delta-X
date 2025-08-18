import os
import cv2 as cv
from colorama import init, Fore, Style
import shutil
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from datetime import datetime
import re

init(autoreset=True) # after each prompt, go back to homepage
init(convert=True, strip=False)

# Colors
C = Fore.CYAN + Style.BRIGHT
B = Fore.BLUE + Style.BRIGHT
Y = Fore.YELLOW + Style.BRIGHT
W = Fore.WHITE + Style.BRIGHT
G = Fore.GREEN + Style.BRIGHT
R = Fore.RED + Style.BRIGHT
M = Fore.MAGENTA + Style.BRIGHT

# ==== Main Homepage ====
def banner():
    logo = f"""
{C}   ╔══════════════════════════════════════════════╗
{C}   ║{B}   L U C I D R A F T   D E L T A - X   1 . 0  {C}║
{C}   ║{M} Advanced Paper Plane Engineering Terminal    {C}║
{C}   ╚══════════════════════════════════════════════╝
"""
    print(logo)

def pause():
    input(W + "\nPress any key to return to the homepage...")

breakw = False
rerun_req = 'please try to run the program again!'

# Setup
# Input for real world metrics
ref_obj_m = None
ref_obj_px = None
if not ref_obj_px and not ref_obj_m:
    print(f'{B}\nBefore we begin, I need to know the px per meter for the video. For this, Measure an object in the video and then measer its length in pixels so that we can give you distance\nin meters rather than in pixels (required!)')
    while True:
        try:
            ref_obj_m = float(input(f'{Y}📏 Length of reference object in meters (e.g., 1.0 for a 1-meter stick): {W}').strip())
            ref_obj_px = float(input(f'{Y}👾 Pixel length of reference object in video (e.g., 100.0 pixels): {W}').strip())
            if ref_obj_m <= 0 or ref_obj_px <= 0:
                print(f'{R}⚠ Values must be positive numbers, {rerun_req}')
                continue
            px_per_m = float(ref_obj_px / ref_obj_m)
            print(f'\n{G}🤗 We are done, thanks for your patience!')
            break
        except ValueError:
            print(f'{R}⚠ Please provide inputs as numbers, {rerun_req}')

def homepage():
    banner()
    print(B + " [1]" + W + " ➕ Add New Model")
    print(B + " [2]" + W + " 📂 View Models")
    print(B + " [3]" + W + " ✨ Update Existing Model")
    print(B + " [4]" + W + " 🗑 Delete Existing Model")
    print(B + " [5]" + W + " 🔍 Compare Models")
    print(B + " [6]" + W + " 📊 Generate Overall Report, done for this phase of iterations!")
    print(B + " [7]" + W + " 🚪 Exit")
    print(C + "─" * 50)
    return input(Y + "Select an option (1-7): " + W).strip()

# ===== Log New Model =====
def log_new_model(model_name=None, model_version=1.0):
    global breakw
    need = "Add" if not model_name else 'Update'
    print(C + f"\n--- {need} New Model ---\n") # state management like react? 🤪

    # ---- Take important inputs ------
    if not model_name:
        model_name = input(Y + "✈  Model Name: " + W).strip()
        if not model_name or not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
            model_name = input(f'{R}⚠ Please provide a valid name (alphanumeric, underscores, or hyphens only): {W}').strip()
            if not model_name or not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
                print(f'{R}🤦 You entered wrong name again! {rerun_req}')
                return None
    
    outputs_dir = os.path.join("outputs", model_name, str(model_version))
    if os.path.exists(outputs_dir):
        while True:
            con = input(f'{M}Seems like a model named {model_name} already exists, do you want to change it?(y/N): {W}').strip().lower()
            if con == 'y':
                os.makedirs(outputs_dir, exist_ok=True)
                break
            elif con == 'n':
                model_name = input(f'{Y}✒ Enter a different model name: {W}').strip()
                if not model_name or not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
                    print(f'{R}🤦 Invalid name! {rerun_req}')
                    return None
                outputs_dir = os.path.join("outputs", model_name, str(model_version))
                break
            else:
                print(f'{Y}❌ Please enter y or n, {rerun_req}')

    def path_validation_msg():
        print(f'{R}⚠  Dude, will you never provide a valid path?')

    # picture input with validation
    in_picture_path = input(Y + "🖼  Model's Picture: " + W).strip()
    picture_path = os.path.join(outputs_dir, "model_picture.jpg")
    os.makedirs(os.path.dirname(picture_path), exist_ok=True)
    if os.path.exists(in_picture_path):
        img = cv.imread(in_picture_path)
        if img is not None:
            cv.imwrite(picture_path, img)
            print(G + f"\n✅ Picture saved → {picture_path}")
        else:
            print(R + f"\n❌ Error: Could not read image from '{in_picture_path}'")
            retry_path = input(f"{Y}📷 Please enter a valid picture path: {W}").strip()
            if os.path.exists(retry_path):
                img = cv.imread(retry_path)
                if img is not None:
                    cv.imwrite(picture_path, img)
                    print(G + f"\n✅ Picture saved → {picture_path}")
                else:
                    print(R + f"\n❌ Error: Could not read image from '{retry_path}'")
                    path_validation_msg()
                    return None
            else:
                path_validation_msg()
                return None
    else:
        print(R + f"\n❌ Error: Image path '{in_picture_path}' does not exist.")
        retry_path = input(f"{Y}📷 Please enter a valid picture path: {W}").strip()
        if os.path.exists(retry_path):
            img = cv.imread(retry_path)
            if img is not None:
                cv.imwrite(picture_path, img)
                print(G + f"\n✅ Picture saved → {picture_path}")
            else:
                print(R + f"\n❌ Error: Could not read image from '{retry_path}'")
                path_validation_msg()
                return None
        else:
            path_validation_msg()
            return None
    
    # video input with validation
    in_videos_path = input(Y + "🎥 Flight Videos (comma separated): " + W).strip()
    videos_dir = os.path.join(outputs_dir, "Flight Videos")
    os.makedirs(videos_dir, exist_ok=True)

    video_paths = (
        [v.strip() for v in in_videos_path.split(",")] if in_videos_path else []
    )
    if not video_paths:
        print(f"{R}❌ No video paths provided, {rerun_req}")
        return None

    for video_path in video_paths:
        if os.path.exists(video_path) and video_path.lower().endswith(('.mp4', '.avi', '.mov')):
            shutil.copy(video_path, videos_dir)
            print(G + f"✅ Video copied: {video_path} → {videos_dir}")
        else:
            print(f"{R}❌ Invalid or unsupported video format: {video_path}")
            retry_path = input(f"{Y}📹 Please enter a valid video path for {video_path}: {W}").strip()
            if os.path.exists(retry_path) and retry_path.lower().endswith(('.mp4', '.avi', '.mov')):
                shutil.copy(retry_path, videos_dir)
                print(G + f"✅ Video copied: {retry_path} → {videos_dir}")
            else:
                print(f"{R}❌ Invalid or unsupported video format: {retry_path}")
                return None

    preview_mode = True # toggle this if u don't want to watch the video, but note that it wil take the same amount of time!
    design_notes = input(Y + "📝 Design Notes: " + W).strip()

    # ------ Save datas ------
    # Save Metrics
    metadata_path = os.path.join(outputs_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Version: {model_version}\n")
        f.write(f"Design Notes: {design_notes}\n")
    print(f"{Fore.CYAN}{Style.BRIGHT}📓 Metadata Saved to {metadata_path}")

    # per video processing
    for video_path in video_paths:
        distance_px = 0
        trajectory_points = []
        frame_count = 0
        first_detected = None # these are for accurate airtime
        last_detected = None

        video_n = os.path.basename(video_path)[:-4]
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(R + f"❌ Error: Could not open video '{video_path}'")
            continue

        fps = cap.get(cv.CAP_PROP_FPS) if cap.get(cv.CAP_PROP_FPS) > 0 else 30

        while True:
            # detect the plane and draw trajectory line live and also draw a box around it with label
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv.resize(frame, (960, 540))
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            lower, upper = np.array([35, 50, 102]), np.array([179, 255, 255])
            mask = cv.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

            contours, _ = cv.findContours(
                mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            detected_this_frame = False

            for contour in contours:
                area = cv.contourArea(contour)
                plane_count = 0
                if area > 500:
                    if plane_count > 1:
                        print(f'{R}⚠ More than one plane detected, please provide a video where there is only one blue object which is your plane, {rerun_req}')
                        return None
                    plane_count+=1
                    detected_this_frame = True
                    x, y, w, h = cv.boundingRect(contour)
                    center = (x + w // 2, y + h // 2)
                    trajectory_points.append(center)

                    cv.putText(
                        frame,
                        "Plane",
                        (x, y - 10),
                        cv.FONT_HERSHEY_DUPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    draw_rectangle(frame, x, y, w, h)
                    if len(trajectory_points) > 1:
                        cv.polylines(
                            frame, [np.array(trajectory_points)], False, (0, 255, 0), 2
                        )
                    draw_grid(frame)
                    cv.putText(
                        frame,
                        f"{center}",
                        (frame.shape[1] - 90, frame.shape[0] - 10),
                        cv.FONT_HERSHEY_DUPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )

            if detected_this_frame:
                if first_detected is None:
                    first_detected = frame_count
                last_detected = frame_count

            if preview_mode:
                cv.imshow(f"Video for {model_name}", frame)
                if cv.waitKey(20) & 0xFF == ord("q"):
                    break

        cap.release()
        if preview_mode:
            cv.destroyAllWindows()

        if not trajectory_points:
            print(f'{R}⚠ No plane was detected, Please try to provide a video where you clearly show the plane and make sure that there is not another or more blue object(s), {rerun_req}')
            return None
        
        # Save the coordinates
        coordinates_path_f = os.path.join(
            outputs_dir, "Flight Coordinates", f"{video_n}_coordinates.csv"
        )
        os.makedirs(os.path.dirname(coordinates_path_f), exist_ok=True)
        with open(coordinates_path_f, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y"])
            for x, y in trajectory_points:
                writer.writerow([x, y])
        print(f"{G}📈 Flight Trajectory Coordinates saved to {coordinates_path_f}")

        # Save the metrics
        if trajectory_points:
            distance_px = max(x for x, y in trajectory_points) - min(
                x for x, y in trajectory_points
            ) # horizontal distance
        else:
            distance_px = 0

        if first_detected is not None and last_detected is not None:
            airtime_frames = last_detected - first_detected + 1
            airtime = airtime_frames / fps # see, we used that vars for acc airtime :)
        else:
            airtime = 0

        # Real Metrics Conversion
        distance_m = distance_px * px_per_m
        speed = distance_m / airtime if airtime > 0 else 0

        # Take the stability score as input from user
        try:
            stability_score = int(input(f"{C}📝 Enter stability score for {model_name} (0-10): {W}").strip())
            if stability_score > 10:
                print(f"{M} We understand you might be amused of it's stability, bro just go with 10 if you like it then! the good news though, you don't have to because we will count it as 10")
                stability_score = 10
            elif stability_score < 0:
                print(f'{M} Bruh, did that fly that bad? but you are only allowed to enter in range of 0 to 10. We will take that as 0, btw!')
                stability_score = 0
        except ValueError:
            print(f'{R}⚠ Only integers between 0 and 10 allowed, {rerun_req}')
            return None
        

        metrics_path_f = os.path.join(
            outputs_dir, "Flight Metrics", f"{video_n}_metrics.csv"
        )
        os.makedirs(os.path.dirname(metrics_path_f), exist_ok=True)
        with open(metrics_path_f, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Distance(m)", "Airtime(s)", "Speed(m/s)", "Stability"])
            writer.writerow(
                [
                    f"{distance_m:.2f}",
                    f"{airtime:.2f}",
                    f"{speed:.2f}",
                    stability_score,
                ]
            )
        print(f"{B}📐 Metrics saved to {metrics_path_f}\n")

    # ---- Averages -----
    # Metrics
    tot_distance = tot_airtime = tot_speed = tot_stability = 0 # declare all as 0 initially
    metrics_path = os.path.join(outputs_dir, "Flight Metrics")
    mfile_count = 0

    if os.path.exists(metrics_path):
        for filename in os.listdir(metrics_path):
            if filename.endswith("_metrics.csv"):
                filepath = os.path.join(metrics_path, filename)
                try:
                    with open(filepath, "r", encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            tot_distance += float(row["Distance(m)"])
                            tot_airtime += float(row["Airtime(s)"])
                            tot_speed += float(row["Speed(m/s)"])
                            tot_stability += float(row["Stability"])
                            mfile_count += 1
                except (csv.Error, KeyError, ValueError):
                    print(f"{R}❌ Error reading metrics file {filepath}, {rerun_req}")
                    return None

    avg_distance = tot_distance / mfile_count if mfile_count > 0 else 0
    avg_airtime = tot_airtime / mfile_count if mfile_count > 0 else 0
    avg_speed = tot_speed / mfile_count if mfile_count > 0 else 0
    avg_stability = tot_stability / mfile_count if mfile_count > 0 else 0

    avg_metrics_path = os.path.join(outputs_dir, "avg_metrics.csv")
    with open(avg_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Distance(m)", "Airtime(s)", "Speed(m/s)", "Stability"])
        writer.writerow(
            [
                f"{avg_distance:.2f}",
                f"{avg_airtime:.2f}",
                f"{avg_speed:.2f}",
                f"{avg_stability:.2f}",
            ]
        )
    print(f"{G}💾 Average metrics of {model_name} saved to {avg_metrics_path}")

    # Coords
    coordinates_path = os.path.join(outputs_dir, "Flight Coordinates")
    avg_coordinates_path = os.path.join(outputs_dir, "avg_coordinates.csv")
    all_trajectories = []

    if os.path.exists(coordinates_path):
        for filename in sorted(os.listdir(coordinates_path)):
            if filename.endswith("_coordinates.csv"):
                filepath = os.path.join(coordinates_path, filename)
                points = []
                try:
                    with open(filepath, "r", encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            points.append((float(row["X"]), float(row["Y"])))
                    if points:
                        all_trajectories.append(points)
                except (csv.Error, KeyError, ValueError):
                    print(f"{R}❌ Error reading coordinates file {filepath}, {rerun_req}")
                    return None

    avg_points = []
    if all_trajectories:
        try:
            min_length = min(len(traj) for traj in all_trajectories)
            num_videos = len(all_trajectories)
            for i in range(min_length):
                avg_x = sum(traj[i][0] for traj in all_trajectories) / num_videos
                avg_y = sum(traj[i][1] for traj in all_trajectories) / num_videos
                avg_points.append((avg_x, avg_y))
        except ValueError:
            print(f"{R}❌ No valid trajectories to average, {rerun_req}")
            return None

    with open(avg_coordinates_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        for x, y in avg_points:
            writer.writerow([f"{x:.2f}", f"{y:.2f}"])
    print(f"{G}💾 Average Coordinates of {model_name} saved to {avg_coordinates_path}")

    # -------- Make the Trajectory Graphs
    plt.figure(figsize=(8, 6))
    if os.path.exists(coordinates_path):
        for filename in os.listdir(coordinates_path):
            if filename.endswith("_coordinates.csv"):
                filepath = os.path.join(coordinates_path, filename)
                x_coords, y_coords = [], []
                try:
                    with open(filepath, "r", encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            x_coords.append(float(row["X"]))
                            y_coords.append(float(row["Y"]))
                    plt.plot(x_coords, y_coords, color="cyan")
                except (csv.Error, KeyError, ValueError):
                    print(f"{R}❌ Error reading coordinates for graph {filepath}, {rerun_req}")
                    return None

    if avg_points:
        avg_x_coords, avg_y_coords = zip(*avg_points)
        plt.plot(
            avg_x_coords,
            avg_y_coords,
            color="red",
            linewidth=2,
            label="Average Trajectory",
        )

    plt.title(f"Trajectory Graph for {model_name} v{model_version}")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle="--", alpha=0.3)
    if all_trajectories:
        plt.legend()

    trajectory_graph_path = os.path.join(outputs_dir, "trajectory_graph.png")
    plt.savefig(trajectory_graph_path)
    plt.close()
    print(f"{G}📉 Trajectory Graph saved to → {trajectory_graph_path}")

    # Generate enhanced Markdown report with UTF-8 encoding
    report_path = os.path.join(outputs_dir, 'report.md')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f'# ✈️ {model_name} v{model_version} Flight Report\n\n')

            # Overview
            f.write('## Overview\n')
            if os.path.exists(os.path.join(outputs_dir, 'model_picture.jpg')):
                f.write('![Model Design](model_picture.jpg)\n')
            else:
                f.write('Model image not found—add to document design!\n')
            f.write(f'- **Model**: {model_name}\n')
            f.write(f'- **Version**: {model_version}\n')
            f.write(f'- **Design Notes**: {design_notes}\n')
            f.write(f'- **Created**: {datetime.now().strftime("%Y-%m-%d %I:%M %p +06")}\n\n')

            # Trajectory Graph
            f.write('## Trajectory Graph\n')
            if os.path.exists(os.path.join(outputs_dir, 'trajectory_graph.png')):
                f.write('![Flight Trajectory](trajectory_graph.png)\n')
            else:
                f.write('Trajectory graph not generated!\n')
            f.write('\n')

            is_best_distance = True
            is_most_stable = False

            # Average Metrics
            avg_metrics = {}
            metrics_path = os.path.join(outputs_dir, 'avg_metrics.csv')
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r', encoding='utf-8') as mf:
                        reader = csv.DictReader(f)
                        for row in reader:
                            avg_metrics = {
                                'Distance': float(row['Distance(m)']),
                                'Airtime': float(row['Airtime(s)']),
                                'Speed': float(row['Speed(m/s)']),
                                'Stability': float(row['Stability'])
                            }
                    f.write('## Average Metrics\n')
                    f.write('| Metric    | Value      |\n')
                    f.write('|--|--|\n')
                    for metric, value in avg_metrics.items():
                        f.write(f'| {metric} | {value:.2f} {"px" if metric == "Distance" else "s" if metric == "Airtime" else "px/s" if metric == "Speed" else ""} |\n')
                
                    # Check for achievement (compare to other versions)
                    model_dir = os.path.join('outputs', model_name)
                    max_distance = avg_metrics['Distance']
                    if os.path.exists(model_dir):
                        for version in os.listdir(model_dir):
                            if version != str(model_version) and os.path.isdir(os.path.join(model_dir, version)):
                                v_metrics_path = os.path.join(model_dir, version, 'avg_metrics.csv')
                                if os.path.exists(v_metrics_path):
                                    try:
                                        with open(v_metrics_path, 'r', encoding='utf-8') as vf:
                                            reader = csv.DictReader(vf)
                                            for row in reader:
                                                if float(row['Distance(m)']) > max_distance:
                                                    is_best_distance = False
                                                    break
                                    except (csv.Error, KeyError, ValueError):
                                        print(f"{R}❌ Error reading metrics for comparison {v_metrics_path}, {rerun_req}")
                                        return None
                except (csv.Error, KeyError, ValueError):
                    f.write('## Average Metrics\n')
                    f.write('No flight data recorded! Conduct tests, engineer! ✈️\n')
                    return None
            else:
                f.write('## Average Metrics\n')
                f.write('No flight data recorded! Conduct tests, engineer! ✈️\n')
            f.write('\n')

            # Per-Video Analysis
            video_metrics = []
            flight_metrics_dir = os.path.join(outputs_dir, 'Flight Metrics')
            if os.path.exists(flight_metrics_dir) and os.listdir(flight_metrics_dir):
                for video_file in os.listdir(flight_metrics_dir):
                    if video_file.endswith('_metrics.csv'):
                        video_name = video_file.replace('_metrics.csv', '')
                        try:
                            with open(os.path.join(flight_metrics_dir, video_file), 'r', encoding='utf-8') as vf:
                                reader = csv.DictReader(vf)
                                for row in reader:
                                    video_metrics.append({
                                        'Video Name': video_name,
                                        'Distance': float(row['Distance(m)']),
                                        'Airtime': float(row['Airtime(s)']),
                                        'Speed': float(row['Speed(m/s)']),
                                        'Stability': float(row['Stability']),
                                        'Performance Note': ''
                                    })
                            # Tag performance notes
                            if video_metrics:
                                max_distance = max(vm['Distance'] for vm in video_metrics)
                                max_stability = max(vm['Stability'] for vm in video_metrics)
                                for vm in video_metrics:
                                    if vm['Distance'] == max_distance:
                                        vm['Performance Note'] = 'Longest flight'
                                    elif vm['Stability'] == max_stability:
                                        is_most_stable = True
                                        vm['Performance Note'] = 'Most stable'
                        except (csv.Error, KeyError, ValueError):
                            print(f"{R}❌ Error reading video metrics {video_file}, {rerun_req}")
                            return None
                   
            else:
                f.write('## Per-Video Analysis\n')
                f.write('No flight data recorded! Conduct tests, engineer! ✈️\n')
            f.write('\n')

            # Summary   
            f.write('## Summary\n')
            f.write(f'- {("Satisfictory distance" if is_best_distance else "Great Distance coverage" if distance_px > 770 else "The distance is not satisfying!")}\n')
            f.write(f'- {"Really Stable" if is_most_stable else "Strong stability" if stability_score > 7 else "fairly stable" if stability_score > 5 else "Unacceptably Bad stablity, Please consider optimal weight distribution and add dihedral"}')           
            # Social Prompt
            f.write('\n---\n')
            f.write('**Share your findings on X with #LucidraftDeltaX to discuss with the research community! 🚀**\n')

        print(f"{G}📄 Report saved to → {report_path}")
    except Exception as e:
        print(f"{R}❌ Turbulence! Failed to generate report: {e}. Check files and try again! ✈️")

    # one-liner
    print(f'{B}✈  {model_name} v{model_version} --> {distance_m}m, {speed}m/s, {stability_score}')

    return model_name, model_version

# ================= View Model ========================
def create_combined_table(data):
    """
    Generates a single ASCII table with model names as internal headers.
    """
    all_data = []

    # Check if data is empty before trying to access it
    if not data:
        return "No data to display."
        
    # Get the column headers from the first entry to ensure consistency
    first_model = list(data.keys())[0]
    # Check if the first model has any versions before trying to access them
    if not data[first_model]:
        return "No version data to display."
    
    headers = list(data[first_model][0].keys())

    for model_name, versions in data.items():
        if not versions:
            continue  # Skip models with no version data
            
        # Add the model name as a list with a blank for each column
        all_data.append([model_name] + [''] * (len(headers) - 1))
        
        # Add the column headers for the version data
        all_data.append(headers)
        
        # Add the data rows
        for version in versions:
            all_data.append(list(version.values()))

    # Generate the table with an empty list for headers to avoid the TypeError
    return tabulate(all_data, headers=[], tablefmt="grid")

def view_models():
    outb = 'outputs'
    models = {}  

    if not os.path.exists(outb):
        print(R + "❌ No models found! Create some planes first! ✈️")
        return models  # Return an empty dict if no models exist

    for model_name in os.listdir(outb):
        model_path = f'{outb}/{model_name}'

        if os.path.isdir(model_path):
            # Initialize the list for the current model *before* the inner loop
            models[model_name] = [] 
            
            for version in os.listdir(model_path):
                version_path = f'{model_path}/{version}/avg_metrics.csv'
                
                # Check if the metrics file exists before trying to open it
                if os.path.exists(version_path):
                    try:
                        with open(version_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            row = next(reader, None)
                            if row is None or any(k not in row for k in ["Distance(m)", "Speed(m/s)", "Stability"]):
                                print(f"{R}❌ Invalid metrics file {version_path}, {rerun_req}")
                                continue
                            models[model_name].append({
                                'version': str(version),
                                'distance': str(row['Distance(m)']),
                                'speed': str(row['Speed(m/s)']),
                                'stability': str(row['Stability'])
                            })
                    except (csv.Error, KeyError, ValueError):
                        print(f"{R}❌ Error reading metrics file {version_path}, {rerun_req}")
                        continue
                            
    table = create_combined_table(models)
    if table == "No data to display." or table == "No version data to display.":
        print(R + f"❌ {table} Create some planes first! ✈️")
    else:
        print(B + table)
    return models

# =================== Update Model ======================
def update_model():
    model_name = input(f'\n{Y}Model Name: {W}').strip()
    if not model_name or not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
        print(f'{R}⚠ Invalid model name! Use alphanumeric, underscores, or hyphens, {rerun_req}')
        return
    if not os.path.exists(f'outputs/{model_name}'):
        print(f'{R}⚠ No models found! Please first create one, then update if needed')
        return
    try:
        prev_version = get_prev_v(model_name)
        if prev_version is None:
            print(f'{R}⚠ No valid versions found for {model_name}, {rerun_req}')
            return
        new_version = round(prev_version + 0.1, 1)
        log_new_model(model_name, new_version)
        compare(model_name, prev_version, model_name, new_version)
    except ValueError:
        print(f'{R}⚠ Invalid version format in {model_name} directory, {rerun_req}')

def get_prev_v(model_name):
    model_path = os.path.join('outputs', model_name)
    if not os.path.exists(model_path):
        return None
    vs = []
    for v in os.listdir(model_path):
        try:
            vs.append(float(v))
        except ValueError:
            continue
    return max(vs) if vs else None

# ===================== Delete Model =============
def delete_model():
    mv = input(f'\n{Y}Model name(include version with a space after model if you want a specific version to be deleted): {W}').strip()
    if mv:
        model = mv.split(' ')[0] 
    else:
        print(f'{Y} Invalid input! {rerun_req}')
        return None
    version = None
    try:
        if len(mv.split(' ')) >= 2 and len(mv.split(' ')) < 3:
            version = mv.split(' ')[1]
        elif len(mv.split(' ')) >= 3:
            print(f'{R} Please first enter the model name and if you want to delete just one version of that model, just add space after the model name and write down the version in float\n. Multiple version deletion is not supported!')
        
        shutil.rmtree(f'outputs/{model}/{version if version else ''}')
        print(f'{G}Successfully removed {model} {version if version else ''}!')
    except:
        print(f"{R}⚠ Please give input in valid format, the model name only if you want to delete the entire model and to only delte a version of a model, after typing the model name, hit space and type version in float(e.g, 'Eagle 1.0')")

# =============== Comare Model ===============
def compare(model_name1=None, model1_v=None, model_name2=None, model2_v=None):
    if not model_name1 and not model1_v and not model_name2 and not model2_v:
        try:
            model_name1, model1_v = input(f"{Y} Please enter the first model's name and version(model's name <space> version): {W}").split(' ')
            model_name2, model2_v = input(f"{Y} Please enter the second model's name and version(model's name <space> version): {W}").split(' ')
            if not re.match(r'^[a-zA-Z0-9_-]+$', model_name1) or not re.match(r'^[a-zA-Z0-9_-]+$', model_name2):
                print(f'{R}⚠ Model names must be alphanumeric, underscores, or hyphens, {rerun_req}')
                return
            if not re.match(r'^\d+\.\d$', model1_v) or not re.match(r'^\d+\.\d$', model2_v):
                print(f'{R}⚠ Versions must be floats (e.g., 1.0), {rerun_req}')
                return
        except ValueError:
            print(f'{R}⚠ Invalid input format, use "model_name version" (e.g., "Eagle 1.0"), {rerun_req}')
            return

    try:
        path1 = f'outputs/{model_name1}/{model1_v}/avg_metrics.csv'
        path2 = f'outputs/{model_name2}/{model2_v}/avg_metrics.csv'
        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f'{R}⚠ One or both models not found, {rerun_req}')
            return

        with open(path1, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row is None or any(k not in row for k in ["Distance(m)", "Speed(m/s)", "Stability"]):
                print(f'{R}⚠ Invalid metrics file {path1}, {rerun_req}')
                return
            model1_distance = float(row["Distance(m)"])
            model1_speed = float(row["Speed(m/s)"])
            model1_stability = float(row["Stability"])
        
        with open(path2, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row is None or any(k not in row for k in ["Distance(m)", "Speed(m/s)", "Stability"]):
                print(f'{R}⚠ Invalid metrics file {path2}, {rerun_req}')
                return
            model2_distance = float(row["Distance(m)"])
            model2_speed = float(row["Speed(m/s)"])
            model2_stability = float(row['Stability'])

        print(f"\n{B}⚖ Comparing {model_name2}_v{model2_v} to {model_name1}_v{model1_v}\n")
        distance_gap = model2_distance - model1_distance
        stability_gap = model2_stability - model1_stability
        speed_gap = model2_speed - model1_speed

        def get_sign(v):
            return '+' if v > 0 else ''

        print(f'{Y} Distance: {get_sign(distance_gap)}{distance_gap}')
        print(f'{Y} Stability: {get_sign(stability_gap)}{stability_gap}')
        print(f'{Y} Speed: {get_sign(speed_gap)}{speed_gap}')
    except (csv.Error, KeyError, ValueError):
        print(f'{R}⚠ Invalid request, {rerun_req}')
        global breakw
        breakw = True

# ============== Generate Overall Report =====================
def generate_overall():
    """
    Generate an overall report summarizing all models' metrics.
    Creates a terminal table, CSV, bar chart, and Markdown report for research papers.
    Matches the specified research-paper-ready format with detailed insights.
    """
    base_dir = "outputs"
    report_file = os.path.join(base_dir, "overall_report.md")
    csv_file = os.path.join(base_dir, "overall_report.csv")
    chart_file = os.path.join(base_dir, "overall_report.png")

    # Collect data
    records = []
    if not os.path.exists(base_dir):
        print(R + "❌ Hangar’s empty! Fold some planes, pilot! ✈️")
        return

    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue
        for version in os.listdir(model_path):
            csv_path = os.path.join(model_path, version, "avg_metrics.csv")
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        row = next(reader, None)
                        if row is None or any(k not in row for k in ["Distance(m)", "Stability", "Speed(m/s)"]):
                            print(f"{R}❌ Invalid metrics file {csv_path}, try option 1! ✈️")
                            continue
                        records.append({
                            "Model": model,
                            "Version": version,
                            "Distance(m)": float(row["Distance(m)"]),
                            "Stability": float(row["Stability"]),
                            "Speed(m/s)": float(row["Speed(m/s)"])
                        })
                except (csv.Error, KeyError, ValueError):
                    print(R + f"❌ Crash landing! Bad metrics file at {csv_path}. Try option 1! ✈️")
                    continue

    if not records:
        print(R + "❌ No metrics found! Create models with option 1, ace! ✈️")
        return

    # Sort by Distance, Stability, Speed (descending)
    records.sort(key=lambda x: (x["Distance(m)"], x["Stability"], x["Speed(m/s)"]), reverse=True)

    # Prepare table data with ★ for top 3
    table_data = []
    for i, record in enumerate(records):
        model_name = f"★ {record['Model']}" if i < 3 else record["Model"]
        table_data.append([
            model_name,
            record["Version"],
            record["Distance(m)"],
            record["Stability"],
            record["Speed(m/s)"]
        ])

    # Terminal table
    print(C + "✈️ Lucid Raft Delta-X: Fleet Performance Report ✈️")
    print(C + "─" * 50)
    print(tabulate(
        table_data,
        headers=["Model", "Version", "Distance (px)", "Stability", "Speed (px/s)"],
        tablefmt="grid",
        floatfmt=".2f"
    ))
    print(Y + "Top 3 planes marked with ★! Great flying, pilots!")

    # Save CSV
    os.makedirs(base_dir, exist_ok=True)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Version", "Distance(m)", "Stability", "Speed(m/s)"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "Model": record["Model"],
                "Version": record["Version"],
                "Distance(m)": f"{record['Distance(m)']:.2f}",
                "Stability": f"{record['Stability']:.2f}",
                "Speed(m/s)": f"{record['Speed(m/s)']:.2f}"
            })
    print(G + f"📊 Overall report saved to → {csv_file}")

    # Bar chart for distance
    plt.figure(figsize=(8, 6))
    labels = [f"{r['Model']}_v{r['Version']}" for r in records]
    distances = [r["Distance(m)"] for r in records]
    plt.bar(labels, distances, color="#00FFFF")
    plt.title("Fleet Distance Comparison")
    plt.xlabel("Model and Version")
    plt.ylabel("Distance (px)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(chart_file)
    plt.close()
    print(G + f"📉 Chart saved to → {chart_file}")

    # Markdown report
    now = datetime.now().strftime("%Y-%m-%d %I:%M %p +06")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Overall Report\n")
        f.write(f"*Phase Summary of Paper Aircraft Engineering Iterations*  \n")
        f.write(f"Generated: {now}  \n\n")
        f.write("---\n\n")
        f.write("## 🚀 Models & Versions Overview (Sorted by Distance → Stability → Speed)\n\n")
        f.write("| Model | Version | Avg Distance (px) | Avg Stability | Avg Speed (px/s) |\n")
        f.write("|-------|---------|-------------------|---------------|------------------|\n")
        for i, record in enumerate(records):
            model_name = f"★ {record['Model']}" if i < 3 else record["Model"]
            f.write(f"| {model_name} | {record['Version']} | {record['Distance(m)']:.2f} | {record['Stability']:.2f} | {record['Speed(m/s)']:.2f} |\n")
        f.write("\n---\n\n")
        f.write("## 🏆 Top Performers\n")
        best_distance = max(records, key=lambda x: x["Distance(m)"])
        best_stability = max(records, key=lambda x: x["Stability"])
        best_speed = max(records, key=lambda x: x["Speed(m/s)"])
        f.write(f"- **Longest Distance** → {best_distance['Model']} v{best_distance['Version']} ({best_distance['Distance(m)']:.2f} px) ✅\n")
        f.write(f"- **Most Stable** → {best_stability['Model']} v{best_stability['Version']} ({best_stability['Stability']:.2f})\n")
        f.write(f"- **Fastest Speed** → {best_speed['Model']} v{best_speed['Version']} ({best_speed['Speed(m/s)']:.2f} px/s)\n\n")
        f.write("---\n\n")
        f.write("## 📈 Comparative Trends\n\n")
        f.write("### Distance Progression\n")
        for model in set(r["Model"] for r in records):
            model_records = [r for r in records if r["Model"] == model]
            if len(model_records) > 1:
                versions = sorted(model_records, key=lambda x: float(x["Version"]))
                diff = versions[0]["Distance(m)"] - versions[-1]["Distance(m)"]
                f.write(f"- {model} branch {'dominates with consistent gains' if diff > 0 else 'shows modest progress' if diff > -50 else 'stagnated'} ({'+' if diff > 0 else ''}{diff:.1f} px from v{versions[-1]['Version']} → v{versions[0]['Version']}).\n")
            else:
                f.write(f"- {model} branch has single version, no progression data.\n")
        f.write("\n### Stability vs Distance\n")
        best = records[0]
        f.write(f"- **Best Stability Plane** → {best_stability['Model']} v{best_stability['Version']} ({best_stability['Stability']:.2f})\n")
        f.write(f"- **Best Distance Plane** → {best_distance['Model']} v{best_distance['Version']} ({best_distance['Distance(m)']:.2f} px)\n")
        f.write(f"- Clear correlation: distance and stability peaked together in {best['Model']} branch.\n\n")
        f.write(f"![Fleet Distance Comparison](overall_report.png)\n\n")
        f.write("---\n\n")
        f.write("## 🔬 Insights\n")
        f.write(f"- **{best['Model']} branch** = king. Strong lead in distance, stability, and speed.\n")
        if len(records) > 1:
            second = records[1]
            f.write(f"- **{second['Model']} branch** = decent, but already lagging too far behind.\n")
        if len(records) > 2:
            worst = records[-1]
            f.write(f"- **{worst['Model']} branch** = inefficient — cut losses here.\n")
        f.write("- Micro-fold adjustments (seen in top performers) yield massive improvements.\n\n")
        f.write("---\n\n")
        f.write("## 📌 Phase Verdict\n")
        f.write(f"- **Total Models**: {len(set(r['Model'] for r in records))}\n")
        f.write(f"- **Total Versions Tested**: {len(records)}\n")
        f.write(f"- **Best Candidate:** {best['Model']} v{best['Version']}\n")
        f.write(f"- **Recommendation:** Keep pushing the *{best['Model']}* branch. {second['Model'] if len(records) > 1 else 'Other models'} can stay archived for reference. {worst['Model'] if len(records) > 2 else 'Underperforming models'} should be abandoned.\n")
        f.write(f"- If you go way more → {best['Model']} could become the city-burner model. 🔥\n\n")
        f.write("---\n\n")
        f.write("## 📎 Appendices\n")
        for record in records:
            report_path = f"outputs/{record['Model']}/{record['Version']}/report.md"
            if os.path.exists(report_path):
                f.write(f"- [{record['Model']} v{record['Version']} Detailed Report]({record['Model']}/{record['Version']}/report.md)\n")
        f.write("\n---\n\n")
        f.write("Share this on X with **#LucidraftDeltaX** \n")
    print(G + f"📄 Markdown report saved to → {report_file}")

    # Social prompt
    print(Y + "🚀 Share your fleet’s stats on X with #LucidraftDeltaX! Show the world your top planes! ✈️")

# ===== Basic Utility Funcs =======
def draw_rectangle(img, x, y, w, h):
    try:
        color = (0, 255, 255)
        thickness = 2
        length = 20
        corners = [
            ((x, y), (x + length, y), (x, y + length)), # top-left
            ((x + w, y), (x + w - length, y), (x + w, y + length)), # top-right
            ((x, y + h), (x + length, y + h), (x, y + h - length)), # bot-left
            ((x + w, y + h), (x + w - length, y + h), (x + w, y + h - length)), # bot-right
        ]
        # Draw the rectangle
        for pt1, pt2, pt3 in corners:
            cv.line(img, pt1, pt2, color, thickness)
            cv.line(img, pt1, pt3, color, thickness)

        # Draw a circle and cross
        cv.line(img, (x, y), (x + w, y + h), color, thickness)
        cv.line(img, (x + w, y), (x, y + h), color, thickness)
        center = (x + w // 2, y + h // 2)
        for i in range(6, 0, -1):
            alpha = 0.1 * i
            overlay = img.copy()
            cv.circle(overlay, center, i * 3, color, -1)
            cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv.circle(img, center, 4, color, -1)
    except:
        print(f'{R}⚠ Seems like frame has failed to process, {rerun_req}')
        global breakw
        breakw = True

def draw_grid(img, spacing=60, color=(0, 255, 255), thickness=1, alpha=0.1): # Just draws a grid on the video to make it cool, maybe 🤷‍♂️
    overlay = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, spacing):
        cv.line(overlay, (x, 0), (x, h), color, thickness)
    for y in range(0, h, spacing):
        cv.line(overlay, (0, y), (w, y), color, thickness)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# RUN!
if __name__ == "__main__":
    while True:
        breakw = False  # Reset breakw each loop
        choice = homepage()
        try:
            choice_int = int(choice)
            if choice_int < 1 or choice_int > 7:
                print(f'{R}⚠ Please select a number between 1 and 7, {rerun_req}')
                pause()
                continue
        except ValueError:
            print(f'{R}⚠ Invalid input, please enter a number (1-7), {rerun_req}')
            pause()
            continue
        if choice == "7":
            print(
                B + "\n------------------ Goodbye! Fly high! ✈️ --------------------\n"
            )
            break
        elif choice == "1":
            log_new_model()
            pause()
        elif choice == "2":
            view_models()
        elif choice == "3":
            update_model()
            pause()
        elif choice == "4":
            delete_model()
            pause()
        elif choice == "5":
            compare()
            pause()
        elif choice == '6':
            generate_overall()
            pause()