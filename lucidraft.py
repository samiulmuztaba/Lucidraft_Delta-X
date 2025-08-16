import os
import cv2 as cv
from colorama import init, Fore, Style
import shutil
import numpy as np
import csv
import matplotlib.pyplot as plt
from tabulate import tabulate
import datetime

init(autoreset=True)

# Colors
C = Fore.CYAN + Style.BRIGHT
B = Fore.BLUE + Style.BRIGHT
Y = Fore.YELLOW + Style.BRIGHT
W = Fore.WHITE + Style.BRIGHT
G = Fore.GREEN + Style.BRIGHT
R = Fore.RED + Style.BRIGHT
M = Fore.MAGENTA + Style.BRIGHT


def banner():
    logo = f"""
{C}   ╔══════════════════════════════════════════════╗
{C}   ║{B}   L U C I D R A F T   D E L T A - X   1 . 0  {C}║
{C}   ║{M} Advanced Paper Plane Engineering Terminal    {C}║
{C}   ╚══════════════════════════════════════════════╝
"""
    print(logo)


def homepage():
    banner()
    print(B + " [1]" + W + " ➕ Add New Model")
    print(B + " [2]" + W + " ✨ Update Existing Model")
    print(B + " [3]" + W + " 📂 View Models")
    print(B + " [4]" + W + " 🔍 Compare Models")
    print(B + " [5]" + W + " 📊 Generate Overall Report, done for this phase of iterations!")
    print(B + " [6]" + W + " ❌ Exit")
    print(C + "─" * 50)
    return input(Y + "Select an option (1-6): " + W).strip()


def log_new_model(model_name=None, model_version=1.0):
    need = "Add" if not model_name else 'Update'
    print(C + f"\n--- {need} New Model ---\n")
    if not model_name:
        model_name = input(Y + "✈  Model Name: " + W).strip()
    # model_version = 1.0

    outputs_dir = os.path.join("outputs", model_name, str(model_version))
    os.makedirs(outputs_dir, exist_ok=True)

    in_picture_path = input(Y + "🖼  Model's Picture: " + W).strip()
    picture_path = os.path.join(outputs_dir, "model_picture.jpg")

    if os.path.exists(in_picture_path):
        img = cv.imread(in_picture_path)
        if img is not None:
            cv.imwrite(picture_path, img)
            print(G + f"\n✅ Picture saved → {picture_path}")
        else:
            print(R + f"\n❌ Error: Could not read image from '{in_picture_path}'")
    else:
        print(R + f"\n❌ Error: Image path '{in_picture_path}' does not exist.")
        retry_path = input(f"{Y}📷 Please enter a valid picture path: ").strip()
        if os.path.exists(retry_path):
            cv.imwrite(picture_path, cv.imread(retry_path))
            print(G + f"\n✅ Picture saved → {picture_path}")

    in_videos_path = input(Y + "🎥 Flight Videos (comma separated): " + W).strip()
    videos_dir = os.path.join(outputs_dir, "Flight Videos")
    os.makedirs(videos_dir, exist_ok=True)

    video_paths = (
        [v.strip() for v in in_videos_path.split(",")] if in_videos_path else []
    )

    for video_path in video_paths:
        if os.path.exists(video_path):
            shutil.copy(video_path, videos_dir)
            print(G + f"✅ Video copied: {video_path} → {videos_dir}")
        else:
            print(f"{R}❌ Video not found: {video_path}")
            return None

    preview_mode = True
    design_notes = input(Y + "📝 Design Notes: " + W).strip()

    metadata_path = os.path.join(outputs_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Version: {model_version}\n")
        f.write(f"Design Notes: {design_notes}\n")
    print(f"{Fore.CYAN}{Style.BRIGHT}📓 Metadata Saved to {metadata_path}")

    for video_path in video_paths:
        distance_px = 0
        trajectory_points = []
        frame_count = 0
        first_detected = None
        last_detected = None

        video_n = os.path.basename(video_path)[:-4]
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(R + f"❌ Error: Could not open video '{video_path}'")
            continue

        fps = cap.get(cv.CAP_PROP_FPS) if cap.get(cv.CAP_PROP_FPS) > 0 else 30

        while True:
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
                if area > 500:
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

        if trajectory_points:
            distance_px = max(x for x, y in trajectory_points) - min(
                x for x, y in trajectory_points
            )
        else:
            distance_px = 0

        if first_detected is not None and last_detected is not None:
            airtime_frames = last_detected - first_detected + 1
            airtime = airtime_frames / fps
        else:
            airtime = 0

        speed = distance_px / airtime if airtime > 0 else 0

        print(f"{C}📝 Enter stability score for {model_name} (1-10): ", end="")
        stability_score = input().strip()

        metrics_path_f = os.path.join(
            outputs_dir, "Flight Metrics", f"{video_n}_metrics.csv"
        )
        os.makedirs(os.path.dirname(metrics_path_f), exist_ok=True)
        with open(metrics_path_f, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Distance_px", "Airtime_s", "Speed_px_per_s", "Stability"])
            writer.writerow(
                [
                    f"{distance_px:.2f}",
                    f"{airtime:.2f}",
                    f"{speed:.2f}",
                    stability_score,
                ]
            )
        print(f"{B}📐 Metrics saved to {metrics_path_f}\n")

    tot_distance = tot_airtime = tot_speed = tot_stability = 0
    metrics_path = os.path.join(outputs_dir, "Flight Metrics")
    mfile_count = 0

    if os.path.exists(metrics_path):
        for filename in os.listdir(metrics_path):
            if filename.endswith("_metrics.csv"):
                filepath = os.path.join(metrics_path, filename)
                with open(filepath, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tot_distance += float(row["Distance_px"])
                        tot_airtime += float(row["Airtime_s"])
                        tot_speed += float(row["Speed_px_per_s"])
                        tot_stability += float(row["Stability"])
                        mfile_count += 1

    avg_distance = tot_distance / mfile_count if mfile_count > 0 else 0
    avg_airtime = tot_airtime / mfile_count if mfile_count > 0 else 0
    avg_speed = tot_speed / mfile_count if mfile_count > 0 else 0
    avg_stability = tot_stability / mfile_count if mfile_count > 0 else 0

    avg_metrics_path = os.path.join(outputs_dir, "avg_metrics.csv")
    with open(avg_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Distance_px", "Airtime_s", "Speed_px_per_s", "Stability"])
        writer.writerow(
            [
                f"{avg_distance:.2f}",
                f"{avg_airtime:.2f}",
                f"{avg_speed:.2f}",
                f"{avg_stability:.2f}",
            ]
        )
    print(f"{G}💾 Average metrics of {model_name} saved to {avg_metrics_path}")

    coordinates_path = os.path.join(outputs_dir, "Flight Coordinates")
    avg_coordinates_path = os.path.join(outputs_dir, "avg_coordinates.csv")
    all_trajectories = []

    if os.path.exists(coordinates_path):
        for filename in sorted(os.listdir(coordinates_path)):
            if filename.endswith("_coordinates.csv"):
                filepath = os.path.join(coordinates_path, filename)
                points = []
                with open(filepath, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        points.append((float(row["X"]), float(row["Y"])))
                if points:
                    all_trajectories.append(points)

    avg_points = []
    if all_trajectories:
        min_length = min(len(traj) for traj in all_trajectories)
        num_videos = len(all_trajectories)
        for i in range(min_length):
            avg_x = sum(traj[i][0] for traj in all_trajectories) / num_videos
            avg_y = sum(traj[i][1] for traj in all_trajectories) / num_videos
            avg_points.append((avg_x, avg_y))

    with open(avg_coordinates_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        for x, y in avg_points:
            writer.writerow([f"{x:.2f}", f"{y:.2f}"])
    print(f"{G}💾 Average Coordinates of {model_name} saved to {avg_coordinates_path}")

    plt.figure(figsize=(8, 6))
    if os.path.exists(coordinates_path):
        for filename in os.listdir(coordinates_path):
            if filename.endswith("_coordinates.csv"):
                filepath = os.path.join(coordinates_path, filename)
                x_coords, y_coords = [], []
                with open(filepath, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        x_coords.append(float(row["X"]))
                        y_coords.append(float(row["Y"]))
                plt.plot(x_coords, y_coords, color="cyan")

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
            f.write(f'# ✈️ {model_name} v{model_version} Engineering Flight Report\n\n')
            f.write(f'Generated for [Your Research Team] on {datetime.datetime.now().strftime("%B %d, %Y, %I:%M %p +06")}\n\n')
            f.write('Evaluated for aerodynamic performance and design optimization.\n\n')

            # Overview
            f.write('## Overview\n')
            f.write(f'- **Model**: {model_name}\n')
            f.write(f'- **Version**: {model_version}\n')
            f.write(f'- **Design Notes**: {design_notes}\n')
            f.write(f'- **Created**: {datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p +06")}\n\n')

            # Visuals
            f.write('## Visuals\n')
            if os.path.exists(os.path.join(outputs_dir, 'model_picture.jpg')):
                f.write('![Model Design](model_picture.jpg)\n')
            else:
                f.write('Model image not found—add to document design!\n')
            if os.path.exists(os.path.join(outputs_dir, 'trajectory_graph.png')):
                f.write('![Flight Trajectory](trajectory_graph.png)\n')
            else:
                f.write('Trajectory graph not generated!\n')
            f.write('\n')

            # Average Metrics
            avg_metrics = {}
            metrics_path = os.path.join(outputs_dir, 'avg_metrics.csv')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as mf:
                    reader = csv.DictReader(mf)
                    for row in reader:
                        avg_metrics = {
                            'Distance': float(row['Distance_px']),
                            'Airtime': float(row['Airtime_s']),
                            'Speed': float(row['Speed_px_per_s']),
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
                is_best_distance = True
                if os.path.exists(model_dir):
                    for version in os.listdir(model_dir):
                        if version != str(model_version) and os.path.isdir(os.path.join(model_dir, version)):
                            v_metrics_path = os.path.join(model_dir, version, 'avg_metrics.csv')
                            if os.path.exists(v_metrics_path):
                                with open(v_metrics_path, 'r', encoding='utf-8') as vf:
                                    reader = csv.DictReader(vf)
                                    for row in reader:
                                        if float(row['Distance_px']) > max_distance:
                                            is_best_distance = False
                                            break
                if is_best_distance and avg_metrics:
                    f.write('\n**🏆 Achievement**: Best distance in test series!\n')
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
                        with open(os.path.join(flight_metrics_dir, video_file), 'r', encoding='utf-8') as vf:
                            reader = csv.DictReader(vf)
                            for row in reader:
                                video_metrics.append({
                                    'Video Name': video_name,
                                    'Distance': float(row['Distance_px']),
                                    'Airtime': float(row['Airtime_s']),
                                    'Speed': float(row['Speed_px_per_s']),
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
                            vm['Performance Note'] = 'Most stable'
                    # Calculate standard deviations
                    distances = [vm['Distance'] for vm in video_metrics]
                    stabilities = [vm['Stability'] for vm in video_metrics]
                    std_distance = np.std(distances) if len(distances) > 1 else 0.0
                    std_stability = np.std(stabilities) if len(stabilities) > 1 else 0.0
                    f.write('## Per-Video Analysis\n')
                    f.write('| Video Name | Distance (px) | Airtime (s) | Speed (px/s) | Stability | Performance Note |\n')
                    f.write('|--|--|--|--|--|--|\n')
                    for vm in video_metrics:
                        f.write(f"| {vm['Video Name']} | {vm['Distance']:.2f} | {vm['Airtime']:.2f} | {vm['Speed']:.2f} | {vm['Stability']:.2f} | {vm['Performance Note']} |\n")
                    f.write(f'\n**Stats**: Standard Deviation: Distance: {std_distance:.2f}px, Stability: {std_stability:.2f} ({"highly consistent" if std_distance < 10 else "variable"})\n')
            else:
                f.write('## Per-Video Analysis\n')
                f.write('No flight data recorded! Conduct tests, engineer! ✈️\n')
            f.write('\n')

            # Research Insights
            f.write('## Research Insights\n')
            if avg_metrics:
                # Compare to previous versions
                prev_distance = None
                if os.path.exists(model_dir):
                    for version in os.listdir(model_dir):
                        if version != str(model_version) and os.path.isdir(os.path.join(model_dir, version)):
                            v_metrics_path = os.path.join(model_dir, version, 'avg_metrics.csv')
                            if os.path.exists(v_metrics_path):
                                with open(v_metrics_path, 'r', encoding='utf-8') as vf:
                                    reader = csv.DictReader(vf)
                                    for row in reader:
                                        prev_distance = float(row['Distance_px'])
                                        break
                                break
                if prev_distance is not None:
                    percent_change = ((avg_metrics['Distance'] - prev_distance) / prev_distance * 100) if prev_distance != 0 else 0
                    f.write(f"- **Distance**: {'Up' if percent_change > 0 else 'Down'} {abs(percent_change):.1f}% from v{version} ({prev_distance:.2f}px), indicating {'improved' if percent_change > 0 else 'reduced'} glide efficiency\n")
                f.write(f"- **Stability**: {avg_metrics['Stability']:.1f}/10, {'excellent' if avg_metrics['Stability'] > 8 else 'good' if avg_metrics['Stability'] > 6 else 'needs work'} for controlled flights\n")
                airtime = avg_metrics['Airtime']
                recommendation = f"Airtime {airtime:.1f}s < 4s target—try lighter paper or sharper wings" if airtime < 4 else "Solid airtime—test with varied wind conditions"
                f.write(f"- **Recommendation**: {recommendation}\n")
                f.write(f"- **Note**: {'Consistent' if std_distance < 10 else 'Variable'} metrics suggest {'reliability' if std_distance < 10 else 'need for design tweaks'} for further testing\n")
            else:
                f.write('- No insights available—log flight data to analyze! ✈️\n')
            f.write('\n')

            # Social Prompt
            f.write('**Share your findings on X with #LucidraftDeltaX to discuss with the research community! 🚀**\n')

        print(f"{G}📄 Report saved to → {report_path}")
    except Exception as e:
        print(f"{R}❌ Turbulence! Failed to generate report: {e}. Check files and try again! ✈️")


    return model_name, model_version

# =================== Update Model ======================
def get_prev_v(model_name):
    model_path = os.path.join('outputs', model_name)
    if not os.path.exists(model_path):
        return None
    vs = []
    for v in os.listdir(model_path):
        vs.append(float(v))
    
    return max(vs)

def update_model():
    model_name = input(f'\n{Y}Please enter the name of the model you want to update: ')
    prev_version = get_prev_v(model_name)
    new_version = round(prev_version + 0.1, 1)
    if not prev_version:
        print(f'{R}⚠ No models found! Please first create one, then update if needed')
    else:
        log_new_model(model_name, new_version)
        compare(model_name, prev_version, model_name, new_version)

def compare(model_name1=None, model1_v=None, model_name2=None, model2_v=None):

    if not model_name1 and not model1_v and not model_name2 and not model2_v:
        model_name1, model1_v = input(f"{Y} Please enter the first model's name and version(model's name <space> version): ").split(' ')
        model_name2, model2_v = input(f"{Y} Please enter the second model's name and version(model's name <space> version): ").split(' ')

    with open(f'outputs/{model_name1}/{model1_v}/avg_metrics.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model1_distance = float(row["Distance_px"])
            model1_speed = float(row['Speed_px_per_s'])
            model1_stability = float(row["Stability"])
    
    with open(f'outputs/{model_name2}/{model2_v}/avg_metrics.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model2_distance = float(row["Distance_px"])
            model2_speed = float(row["Speed_px_per_s"])
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



# ================= Model view ========================
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
                    with open(version_path, 'r') as f:
                        reader = csv.DictReader(f)
                        
                        for row in reader:
                            # Append a new dictionary to the list for this model
                            models[model_name].append({
                                'version': str(version),
                                'distance': str(row['Distance_px']),
                                'speed': str(row['Speed_px_per_s']),
                                'stability': str(row['Stability'])
                            })
                            
    print(create_combined_table(models))


def pause():
    input(W + "\nPress Enter to return to the homepage...")


def draw_rectangle(img, x, y, w, h):
    glow_color = (0, 255, 255)
    thickness = 2
    length = 20
    corners = [
        ((x, y), (x + length, y), (x, y + length)),
        ((x + w, y), (x + w - length, y), (x + w, y + length)),
        ((x, y + h), (x + length, y + h), (x, y + h - length)),
        ((x + w, y + h), (x + w - length, y + h), (x + w, y + h - length)),
    ]
    for pt1, pt2, pt3 in corners:
        cv.line(img, pt1, pt2, glow_color, thickness)
        cv.line(img, pt1, pt3, glow_color, thickness)
    cv.line(img, (x, y), (x + w, y + h), glow_color, thickness)
    cv.line(img, (x + w, y), (x, y + h), glow_color, thickness)
    center = (x + w // 2, y + h // 2)
    for i in range(6, 0, -1):
        alpha = 0.1 * i
        overlay = img.copy()
        cv.circle(overlay, center, i * 3, glow_color, -1)
        cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv.circle(img, center, 4, glow_color, -1)


def draw_grid(img, spacing=60, color=(0, 255, 255), thickness=1, alpha=0.1):
    overlay = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, spacing):
        cv.line(overlay, (x, 0), (x, h), color, thickness)
    for y in range(0, h, spacing):
        cv.line(overlay, (0, y), (w, y), color, thickness)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


if __name__ == "__main__":
    while True:
        choice = homepage()
        if choice == "6":
            print(
                B + "\n------------------ Goodbye! Fly high! ✈️ --------------------\n"
            )
            break
        elif choice == "1":
            log_new_model()
            pause()
        elif choice == "2":
            update_model()
        elif choice == "3":
            view_models()
            pause()
        elif choice == "4":
            compare()
            pause()
        elif choice == "5":
            print('Hold tight, coming soon!')
            pause()
        else:
            pause()
