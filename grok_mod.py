import os
import cv2 as cv
from colorama import init, Fore, Style
import shutil
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm  # Added for progress bars
from PIL import Image, ImageDraw, ImageFont  # Added for summary image
import glob  # For leaderboard

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
    print(B + " [5]" + W + " 📊 Generate Overall Report")
    print(B + " [6]" + W + " ❌ Exit")
    print(C + "─" * 50)
    return input(Y + "Select an option (1-6): " + W).strip()

def create_summary_image(outputs_dir, model_name, model_version, avg_metrics):
    # Create a summary image with model picture, trajectory graph, and stats
    summary_img = Image.new('RGB', (800, 600), color='black')
    draw = ImageDraw.Draw(summary_img)
    
    # Load fonts (use default if none available)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Add title
    draw.text((20, 20), f"{model_name} v{model_version}", fill='cyan', font=title_font)

    # Add model picture
    picture_path = os.path.join(outputs_dir, "model_picture.jpg")
    if os.path.exists(picture_path):
        model_img = Image.open(picture_path).resize((200, 200))
        summary_img.paste(model_img, (20, 70))

    # Add trajectory graph
    trajectory_graph_path = os.path.join(outputs_dir, "trajectory_graph.png")
    if os.path.exists(trajectory_graph_path):
        traj_img = Image.open(trajectory_graph_path).resize((300, 300))
        summary_img.paste(traj_img, (250, 70))

    # Add stats
    stats_text = (
        f"Distance: {avg_metrics['Distance_px']:.2f} px\n"
        f"Speed: {avg_metrics['Speed_px_per_s']:.2f} px/s\n"
        f"Stability: {avg_metrics['Stability']:.2f}"
    )
    draw.text((20, 300), stats_text, fill='yellow', font=font)

    # Save summary image
    summary_path = os.path.join(outputs_dir, "summary.png")
    summary_img.save(summary_path)
    print(f"{G}📸 Summary image saved to → {summary_path}")

def view_models():
    print(C + "\n--- Leaderboard: Top Paper Planes ---\n")
    models = []
    outputs_base = "outputs"
    if not os.path.exists(outputs_base):
        print(R + "❌ No models found! Create some planes first! ✈️")
        return

    for model_name in os.listdir(outputs_base):
        model_path = os.path.join(outputs_base, model_name)
        if os.path.isdir(model_path):
            for version in os.listdir(model_path):
                try:
                    version = float(version)
                    avg_metrics_path = os.path.join(model_path, str(version), "avg_metrics.csv")
                    if os.path.exists(avg_metrics_path):
                        with open(avg_metrics_path, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                models.append({
                                    'name': model_name,
                                    'version': version,
                                    'distance': float(row['Distance_px']),
                                    'speed': float(row['Speed_px_per_s']),
                                    'stability': float(row['Stability'])
                                })
                except ValueError:
                    continue

    if not models:
        print(R + "❌ No valid models with metrics found!")
        return

    # Sort by distance (descending)
    models.sort(key=lambda x: x['distance'], reverse=True)
    
    print(B + "🏆 Leaderboard (Ranked by Distance)")
    print(C + "─" * 50)
    for i, model in enumerate(models, 1):
        print(
            f"{Y}[{i}] {model['name']} v{model['version']}: "
            f"Distance: {model['distance']:.2f}px, "
            f"Speed: {model['speed']:.2f}px/s, "
            f"Stability: {model['stability']:.2f}"
        )
    print(C + "─" * 50)

def log_new_model(model_name=None, model_version=1.0):
    need = "Add" if not model_name else 'Update'
    print(C + f"\n--- {need} New Model ---\n")
    if not model_name:
        model_name = input(Y + "✈  Model Name: " + W).strip()

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
            retry_path = input(f"{Y}📷 Please enter a valid picture path or press Enter to skip: ").strip()
            if retry_path and os.path.exists(retry_path):
                cv.imwrite(picture_path, cv.imread(retry_path))
                print(G + f"\n✅ Picture saved → {picture_path}")
    else:
        print(R + f"\n❌ Error: Image path '{in_picture_path}' does not exist.")
        retry_path = input(f"{Y}📷 Please enter a valid picture path or press Enter to skip: ").strip()
        if retry_path and os.path.exists(retry_path):
            cv.imwrite(picture_path, cv.imread(retry_path))
            print(G + f"\n✅ Picture saved → {picture_path}")

    in_videos_path = input(Y + "🎥 Flight Videos (comma separated): " + W).strip()
    videos_dir = os.path.join(outputs_dir, "Flight Videos")
    os.makedirs(videos_dir, exist_ok=True)

    video_paths = [v.strip() for v in in_videos_path.split(",")] if in_videos_path else []

    for video_path in video_paths:
        if os.path.exists(video_path):
            shutil.copy(video_path, videos_dir)
            print(G + f"✅ Video copied: {video_path} → {videos_dir}")
        else:
            print(f"{R}❌ Video not found: {video_path}. Try again, champ! ✈️")
            return None

    preview_mode = True
    design_notes = input(Y + "📝 Design Notes: " + W).strip()

    metadata_path = os.path.join(outputs_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Version: {model_version}\n")
        f.write(f"Design Notes: {design_notes}\n")
    print(f"{C}📓 Metadata Saved to {metadata_path}")

    for video_path in tqdm(video_paths, desc=f"{Y}Processing Videos for {model_name}", unit="video"):
        distance_px = 0
        trajectory_points = []
        frame_count = 0
        first_detected = None
        last_detected = None

        video_n = os.path.basename(video_path)[:-4]
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(R + f"❌ Error: Could not open video '{video_path}'. Crashed before takeoff! 😅")
            continue

        fps = cap.get(cv.CAP_PROP_FPS) if cap.get(cv.CAP_PROP_FPS) > 0 else 30

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc=f"{Y}Analyzing {video_n}", unit="frame") as pbar:
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

                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                detected_this_frame = False

                for contour in contours:
                    area = cv.contourArea(contour)
                    if area > 500:
                        detected_this_frame = True
                        x, y, w, h = cv.boundingRect(contour)
                        center = (x + w // 2, y + h // 2)
                        trajectory_points.append(center)

                        cv.putText(frame, "Plane", (x, y - 10), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
                        draw_rectangle(frame, x, y, w, h)
                        if len(trajectory_points) > 1:
                            cv.polylines(frame, [np.array(trajectory_points)], False, (0, 255, 0), 2)
                        draw_grid(frame)
                        cv.putText(frame, f"{center}", (frame.shape[1] - 90, frame.shape[0] - 10), cv.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)

                if detected_this_frame:
                    if first_detected is None:
                        first_detected = frame_count
                    last_detected = frame_count

                if preview_mode:
                    cv.imshow(f"Video for {model_name}", frame)
                    if cv.waitKey(20) & 0xFF == ord("q"):
                        break

                pbar.update(1)

        cap.release()
        if preview_mode:
            cv.destroyAllWindows()

        coordinates_path_f = os.path.join(outputs_dir, "Flight Coordinates", f"{video_n}_coordinates.csv")
        os.makedirs(os.path.dirname(coordinates_path_f), exist_ok=True)
        with open(coordinates_path_f, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y"])
            for x, y in trajectory_points:
                writer.writerow([x, y])
        print(f"{G}📈 Flight Trajectory Coordinates saved to {coordinates_path_f}")

        if trajectory_points:
            distance_px = max(x for x, y in trajectory_points) - min(x for x, y in trajectory_points)
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

        metrics_path_f = os.path.join(outputs_dir, "Flight Metrics", f"{video_n}_metrics.csv")
        os.makedirs(os.path.dirname(metrics_path_f), exist_ok=True)
        with open(metrics_path_f, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Distance_px", "Airtime_s", "Speed_px_per_s", "Stability"])
            writer.writerow([f"{distance_px:.2f}", f"{airtime:.2f}", f"{speed:.2f}", stability_score])
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

    avg_metrics = {
        'Distance_px': tot_distance / mfile_count if mfile_count > 0 else 0,
        'Airtime_s': tot_airtime / mfile_count if mfile_count > 0 else 0,
        'Speed_px_per_s': tot_speed / mfile_count if mfile_count > 0 else 0,
        'Stability': tot_stability / mfile_count if mfile_count > 0 else 0
    }

    avg_metrics_path = os.path.join(outputs_dir, "avg_metrics.csv")
    with open(avg_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Distance_px", "Airtime_s", "Speed_px_per_s", "Stability"])
        writer.writerow([f"{avg_metrics['Distance_px']:.2f}", f"{avg_metrics['Airtime_s']:.2f}", f"{avg_metrics['Speed_px_per_s']:.2f}", f"{avg_metrics['Stability']:.2f}"])
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
        plt.plot(avg_x_coords, avg_y_coords, color="red", linewidth=2, label="Average Trajectory")

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

    # Create shareable summary image
    create_summary_image(outputs_dir, model_name, model_version, avg_metrics)

    return model_name, model_version

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

def update_model():
    model_name = input(f'\n{Y}Please enter the name of the model you want to update: ').strip()
    prev_version = get_prev_v(model_name)
    if not prev_version:
        print(f'{R}⚠ No models found! Please first create one, then update if needed')
    else:
        log_new_model(model_name, round(prev_version + 0.1, 1))
    pause()

def compare(model_name1=None, model1_v=None, model_name2=None, model2_v=None):
    if not model_name1 and not model1_v and not model_name2 and not model2_v:
        try:
            model_name1, model1_v = input(f"{Y}Please enter the first model's name and version (name version): ").strip().split()
            model_name2, model2_v = input(f"{Y}Please enter the second model's name and version (name version): ").strip().split()
            model1_v = float(model1_v)
            model2_v = float(model2_v)
        except ValueError:
            print(R + "❌ Invalid input! Format: 'name version' (e.g., 'Falcon 1.0'). Try again! 😜")
            return

    metrics_path1 = f'outputs/{model_name1}/{model1_v}/avg_metrics.csv'
    metrics_path2 = f'outputs/{model_name2}/{model2_v}/avg_metrics.csv'

    if not os.path.exists(metrics_path1) or not os.path.exists(metrics_path2):
        print(R + "❌ One or both models not found! Check your spelling or create them first! ✈️")
        return

    with open(metrics_path1, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model1_distance = float(row["Distance_px"])
            model1_speed = float(row["Speed_px_per_s"])
            model1_stability = float(row["Stability"])

    with open(metrics_path2, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model2_distance = float(row["Distance_px"])
            model2_speed = float(row["Speed_px_per_s"])
            model2_stability = float(row["Stability"])

    print(f"\n{B}⚖ Comparing {model_name2}_v{model2_v} to {model_name1}_v{model1_v}\n")
    distance_gap = model2_distance - model1_distance
    stability_gap = model2_stability - model1_stability
    speed_gap = model2_speed - model1_speed

    def get_sign(v):
        return '+' if v > 0 else ''

    print(f'{Y}Distance: {get_sign(distance_gap)}{distance_gap:.2f} px')
    print(f'{Y}Stability: {get_sign(stability_gap)}{stability_gap:.2f}')
    print(f'{Y}Speed: {get_sign(speed_gap)}{speed_gap:.2f} px/s')

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
            print(B + "\n------------------ Goodbye! Fly high! ✈️ --------------------\n")
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
            print(M + "\n📊 Generating overall report (Coming Soon) 📈\n")
            pause()
        else:
            print(R + "\n❌ Invalid choice! Pick 1-6 or your plane’s grounded! 😎\n")
            pause()