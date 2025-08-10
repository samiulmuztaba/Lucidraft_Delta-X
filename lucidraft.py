import os
import cv2 as cv
from colorama import init, Fore, Style
import shutil
import numpy as np

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
{C}   ║{B} ✈  L U C I D R A F T   D E L T A - X   1 . 0 {C}║
{C}   ║{M} Advanced Paper Plane Engineering Terminal    {C}║
{C}   ╚══════════════════════════════════════════════╝
"""
    print(logo)

def homepage():
    banner()
    print(B + " [1]" + W + " ➕ Add New Model")
    print(B + " [2]" + W + " 📂 View Models")
    print(B + " [3]" + W + " 🔍 Compare Models")
    print(B + " [4]" + W + " 📊 Generate Reports")
    print(B + " [5]" + W + " ❌ Exit")
    print(C + "─" * 50)
    return input(Y + "Select an option (1-5): " + W).strip()

def log_new_model():
    print(C + "\n--- Add New Model ---\n")
    model_name = input(Y + "✈  Model Name: " + W).strip()
    model_version = input(Y + "🆚 Version: " + W).strip()
    in_picture_path = input(Y + "🖼  Model's Picture: " + W).strip()
    in_videos_path = input(Y + '🎥 Flight Videos (comma separated): ' + W).strip()
    design_notes = input(Y + "📝 Design Notes: " + W).strip()
    
    outputs_dir = f"outputs/{model_name}_{model_version}"
    os.makedirs(outputs_dir, exist_ok=True)

    # Copy the picture
    picture_path = f"{outputs_dir}/model_picture.jpg"
    if os.path.exists(in_picture_path):
        img = cv.imread(in_picture_path)
        if img is not None:
            cv.imwrite(picture_path, img)
            print(G + f"\n✅ Picture saved → {picture_path}")
        else:
            print(R + f"\n❌ Error: Could not read image from '{in_picture_path}'")
    else:
        print(R + f"\n❌ Error: Image path '{in_picture_path}' does not exist.")

    # Copy videos
    videos_dir = f"{outputs_dir}/videos"
    os.makedirs(videos_dir, exist_ok=True)
    video_paths = []
    if in_videos_path:
        videos = [v.strip() for v in in_videos_path.split(',')]
        for video in videos:
            if os.path.exists(video):
                dest_path = os.path.join(videos_dir, os.path.basename(video))
                shutil.copy(video, dest_path)
                video_paths.append(dest_path)
                print(G + f"✅ Video copied → {dest_path}")
            else:
                print(R + f"❌ Error: Video '{video}' does not exist.")
    else:
        print(Y + "No videos provided.")

    ref_pixels = 0
    ref_m = 0
    pm = ref_pixels / ref_m if ref_m != 0 else 1  # Avoid division by zero
    distance_px = 0
    distance = distance_px / pm if pm != 0 else 1  # Avoid division by zero
    airtime = 0
    frame_count = 0
    trajectory_points = []
    for video_path in video_paths:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(R + f"❌ Error: Could not open video '{video_path}'")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.resize(frame, (960, 540))
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            lower, upper = np.array([35, 50, 102]), np.array([179, 255, 255])
            mask = cv.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv.boundingRect(contour)
                    center = (x + w // 2, y + h // 2)
                    trajectory_points.append(center)  # Store (x, y)
                    cv.putText(
                        frame, "Plane", (x, y - 10), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2
                    )
                    draw_rectangle(frame, x, y, w, h)
                    if len(trajectory_points) > 1:
                        cv.polylines(
                            frame, [np.array(trajectory_points)], False, (0, 255, 0), 2
                        )
                    draw_grid(frame)
                    cv.putText(
                        frame, f"{center}", (frame.shape[1] - 90, frame.shape[0] - 10),
                        cv.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1
                    )


            # Optionally display frame (for debugging)
            cv.imshow(f'Video for {model_name}', frame)
            if cv.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()  

    cv.destroyAllWindows()  # Close any OpenCV windows if used

    # Store metadata
    md = f"{outputs_dir}/metadata.txt"
    with open(md, "w") as meta_file:
        meta_file.write(f"Name: {model_name}\n")
        meta_file.write(f"Version: {model_version}\n")
        meta_file.write(f"Design Notes: {design_notes}\n")
        meta_file.write(f"Path to Picture: {picture_path}\n")
    print(G + f"🗂 Metadata saved → {md}\n")

    # Store flight trajectory
    coordinates_path = f'{outputs_dir}/flight_coordinates.csv'
    with open(coordinates_path, "w") as f:
                f.write("Frame,X,Y\n")
                for frame_num, x, y in trajectory_points:
                    f.write(f"{frame_num},{x},{y}\n")
                    distance_px += np.sqrt(
                        (x - trajectory_points[0][1]) ** 2
                        + (y - trajectory_points[0][2]) ** 2
                    )
                    airtime += 1 / 30  # Assuming 30 FPS, adjust as necessary

    print(f'{G} 📈 Flight Trajectory Coordinates saved to {coordinates_path}')

    # Make the trajectory graph <--------------------------------HERE YOU ARE ATTENTION PLEASEEEEE

    # Store flight metrics
    metrics_path = f'{outputs_dir}/flight_metrics.csv'
    print(f"{Fore.CYAN}{Style.BRIGHT}  📝 Enter stability score for {model_name} (1-10): ", end='')
    stabilty_score = input().strip()


    with open(metrics_path, "w") as f:
            f.write("Distance,Airtime,Speed,Stability\n")
            if airtime > 0:
                speed = distance_px / airtime
                f.write(f"{distance_px:.2f},{airtime:.2f},{speed:.2f},{stabilty_score}\n")
            else:
                f.write("0,0,0,0\n")
    print(f"{G} 🎛 Metrics data saved to {metrics_path}")



def pause():
    input(W + "\nPress Enter to return to the homepage...")

# ==== Utility Functions =====
def draw_rectangle(img, x, y, w, h):
    glow_color = (0, 255, 255)
    thickness = 2
    length = 20

    # Main rectangle
    corners = [
        ((x, y), (x + length, y), (x, y + length)),
        ((x + w, y), (x + w - length, y), (x + w, y + length)),
        ((x, y + h), (x + length, y + h), (x, y + h - length)),
        ((x + w, y + h), (x + w - length, y + h), (x + w, y + h - length)),
    ]
    for pt1, pt2, pt3 in corners:
        cv.line(img, pt1, pt2, glow_color, thickness)
        cv.line(img, pt1, pt3, glow_color, thickness)

    # Cross
    cv.line(img, (x, y), (x + w, y + h), glow_color, thickness)
    cv.line(img, (x + w, y), (x, y + h), glow_color, thickness)
    center = (x + w // 2, y + h // 2)

    # Circle
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
        if choice == '5':
            print(R + "\nGoodbye! Fly high! ✈️\n")
            break
        elif choice == '1':
            log_new_model()
            pause()
        elif choice == '2':
            print(M + "\n📂 Viewing models (Coming Soon)\n")
            pause()
        elif choice == '3':
            print(M + "\n🔍 Comparing models (Coming Soon)\n")
            pause()
        elif choice == '4':
            print(M + "\n📊 Generating reports (Coming Soon)\n")
            pause()
        else:
            print(R + "\n❌ Invalid choice. Please enter 1-5.\n")
            pause()