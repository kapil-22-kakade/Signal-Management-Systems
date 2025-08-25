import pygame
import cv2
import torch
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

# Initialize Pygame
pygame.init()

# Screen setup
screen_width, screen_height = 1000, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Smart Traffic Signal System - Advanced")

# Colors
ROAD_COLOR = (70, 70, 70)
LANE_COLOR = (200, 200, 200)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 215, 0)
GREEN = (0, 255, 0)
GRAY = (50, 50, 50)
BLACK = (0, 0, 0)

# Fonts
font = pygame.font.Font(None, 24)

# YOLO model for vehicle detection
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit()

# Function to calculate signal timers proportionally
def calculate_timers(car_counts):
    total_cars = sum(car_counts)
    if total_cars == 0:
        return [1] * 4  # Default timing when no cars (1 minute)
    return [max(1, int((count / total_cars) * 60)) for count in car_counts]  # Longer green light duration

# Count vehicles using YOLO
def count_vehicles(frame):
    # Convert BGR to RGB for YOLO input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)  # Pass the RGB frame
    vehicle_classes = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
    detections = results.xyxy[0].cpu().numpy()
    vehicle_count = sum(1 for det in detections if int(det[5]) in vehicle_classes)
    return vehicle_count

# Capture vehicle counts from file dialog input
def capture_from_file_dialog():
    Tk().withdraw()  # Hide the root Tk window
    # Let the user select four images
    image_paths = askopenfilenames(title="Select Four Image Files",
                                   filetypes=[("Image Files", ".jpg;.png;.jpeg;.bmp")])
    
    if len(image_paths) != 4:
        print("Error: Please select exactly four images.")
        return [0, 0, 0, 0]  # Default vehicle counts

    vehicle_counts = []
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image at {image_path}.")
            vehicle_counts.append(0)  # Default to zero for invalid images
        else:
            vehicle_counts.append(count_vehicles(image))
            print(f"Image {i+1}: Detected {vehicle_counts[-1]} vehicles.")

    return vehicle_counts

# Capture vehicle counts for four directions using webcam (one-by-one)
def capture_one_by_one_images():
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    directions = ["North", "East", "South", "West"]
    vehicle_counts = []

    print("Using webcam to capture images one by one.")
    for direction in directions:
        print(f"Position the camera for the {direction} side and press 's' to capture.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame for {direction} side. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                return [0, 0, 0, 0]  # Default vehicle counts

            # Display the webcam feed
            cv2.imshow(f"Capture {direction} Side", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):  # 's' to save the current frame
                vehicle_counts.append(count_vehicles(frame))
                print(f"Captured {direction} side. Detected {vehicle_counts[-1]} vehicles.")
                break
            elif key == ord('q'):  # 'q' to quit
                cap.release()
                cv2.destroyAllWindows()
                return [0, 0, 0, 0]

    cap.release()
    cv2.destroyAllWindows()
    return vehicle_counts

# Draw road intersection with markings
def draw_intersection():
    pygame.draw.rect(screen, ROAD_COLOR, (400, 0, 200, 800))
    pygame.draw.rect(screen, ROAD_COLOR, (0, 400, 1000, 200))
    pygame.draw.line(screen, LANE_COLOR, (500, 0), (500, 400), 5)
    pygame.draw.line(screen, LANE_COLOR, (500, 600), (500, 800), 5)
    pygame.draw.line(screen, LANE_COLOR, (0, 500), (400, 500), 5)
    pygame.draw.line(screen, LANE_COLOR, (600, 500), (1000, 500), 5)
    pygame.draw.line(screen, WHITE, (450, 400), (550, 400), 3)
    pygame.draw.line(screen, WHITE, (450, 600), (550, 600), 3)
    pygame.draw.line(screen, WHITE, (400, 450), (400, 550), 3)
    pygame.draw.line(screen, WHITE, (600, 450), (600, 550), 3)

# Draw traffic signals
def draw_signals(active_lane, car_counts, timers, timer_remaining):
    signals = [
        (500, 300, active_lane, car_counts[0], timers[0], "North"),
        (600, 500, active_lane, car_counts[1], timers[1], "East"),
        (500, 600, active_lane, car_counts[2], timers[2], "South"),
        (400, 500, active_lane, car_counts[3], timers[3], "West")
    ]

    for x, y, active_lane, count, total_time, direction in signals:
        pygame.draw.rect(screen, GRAY, (x - 15, y - 80, 30, 160))
        if timer_remaining <= 3 and active_lane == signals.index((x, y, active_lane, count, total_time, direction)):
            color = "yellow"
        elif active_lane == signals.index((x, y, active_lane, count, total_time, direction)):
            color = "green"
        else:
            color = "red"

        pygame.draw.circle(screen, RED if color == "red" else GRAY, (x, y - 60), 10)
        pygame.draw.circle(screen, YELLOW if color == "yellow" else GRAY, (x, y), 10)
        pygame.draw.circle(screen, GREEN if color == "green" else GRAY, (x, y + 60), 10)

        car_text = font.render(f"{direction}: {count} cars", True, WHITE)
        screen.blit(car_text, (x - 70, y + 90))
        if color == "green":
            time_text = font.render(f"Open for: {timer_remaining:.1f} min", True, WHITE)
        else:
            time_text = font.render(f"Time: {total_time} min", True, WHITE)
        screen.blit(time_text, (x - 70, y + 120))

# Main program
def main():
    clock = pygame.time.Clock()
    running = True
    cycles_completed = 0

    print("Choose input type:\n1. Webcam for one-by-one images\n2. File dialog to select images")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        car_counts = capture_one_by_one_images()
    elif choice == "2":
        car_counts = capture_from_file_dialog()
    else:
        print("Invalid choice.")
        return

    if car_counts == [0, 0, 0, 0]:
        print("Error: No valid input provided. Exiting.")
        return

    timers = calculate_timers(car_counts)
    active_lane = 0
    timer_remaining = timers[active_lane]

    while running:
        screen.fill(BLACK)
        draw_intersection()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        timer_remaining -= 1 / 30
        if timer_remaining <= 0:
            active_lane = (active_lane + 1) % 4
            timer_remaining = timers[active_lane]

            if active_lane == 0:
                cycles_completed += 1
                if cycles_completed >= 1:
                    running = False

        draw_signals(active_lane, car_counts, timers, timer_remaining)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if _name_ == "_main_":
    main()