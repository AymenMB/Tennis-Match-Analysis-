import cv2
import numpy as np
import os
import pandas as pd # Import pandas for easier CSV writing

# Global variables to store the annotation for the current image
current_annotation = {}
results_list = []
image_being_displayed = None # To pass the image object to the callback
info_panel = None  # Image to display status and visibility information
display_image = None  # Working copy of the image for display purposes

# Default values for visibility and status
current_visibility = 1  # Default: Ball is clearly visible
current_status = 0      # Default: Normal flight

# State variables for keyboard input
waiting_for_visibility = False
waiting_for_status = False

# CSV delimiter - will be determined based on existing file format
csv_delimiter = ","  # Default to comma

# Function to load existing annotations from CSV
def load_existing_annotations(csv_path):
    global csv_delimiter
    annotations_dict = {}
    if os.path.exists(csv_path):
        try:
            # Try to detect the delimiter
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                if ';' in first_line:
                    csv_delimiter = ';'
                else:
                    csv_delimiter = ','
            
            # Load the existing CSV file with the detected delimiter
            df = pd.read_csv(csv_path, delimiter=csv_delimiter)
            
            # Convert DataFrame to dictionary with filename as key
            for _, row in df.iterrows():
                filename = row['file name']
                annotations_dict[filename] = dict(row)
            
            print(f"Loaded {len(annotations_dict)} existing annotations from {csv_path}")
        except Exception as e:
            print(f"Warning: Error loading existing annotations: {e}")
    
    return annotations_dict

# Function to save a single annotation to CSV
def save_annotation(annotation, csv_path, append=True):
    global csv_delimiter
    
    # Create a DataFrame with the single annotation
    df_annotation = pd.DataFrame([annotation])
    
    # Handle None values gracefully
    df_annotation['file name'] = df_annotation['file name'].astype(str)
    df_annotation['visibility'] = df_annotation['visibility'].astype(int)
    # Replace None with empty string for CSV
    df_annotation['x-coordinate'] = df_annotation['x-coordinate'].apply(lambda x: '' if x is None else str(x))
    df_annotation['y-coordinate'] = df_annotation['y-coordinate'].apply(lambda y: '' if y is None else str(y))
    df_annotation['status'] = df_annotation['status'].astype(int)
    
    # Reorder columns
    df_annotation = df_annotation[['file name', 'visibility', 'x-coordinate', 'y-coordinate', 'status']]
    
    # Check if file exists for appending
    file_exists = os.path.exists(csv_path) and append
    
    # Save to CSV (append or create new)
    if file_exists:
        # Append mode doesn't write headers if file exists
        df_annotation.to_csv(csv_path, mode='a', index=False, header=False, sep=csv_delimiter)
        print(f"Annotation for {annotation['file name']} appended to CSV")
    else:
        # Create new file with headers
        df_annotation.to_csv(csv_path, index=False, sep=csv_delimiter)
        print(f"New CSV file created with annotation for {annotation['file name']}")

# Mouse callback function to capture clicks
def get_coordinates(event, x, y, flags, param):
    global current_annotation, image_being_displayed, info_panel, display_image

    # If the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if info_panel exists
        if info_panel is not None:
            # Adjust y coordinate to account for info_panel height
            info_panel_height = info_panel.shape[0]
            
            # Check if the click is in the image area (below the info panel)
            if y >= info_panel_height:
                # Adjust y coordinate to be relative to the image, not the stacked display
                adjusted_y = y - info_panel_height
                
                # Only capture coordinates if visibility is 1 or 2 (ball is visible)
                if current_visibility > 0:
                    print(f"Clicked coordinates (x, y): ({x}, {adjusted_y})")
                    # Update the current annotation with the adjusted coordinates
                    current_annotation['x-coordinate'] = x
                    current_annotation['y-coordinate'] = adjusted_y
                    
                    # Create a copy of the original image before drawing on it
                    display_image = image_being_displayed.copy()
                    
                    # Draw a small circle at the adjusted position
                    cv2.circle(display_image, (x, adjusted_y), 5, (0, 255, 0), -1)
                    
                    # Update info panel and display
                    update_info_panel()
                    cv2.imshow("Image", np.vstack([info_panel, display_image]))

# Function to update the information panel
def update_info_panel():
    global info_panel, current_visibility, current_status, current_annotation, waiting_for_visibility, waiting_for_status
    
    # Create a black panel for information
    info_panel = np.zeros((100, image_being_displayed.shape[1], 3), dtype=np.uint8)
    
    # Define visibility descriptions
    visibility_desc = {
        0: "Ball is gone (no coordinates needed)",
        1: "Ball is clearly visible",
        2: "Ball is visible but hard to see"
    }
    
    # Define status descriptions
    status_desc = {
        0: "Normal flight",
        1: "Ball hitting ground (bounce)",
        2: "Ball hitting racket",
        3: "Ball hitting net",
        4: "Ball out of bounds",
        5: "Ball in service toss"
    }
    
    # Set text for visibility and status
    if waiting_for_visibility:
        vis_text = "SELECTING VISIBILITY: Press 0, 1, or 2"
        # Add colored background to show selection mode
        cv2.rectangle(info_panel, (0, 0), (info_panel.shape[1], 25), (0, 0, 150), -1)
    else:
        vis_text = f"Visibility: {current_visibility} - {visibility_desc[current_visibility]}"
    
    if waiting_for_status:
        status_text = "SELECTING STATUS: Press 0, 1, 2, 3, 4, or 5"
        # Add colored background to show selection mode
        cv2.rectangle(info_panel, (0, 25), (info_panel.shape[1], 50), (0, 150, 0), -1)
    else:
        status_text = f"Status: {current_status} - {status_desc[current_status]}"
    
    # Prepare coordinates text
    if current_annotation.get('x-coordinate') is not None and current_annotation.get('y-coordinate') is not None:
        coord_text = f"Coordinates: ({current_annotation['x-coordinate']}, {current_annotation['y-coordinate']})"
    else:
        coord_text = "Coordinates: Not set" if current_visibility > 0 else "Coordinates: N/A (ball not visible)"
    
    # Prepare instructions text
    instructions1 = "Press: V=visibility, S=status, ENTER=next image, Q=quit"
    instructions2 = "First press V or S, then press number key to select value"
    
    # Add text to panel
    cv2.putText(info_panel, vis_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(info_panel, status_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(info_panel, coord_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(info_panel, instructions1, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(info_panel, instructions2, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# --- Configuration ---
# Updated path to game11/Clip1
image_folder = r"D:\cycleing\ml\projet final\Tennis_project_gpu\datasets\trackNet\images\game11\Clip1"

# Specify the output CSV file path
output_csv_path = os.path.join(image_folder, "Label.csv")

# --- End Configuration ---

# Get list of image files (ensure sorted order like 0000.jpg, 0001.jpg, etc.)
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

if not image_files:
    print(f"Error: No supported image files found in the specified folder: {image_folder}")
else:
    print(f"Found {len(image_files)} images in {image_folder}.")
    
    # Load existing annotations
    existing_annotations = load_existing_annotations(output_csv_path)
    
    # Find the last annotated image index to allow resuming
    last_annotated_index = -1
    for i, img_name in enumerate(image_files):
        if img_name in existing_annotations:
            last_annotated_index = i
    
    # If some annotations already exist, ask user if they want to resume
    if last_annotated_index >= 0:
        print(f"Found annotations up to image {last_annotated_index+1}/{len(image_files)}: {image_files[last_annotated_index]}")
        print("You can resume from the next image.")
        start_index = last_annotated_index + 1
        
        # Load existing results to maintain them
        for i in range(start_index):
            img_name = image_files[i]
            if img_name in existing_annotations:
                results_list.append(existing_annotations[img_name])
    else:
        # Start from the beginning
        start_index = 0
        
        # If file exists but we have no valid annotations, we need to create a new file
        if os.path.exists(output_csv_path) and len(existing_annotations) == 0:
            # Create the CSV file with just the header
            header_df = pd.DataFrame(columns=['file name', 'visibility', 'x-coordinate', 'y-coordinate', 'status'])
            header_df.to_csv(output_csv_path, index=False, sep=csv_delimiter)

    # Create a window to display images. cv2.WINDOW_NORMAL allows resizing.
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # Set the mouse callback function globally for this window.
    cv2.setMouseCallback("Image", get_coordinates)

    print("\nInstructions:")
    print("  - Click on the ball in the image to mark its location.")
    print("  - Press 'v' followed by 0-2 to set visibility:")
    print("    - 0: Ball is gone (off-screen, fully blocked). No coordinates needed.")
    print("    - 1: Ball is clearly visible. Coordinates required.")
    print("    - 2: Ball is visible but hard to see. Coordinates required.")
    print("  - Press 's' followed by 0-5 to set status:")
    print("    - 0: Normal flight.")
    print("    - 1: Ball hitting the ground (bounce).")
    print("    - 2: Ball hitting a player's racket.")
    print("    - 3: Ball hitting the net.")
    print("    - 4: Ball out of bounds.")
    print("    - 5: Ball in service toss.")
    print("  - Press ENTER to save the current annotation and go to the next image.")
    print("  - Press 'q' to quit the annotation process.")
    print("  - Annotations are saved in real-time after each image.")
    print("-" * 60)

    # Loop through each image file, starting from the index after the last annotated one
    for i in range(start_index, len(image_files)):
        img_name = image_files[i]
        image_path = os.path.join(image_folder, img_name)

        # --- Initialize annotation for the current image ---
        current_annotation = {}  # Reset annotation dictionary for the new image
        current_annotation['file name'] = img_name
        current_visibility = 1  # Default: Ball is clearly visible
        current_status = 0      # Default: Normal flight
        current_annotation['visibility'] = current_visibility
        current_annotation['status'] = current_status
        current_annotation['x-coordinate'] = None
        current_annotation['y-coordinate'] = None

        # Check if this image already has an annotation (shouldn't happen in normal flow)
        if img_name in existing_annotations:
            # Load existing values
            anno = existing_annotations[img_name]
            current_visibility = int(anno['visibility'])
            current_status = int(anno['status'])
            current_annotation['visibility'] = current_visibility
            current_annotation['status'] = current_status
            
            # Handle coordinates - might be empty strings in CSV
            if anno['x-coordinate'] and anno['y-coordinate']:
                current_annotation['x-coordinate'] = int(anno['x-coordinate'])
                current_annotation['y-coordinate'] = int(anno['y-coordinate'])

        # Reset state variables
        waiting_for_visibility = False
        waiting_for_status = False

        # Load the image
        image_being_displayed = cv2.imread(image_path)
        if image_being_displayed is None:
            print(f"Warning: Could not load image {img_name}. Skipping.")
            results_list.append(current_annotation)
            save_annotation(current_annotation, output_csv_path)
            continue

        print(f"Annotating frame {i+1}/{len(image_files)}: {img_name}")

        # Create a working copy for display
        display_image = image_being_displayed.copy()
        
        # If we have coordinates already, draw the circle
        if current_annotation['x-coordinate'] is not None:
            x = current_annotation['x-coordinate']
            y = current_annotation['y-coordinate']
            cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        
        # Create and update the information panel
        update_info_panel()

        # Display the image with the information panel
        cv2.imshow("Image", np.vstack([info_panel, display_image]))
        
        while True:
            key = cv2.waitKey(50) & 0xFF  # Increased wait time for better key detection
            
            if key != 255:  # If a key was pressed (255 is "no key pressed")
                print(f"Key pressed: {chr(key) if key >= 32 and key <= 126 else key}")
            
            # Handle visibility selection mode
            if waiting_for_visibility:
                if key >= ord('0') and key <= ord('2'):
                    current_visibility = key - ord('0')
                    # If visibility is 0, clear coordinates
                    if current_visibility == 0:
                        current_annotation['x-coordinate'] = None
                        current_annotation['y-coordinate'] = None
                    current_annotation['visibility'] = current_visibility
                    waiting_for_visibility = False
                    
                    # Create fresh display image when visibility changes
                    display_image = image_being_displayed.copy()
                    
                    # If we have coordinates and visibility > 0, show the green dot
                    if current_visibility > 0 and current_annotation.get('x-coordinate') is not None:
                        x = current_annotation['x-coordinate']
                        y = current_annotation['y-coordinate']
                        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                    
                    update_info_panel()
                    cv2.imshow("Image", np.vstack([info_panel, display_image]))
                    print(f"Set visibility to {current_visibility}")
                
                # Allow exit from selection modes with ESC
                elif key == 27:  # ESC key
                    waiting_for_visibility = False
                    update_info_panel()
                    cv2.imshow("Image", np.vstack([info_panel, display_image]))
            
            # Handle status selection mode
            elif waiting_for_status:
                if key >= ord('0') and key <= ord('5'):
                    current_status = key - ord('0')
                    current_annotation['status'] = current_status
                    waiting_for_status = False
                    update_info_panel()
                    cv2.imshow("Image", np.vstack([info_panel, display_image]))
                    print(f"Set status to {current_status}")
                
                # Allow exit from selection modes with ESC
                elif key == 27:  # ESC key
                    waiting_for_status = False
                    update_info_panel()
                    cv2.imshow("Image", np.vstack([info_panel, display_image]))
            
            # Regular key handling
            elif key == ord('v') or key == ord('V'):
                waiting_for_visibility = True
                waiting_for_status = False
                update_info_panel()
                cv2.imshow("Image", np.vstack([info_panel, display_image]))
                print("Select visibility (0-2):")
                print("  0: Ball is gone (off-screen, fully blocked)")
                print("  1: Ball is clearly visible")
                print("  2: Ball is visible but hard to see")
            
            elif key == ord('s') or key == ord('S'):
                waiting_for_status = True
                waiting_for_visibility = False
                update_info_panel()
                cv2.imshow("Image", np.vstack([info_panel, display_image]))
                print("Select status (0-5):")
                print("  0: Normal flight")
                print("  1: Ball hitting ground (bounce)")
                print("  2: Ball hitting racket")
                print("  3: Ball hitting net")
                print("  4: Ball out of bounds")
                print("  5: Ball in service toss")
            
            # Number keys for direct setting (without v/s first)
            # Allow direct setting of visibility with Shift+number
            elif key == ord('!'):  # Shift+1
                current_visibility = 1
                current_annotation['visibility'] = current_visibility
                update_info_panel()
                cv2.imshow("Image", np.vstack([info_panel, display_image]))
                print(f"Set visibility to {current_visibility}")
            elif key == ord('@'):  # Shift+2
                current_visibility = 2 
                current_annotation['visibility'] = current_visibility
                update_info_panel()
                cv2.imshow("Image", np.vstack([info_panel, display_image]))
                print(f"Set visibility to {current_visibility}")
            elif key == ord('#'):  # Shift+3
                current_status = 0  # Normal flight
                current_annotation['status'] = current_status
                update_info_panel()
                cv2.imshow("Image", np.vstack([info_panel, display_image]))
                print(f"Set status to {current_status}")
            elif key == ord('$'):  # Shift+4
                current_status = 1  # Bounce
                current_annotation['status'] = current_status
                update_info_panel()
                cv2.imshow("Image", np.vstack([info_panel, display_image]))
                print(f"Set status to {current_status}")
            elif key == ord('%'):  # Shift+5
                current_status = 2  # Racket hit
                current_annotation['status'] = current_status
                update_info_panel()
                cv2.imshow("Image", np.vstack([info_panel, display_image]))
                print(f"Set status to {current_status}")
            
            # ENTER key saves current frame and advances to next
            elif key == 13 or key == 10:  # ENTER key (13 for Windows, 10 for some systems)
                # Verify we have coordinates if ball is visible
                if current_visibility > 0 and current_annotation['x-coordinate'] is None:
                    print("Warning: Ball is marked as visible but no coordinates set!")
                    print("Please click on the ball or change visibility to 0.")
                    continue
                
                # Save annotation right away to CSV file
                save_annotation(current_annotation, output_csv_path)
                
                # Add to results list for full tracking
                results_list.append(current_annotation)
                break
                
            elif key == ord('q') or key == ord('Q'):
                # Save current annotation before quitting
                if current_visibility > 0 and current_annotation['x-coordinate'] is None:
                    print("Warning: Ball is marked as visible but no coordinates set!")
                    print("Changing visibility to 0 before saving.")
                    current_annotation['visibility'] = 0
                    current_visibility = 0
                
                # Save the current annotation
                save_annotation(current_annotation, output_csv_path)
                results_list.append(current_annotation)
                break

        # Check if the outer loop should break (only happens if 'q' was pressed)
        if key == ord('q') or key == ord('Q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    print("\nAnnotation process complete. All annotations have been saved to:")
    print(output_csv_path)