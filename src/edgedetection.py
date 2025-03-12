import cv2
import sys

def apply_canny(input_path, output_path=None, threshold1=100, threshold2=200):
    # Read the image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Unable to read image {input_path}")
        return

    # Apply Canny edge detection        
    img = cv2.equalizeHist(img)
    edges = cv2.Canny(img, threshold1, threshold2)

    if output_path:
        # Save the result
        cv2.imwrite(output_path, edges)
        print(f"Saved edge-detected image to {output_path}")
    else:
        # Show the result in a window
        cv2.imshow("Canny Edge Detection", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python canny_edge.py <input_image> [output_image] [threshold1] [threshold2]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None  # Use "-" to skip saving

    # Optional thresholds
    thresh1 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    thresh2 = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    apply_canny(input_file, output_file, thresh1, thresh2)
