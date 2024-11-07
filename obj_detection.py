import cv2
import sys
sys.path.insert(0, 'yolov7')
from hubconf import custom  


print('Loading Yolov7 weights')
weights_path = 'best.pt'
model = custom(path_or_model=weights_path)
model.eval()

class_names = ['smoke', 'fire']

def detect_objects(frame):
    # YOLOv7 Detection
    results = model([frame]) 
    detected_labels = []

     # Process each detection
    for i, (x1, y1, x2, y2, conf, class_id) in enumerate(results.xyxy[0]):
        # Get the class name using the class ID
        detected_class = class_names[int(class_id)]
        print(detected_class)
        # Draw rectangle and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'{detected_class} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        detected_labels.append(detected_class)
            
    return(frame, detected_labels)
 
if __name__ == "__main__":
    # Example usage
    frame = cv2.imread('smoke.jpg')

    # Detect objects
    frame, results = detect_objects(frame)
    #detector.crop_objects_by_name(results, frame)
    cv2.imshow('results', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
