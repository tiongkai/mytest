from ultralytics import YOLO
import cv2
import yaml



#load yolo annotations based on COCO. 
annotations_dict = {}
with open("models/configs/data.yaml", "r") as stream:
    try:
        model_dict = yaml.safe_load(stream)
        annotations_dict = model_dict['names']
        print(model_dict)

    except yaml.YAMLError as exc:
        print(exc)
model = YOLO("runs/detect/train8/weights/best.pt") #load pretrain model based on COCO 80 classes.



cap = cv2.VideoCapture('a.mp4')

 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('aResult.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         60, size)
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    image = frame

    results = model.predict(source=image)  # predict on an image

    #print('Results type: ', type(results))
    #print('---')
    #print(results)

    fontScale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_green = (0, 255, 0)

    for _, each_result in enumerate(results):
        #print(type(each_result))
        #print()
        for each_detection in each_result:
            #print((each_detection.boxes))
            boxes = each_detection.boxes.xyxy

            conf =  each_detection.boxes.conf
            class_id = int(each_detection.boxes.cls)
            
            x_min, y_min, x_max, y_max = boxes.tolist()[0]
            
            tl = (int(x_min), int(y_min))
            br = (int(x_max), int(y_max))
            width = x_max - x_min
            height = y_max - y_min
            print(class_id)
            #draw label
            if annotations_dict[class_id] in annotations_dict:
                
                label = annotations_dict[class_id]

                # #draw bounding box
                cv2.rectangle(image, tl, br, color=(0,255,0), thickness=2)
                
                #point origin. 
                org = (tl[0], tl[1]-5) 
                conf_string = "{:.2f}".format(conf[0])

                display_label = label + " " + conf_string
                fontScale = 0.5

                image = cv2.putText(image, display_label, org, font, 
                    fontScale, color_green, thickness=1, lineType = cv2.LINE_AA,)
                

    cv2.imshow("Yolo Results", image)
    result.write(image)
  else: 
    break

cap.release()
cv2.destroyAllWindows()


