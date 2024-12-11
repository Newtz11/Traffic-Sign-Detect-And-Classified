import cv2
from matplotlib import pyplot as plt
import numpy as np


def trafficSignDetect(frame):
    
    # Chuyển đổi ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Chọn các màu đặc trưng của biển báo giao thông
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])
    blue_lower = np.array([100, 150, 50])
    blue_upper = np.array([140, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Tạo các mặt nạ màu cho đỏ, xanh và vàng
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Kết hợp các mặt nạ màu thành một
    mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, blue_mask), yellow_mask)

    # Áp dụng các bộ lọc để giảm nhiễu
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Tìm các đường viền của các đối tượng có trong mặt nạ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = float(w) / h

        # Giới hạn diện tích và tỷ lệ khung cho biển báo giao thông
        if 3000 < area < 5000 and 0.8 < aspect_ratio < 1.2:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

            # Phát hiện các hình dạng cụ thể như hình tròn hoặc tam giác
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.6: 
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = frame[y:y+h, x:x+w]
                text = classifyTrafficLight(roi)
                cv2.putText(frame, text, (x + w + 20, y + int(h/2)), cv2.FONT_HERSHEY_PLAIN, 
                   2, (255,0,0), 2, cv2.LINE_AA)
                
                
            if len(approx) == 3:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = frame[y:y+h, x:x+w]
                text = classifyTrafficLight(roi)
                cv2.putText(frame, text, (x + 80, y + 40) , cv2.FONT_HERSHEY_PLAIN, 
                   2, (255,0,0), 2, cv2.LINE_AA)
                

    return frame
def classifyTrafficLight(image):
    
    # Convert the image to HSV color space
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsvImage, lower_blue, upper_blue)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaBlue = 0
    for contour in contours:
        areaBlue += cv2.contourArea(contour)
    if areaBlue > 2000:
        return "Phai di vong sang ben phai"
    else: 
        #red
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([179, 255, 255])
        red_mask1 = cv2.inRange(hsvImage, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsvImage, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        areaRed = 0
        
        for contour in contours:
            areaRed += cv2.contourArea(contour)
        #white
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([180, 25, 255])  

        white_mask = cv2.inRange(hsvImage, lower_white, upper_white)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areaWhite = 0
        for contour in contours:
            areaWhite += cv2.contourArea(contour)
        
        if areaRed > areaWhite:
            return "Cam di nguoc chieu"
        
        return "Cam re trai"
            
    




# Opening image
def main():
    cap = cv2.VideoCapture('video2.mp4')
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # out = cv2.VideoWriter('task1_output.avi', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện biển báo giao thông
        frame = trafficSignDetect(frame)
        cv2.putText(frame, '52200143-52200218-52200193', (20,30), cv2.FONT_HERSHEY_PLAIN, 
                   2, (255,0,0), 2, cv2.LINE_AA)
        #out.write(frame)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    
    
main()