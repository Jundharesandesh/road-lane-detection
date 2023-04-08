from flask import Flask,render_template,request
import matplotlib.pylab as plt
import cv2
import numpy as np
import matplotlib.pylab as plt
import cv2
import numpy as np
from PIL import Image
import os







def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img



app = Flask(__name__)



@app.route("/")
def home ():
    return render_template("index.html")

@app.route("/",methods=['POST'])
   
def pre ():
     
    imagefile=request.files["imagefile"]
    image_path="./static/" + imagefile.filename
    imagefile.save(image_path)

    image = cv2.imread("./static/" + imagefile.filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=6,
                            theta=np.pi/180,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=25)
    img = drow_the_lines(image, lines)
   
    if not os.path.exists('static'):
     os.makedirs('static')

# Save output image
    cv2.imwrite('static/road_lanes.jpg', img)
    
    

    
   

    
   
   
    return render_template("diaplay.html")


if __name__ == "__main__":
    app.run()