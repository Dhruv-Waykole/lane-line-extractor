Lane Line Extractor:
Libraries used: openCV and Numpy
My code reads a image using opencv and displays it. It displays the original image, blurred image, display the image with canny edges, it displays the image with HSV transformation and displays the image with roi
The region of interest is done by using a polygon or in this case its a trapezoid.
The program first uses hsv and keeps only the yellow and white color on the image and canny edges is used. Then the program divides the image into smaller segments and based on the slope these segements are segregated and then combined to form one straight line.
The program finds the 4 corners of the polygon. It does this by the help of the lines made. It finds two positions: one where the line starts and 2nd where they start to disappear or intersect.
For each height, it calculates the horizontal position of the left and right lane lines, which gives four points: bottom-left, top-left, top-right, and bottom-right. 
The program also shows the preview of the polygon on the original image. This polygon is represent by a red line. The program basically creates another image of the same height and width of the first image, exepct that this time the new image is completely black.
In this black image, the lanes drawn and detected by the program which have the yellow and white color are used to give the final input of the ROI.
