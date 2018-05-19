import cv2
f_name = input('enter filename: ')
f = f_name + '.jpg'
out = f_name +'.png'
img = cv2.imread(f)
cv2.imwrite(out, img)