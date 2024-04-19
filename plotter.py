import cv2
import os
import json

def draw_box(image, box, save_path):
    # Read the image
    x1, y1, w, h = box[0], box[1], box[2], box[3]

    # Draw the box
    x2 = x1+w
    y2 = y1+h
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the modified image
    cv2.imwrite(save_path, image)

    print("Image with box drawn saved successfully at:", save_path)
    return image

# Example usage
if __name__ == "__main__":
    image_dir = "results/images/0"
    boxes_dir = "results/boxes/0"
    output_dir = "results/vis_box"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for image_name in sorted(os.listdir(image_dir), key=lambda x: x.split('.')[0]):
        image_path = os.path.join(image_dir, image_name)
        boxes_name = image_name.split('.')[0] + '.json'
        boxes_path = os.path.join(boxes_dir, boxes_name)

        output_path = os.path.join(output_dir, image_name)

        image = cv2.imread(image_path)
        with open(boxes_path, 'r') as file:
            dictionary = json.load(file)
            for i in dictionary.keys():
                box = dictionary[i]
                image = draw_box(image, box, output_path)


#   {1: [0, 0, 96.0, 33.0], 2: [111.0, 0, 271.0, 33.0], 3: [286.0, 0, 398.0, 33.0], 4: [413.0, 0, 509.0, 33.0]}