"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import json
import numpy as np
from PIL import Image
import random
import cv2
from synthtiger import components, layers, templates, utils


class Multiline(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.count = config.get("count", 1000)
        self.corpus = components.BaseCorpus(**config.get("corpus", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.color = components.RGB(**config.get("color", {}))
        self.layout = components.FlowLayout(**config.get("layout", {}))
        self.texture = components.Switch(components.BaseTexture(), **config.get("texture", {}))
        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {}),
        )

        
    def draw_crop_image_box(self, text_layer, boxes, img_name):
        
        bg_image = self._generate_background(text_layer.size)
        bg_image.topleft = text_layer.topleft 

        image = (text_layer + bg_image).output()
        # image = text_layer.output()

        width=0
        hight=0
        fx, fy, fw, fh = 0,0,0,0
        for box in boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            fw = fx+w
            # print((int(fx), int(fy)), (int(fw), int(h)))
            image=cv2.rectangle(image, (int(fx), int(fy)), (int(fw), int(h)), (255, 0, 0), 2)
            fx += w

            width = width + w
            hight = max(hight, h) 
        
        x2 = 0+width
        y2 = 0+hight
        # cv2.rectangle(image, (0, 0), (int(x2), int(y2)), (0, 255, 0), 2)
        save_path = '/home/gaurav/scratch/synthtiger/results/char_box/' + str(img_name) + '.jpg'
        cv2.imwrite(save_path, image)

    def generate(self):
        texts = [self.corpus.data(self.corpus.sample()) for _ in range(100)]
        # fonts = [self.font.sample() for _ in range(self.count)]
        fonts = [{'path': '/home/gaurav/scratch/synthtiger/resources/font/ABeeZee-Italic.ttf', 'size': 32, 'bold': True, 'vertical': False}]
        print(fonts)
        color = self.color.data(self.color.sample())

        boxes_all_word = []

        i=0
        for word, font in zip(texts,fonts):
            chars = utils.split_text(word, reorder=False)
            # font = fonts[0]
            char_layers = [layers.TextLayer(char, **font) for char in chars]
            word_layers = layers.TextLayer(word, color=color, **font)

            text_layer = layers.Group(char_layers).merge()
            
            # fg_style = self.style.sample()
            
            # self.style.apply([text_layer, *char_layers], fg_style)

            for char_layer in char_layers:
                char_layer.topleft -= text_layer.topleft
            
            bboxes_per_word_no_of_char = [char_layer.bbox for char_layer in char_layers]
            self.draw_crop_image_box(word_layers, bboxes_per_word_no_of_char, i)
            i+=1
            boxes_all_word.append(bboxes_per_word_no_of_char)


        a = []
        i=0
        writer = {}
        fx, fy, fw, fh = 0, 0, 0, 0 

        number_of_words = random.randint(2, 14)
        random_offset = random.randint(5, 30)
        for word, font in zip(texts, fonts):
            
            # print(word)
            # print(boxes)
            boxes = boxes_all_word[i]

            width, hight = 0, 0

            char_layers = layers.TextLayer(word, color=color, **font)
            text_layer = layers.Group(char_layers).merge()

            self.texture.apply([char_layers])
            # fg_style = styles_used[i]
            # self.style.apply([char_layers], fg_style)

            a.append(char_layers)
            for j, box in enumerate(boxes):
                x, y, w, h = box[0], box[1], box[2], box[3]
                width = width + w
                hight = max(hight, h) 

            i+=1
            
            fw = width
            fh = hight

            if i==1:
                box_coordinate = [int(fx+random_offset), int(fy+random_offset), int(fw+5), int(fh)]
            elif i==number_of_words:
                box_coordinate = [int(fx-5+random_offset), int(fy+random_offset), int(fw+5), int(fh)]
            else:
                box_coordinate = [int(fx-5+random_offset), int(fy+random_offset), int(fw+5), int(fh)]
                
            writer[i] = box_coordinate
            # x1 = x2+16
            # x2 = x1
            fx = fx + fw  + 16

            if i==number_of_words:
                break
            
            
    
        text_group = layers.Group(a)

        # text_group = layers.Group(
        #     [
        #         layers.TextLayer(text, color=color, **font)
        #         for text, font in zip(texts, fonts)
        #     ]
        # )
        self.layout.apply(text_group)
        

        bg_image = self._generate_background(text_group.size)
        bg_image.topleft = text_group.topleft - random_offset

        image = (text_group + bg_image).output()
        label = " ".join(texts)

        data = {
            "image": image,
            "label": label,
            "boxes": writer,
        }

        return data

    def _generate_background(self, size):
        random_width = random.randint(20, 30)
        random_height = random.randint(65, 75)
        mod_size = [size[0]+random_width+10, random_height]
        color = self.color.data(self.color.sample())
        layer = layers.RectLayer(mod_size, color)
        self.texture.apply([layer])
        return layer
    
    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        gt_path = os.path.join(root, "gt.txt")
        self.gt_file = open(gt_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        boxes = data["boxes"]

        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        image_path = os.path.join(root, image_key)

        boxes_key = os.path.join("boxes", shard, f"{idx}.json")
        boxes_path = os.path.join(root, boxes_key)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        os.makedirs(os.path.dirname(boxes_path), exist_ok=True)

        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_path, quality=95)

        with open(boxes_path, 'w') as file: 
            json.dump(boxes, file)

        self.gt_file.write(f"{image_key}\t{label}\n")

    def end_save(self, root):
        self.gt_file.close()
