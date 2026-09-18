import os
import random
import cv2
import numpy as np

base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
real_dir = os.path.join(base, 'real')
ai_dir = os.path.join(base, 'ai')
os.makedirs(real_dir, exist_ok=True)
os.makedirs(ai_dir, exist_ok=True)

random.seed(42)


def save_image(path, img):
    cv2.imwrite(path, img)


for i in range(10):
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            img[y, x] = [30 + y // 4, 50 + x // 6, 90 + (x + y) // 8]

    for _ in range(6):
        x0, y0 = random.randint(20, w - 20), random.randint(20, h - 20)
        r = random.randint(20, 60)
        color = (random.randint(80, 220), random.randint(80, 220), random.randint(80, 220))
        cv2.circle(img, (x0, y0), r, color, -1)

    for _ in range(20):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        cv2.line(img, (x1, y1), (x2, y2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)

    save_image(os.path.join(real_dir, f'real_{i + 1}.png'), img)

for i in range(10):
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(0, h, 8):
        for x in range(0, w, 8):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img, (x, y), (x + 6, y + 6), color, -1)

    for y in range(h):
        adjustment = np.sin(np.arange(w) / 10.0) * 8
        img[y, :, :] = np.clip(
            img[y, :, :] + adjustment[:, np.newaxis],
            0,
            255,
        ).astype(np.uint8)

    for _ in range(4):
        x0, y0 = random.randint(0, w - 1), random.randint(0, h - 1)
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        cv2.line(img, (x0, y0), (x1, y1), (255, 255, 255), 1)

    save_image(os.path.join(ai_dir, f'ai_{i + 1}.png'), img)

print('Created sample images in dataset/real and dataset/ai')
