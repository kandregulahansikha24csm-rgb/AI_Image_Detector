from pathlib import Path
import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

folders = [
    ("real", 0),
    ("ai", 1),
    ("fake", 1),
]

for folder_name, label in folders:
    folder = Path("dataset") / folder_name

    total = 0
    readable = 0
    unreadable = []

    print("\nChecking:", folder)

    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total += 1
        image = cv2.imread(str(image_path))

        if image is not None:
            readable += 1
        else:
            unreadable.append(str(image_path))

    print("Total images    :", total)
    print("Readable images :", readable)
    print("Unreadable      :", len(unreadable))

    if unreadable:
        print("\nUnreadable files:")
        for file in unreadable:
            print(file)

print("\nImage readability check completed.")