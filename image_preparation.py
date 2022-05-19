import os
from PIL import Image

scratched_images = os.listdir("scratched_source/")

cur = 0

images_size = (256, 256)

times = 6
angle = 360.0 / times

for image in scratched_images:
    img = Image.open('scratched_source/' + image).convert("L").resize(images_size)
    img.save("scratched_ready/" + str(cur) + ".jpg")
    cur += 1

    original = img

    for i in range(1, times):
        img = original.rotate(angle * i)
        img.save("scratched_ready/" + str(cur) + ".jpg")
        cur += 1


nonscratched_images = os.listdir("nonscratched_source/")

cur = 0

for image in nonscratched_images:
    img = Image.open('nonscratched_source/' + image).convert("L").resize(images_size)
    img.save("nonscratched_ready/" + str(cur) + ".jpg")
    cur += 1

    original = img

    for i in range(1, times):
        img = original.rotate(angle * i)
        img.save("nonscratched_ready/" + str(cur) + ".jpg")
        cur += 1

scratched_test_source = os.listdir("scratched_test_source/")

cur = 0

for image in scratched_test_source:
    img = Image.open('scratched_test_source/' + image).convert("L").resize(images_size)
    img.save("scratched_test_ready/" + str(cur) + ".jpg")
    cur += 1

    original = img

    for i in range(1, times):
        img = original.rotate(angle * i)
        img.save("scratched_test_ready/" + str(cur) + ".jpg")
        cur += 1

nonscratched_test_source = os.listdir("nonscratched_test_source/")

cur = 0

for image in nonscratched_test_source:
    img = Image.open('nonscratched_test_source/' + image).convert("L").resize(images_size)
    img.save("nonscratched_test_ready/" + str(cur) + ".jpg")
    cur += 1

    original = img

    for i in range(1, times):
        img = original.rotate(angle * i)
        img.save("nonscratched_test_ready/" + str(cur) + ".jpg")
        cur += 1
