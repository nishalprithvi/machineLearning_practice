from PIL import Image

# using pillow library to load and analyse the image

try:
    firstImage = Image.open('mo_salah.jpg')
    print(f"Successfully loaded image!")
    print(f"Image mode: {firstImage.mode}") # e.g., 'L' for black & white, 'RGB' for color
    print(f"Image size (width, height): {firstImage.size}")

    # displaying the image 
    firstImage.show()

except FileNotFoundError:
    print('Error !! Image not Found !!!! Make sure you enter the correct path')