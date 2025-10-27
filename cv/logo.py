from PIL import Image

# Open the image
img = Image.open("Modelisation MERISE.jpg").convert("RGBA")

# Choose the color to replace (R, G, B, A)
threshhold = 100
old_color = (40, 159, 232, 255)
new_color = (40, 159, 232, 255)

# Load pixels
pixels = img.load()
print(pixels[0, 0])
print(pixels[0, 0][1])

# Loop through all pixels
for y in range(img.height):
    for x in range(img.width):
        if pixels[x, y][0] > threshhold and pixels[x, y][1] > threshhold and pixels[x, y][2] > threshhold and pixels[x, y][3] > threshhold:
            pixels[x, y] = new_color

# Save the modified image
img.save("newoutput.png")
