import cv2
import numpy as np

paths = ['./Photos_test/chat1.jpg']
output_paths = [f"{path.split('.jpg')[0]}_noised.jpg" for path in paths]

def noise(images, std=1):
	for image in images:
		noise = np.random.normal(0, std, size=image.shape)
		noised_images.append(image + noise)
	return noised_images

images = [cv2.imread(path) for path in paths]

noised = noise(paths, 100)

for image, out_path in zip(noised, output_paths):
	cv2.imwrite(out_path, image)
