import os
import skimage
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import skimage.io as io


def img_hair_extract(image, lower_limit=20, size=640):
    image_resize = cv2.resize(image, (size, size))
    grayScale = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting
    _, threshold = cv2.threshold(blackhat, lower_limit, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    final_image = cv2.inpaint(image_resize, threshold, 1, cv2.INPAINT_TELEA)

    threshold = cv2.bitwise_not(threshold)
    image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    return image_resize, threshold, final_image


if __name__ == '__main__':
    hair_imgs = glob(os.path.join('/home/zyi/MedicalAI/ISIC_2020/hairs2/image_with_hairs', '*.jpg'))
    img_1 = io.imread(hair_imgs[np.random.randint(len(hair_imgs))])
    img_2 = io.imread('/home/zyi/MedicalAI/ISIC_2020/hairs2/tttt/ISIC_0310320.jpg')
    img_2 = cv2.resize(img_2, (640, 640))

    img, mask, _ = img_hair_extract(img_1)
    # im_mask = skimage.filters.gaussian(mask, sigma=0.8)
    # im_add = np.stack([im_mask, im_mask, im_mask], axis=2) + skimage.transform.resize(img_2, im_mask.shape)
    # plt.figure()
    # plt.imshow(im_add)
    img_2_hair = cv2.bitwise_and(img_2, img_2, mask=mask)
    img_2_hair = skimage.filters.gaussian(img_2_hair, sigma=1)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0][0].imshow(img[:, :, ::-1])
    axes[0][1].imshow(img_2)
    axes[1][0].imshow(mask, cmap='binary_r')
    axes[1][1].imshow(img_2_hair)
    plt.figure()
    plt.imshow(img_2_hair)
    plt.imsave('/home/zyi/MedicalAI/ISIC_2020/hairs2/tttt/5.png', img_2_hair)
    io.imsave('/home/zyi/MedicalAI/ISIC_2020/hairs2/tttt/51.png', mask.astype(np.uint8))
    plt.imsave('/home/zyi/MedicalAI/ISIC_2020/hairs2/tttt/52.png', img[:, :, ::-1])
    plt.show()

    # generate hair mask from selected image with hairs
    # hair_imgs = glob(os.path.join('/home/zyi/MedicalAI/ISIC_2020/hairs2/ddd', '*.jpg'))
    # for img_dir in hair_imgs:
    #     img = io.imread(img_dir)
    #     __, mask, _ = img_hair_extract(img)
    #     io.imsave(os.path.join('/home/zyi/MedicalAI/ISIC_2020/hairs2/mask', os.path.basename(img_dir)),
    #               mask.astype(np.uint8))

    # group and save hair mask
    # mask_list = glob(os.path.join('/home/zyi/MedicalAI/ISIC_2020/hairs2/mask', '*.jpg'))
    # mask_array = []
    # for mask in mask_list:
    #     mask = io.imread(mask)
    #     mask_array.append(mask)
    #
    # mask_array = np.stack(mask_array, axis=0)
    # np.save('/home/zyi/MedicalAI/ISIC_2020/hairs2/mask_array.npy', mask_array)