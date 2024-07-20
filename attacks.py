import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import imageio.v2 as imageio
from PIL import Image, ImageFont, ImageDraw
from skimage import exposure
from scipy.ndimage import rotate ,median_filter,uniform_filter,gaussian_filter,convolve
from sklearn.cluster import KMeans
from skimage.util import random_noise
from config_variables import *
from functions import *

def TIFF_save(watermarked_img):
   imageio.imwrite(os.path.join(current_directory, 'Images/Saved.tiff'), watermarked_img, format="tiff")
   TIFF_img = imageio.imread(os.path.join(current_directory, 'Images/Saved.tiff'))

   mse_value, psnr_value, ssim_value = Image_metrics(watermarked_img,TIFF_img)         
   print("TIFF image PSNR : ",psnr_value, "\t \t TIFF image MSE : ",mse_value ,"\t \t TIFF image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(TIFF_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(TIFF_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(TIFF_img, cmap='gray')
   ax1.set_title('TIFF image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Extracted watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()

def PNG_save(watermarked_img):
   pil_image = Image.fromarray(watermarked_img) 
   pil_image.save(os.path.join(current_directory, 'Images/Saved.png'))
   png_img = imageio.imread(os.path.join(current_directory, 'Images/Saved.png'))

   mse_value, psnr_value, ssim_value = Image_metrics(watermarked_img, png_img)        
   print("PNG image PSNR : ",psnr_value, "\t \t PNG image MSE : ",mse_value ,"\t \t PNG image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(png_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(png_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(png_img, cmap='gray')
   ax1.set_title('PNG image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Extracted watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()

def JPEG_save(watermarked_img,qual):
   pil_image = Image.fromarray(watermarked_img)
   pil_image.save(os.path.join(current_directory, 'Images/Saved.jpg'),"JPEG",quality=qual)   
   jpeg_img = imageio.imread(os.path.join(current_directory, 'Images/Saved.jpg'))  

   mse_value, psnr_value, ssim_value = Image_metrics(watermarked_img, jpeg_img)         
   print("JPEG image PSNR : ",psnr_value, "\t \t JPEG image MSE : ",mse_value ,"\t \t JPEG image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(jpeg_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(jpeg_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(jpeg_img, cmap='gray')
   ax1.set_title('JPEG image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Extracted watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()


def Rotation_attack(watermarked_img,rot):
   rotated_img = rotate(watermarked_img, rot,reshape=False)

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,rotated_img)
   print("Rotated image PSNR : ",psnr_value, "\t \t Rotated image MSE : ",mse_value, "\t \t Rotated image MSE : ",ssim_value )

   if self_embed==True:      
      org_water=self_embedding(rotated_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(rotated_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(rotated_img, cmap='gray')
   ax1.set_title('Rotated image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()

def Flip_direction_attack(watermarked_img,ax):
   flipped_img = np.flip(watermarked_img, axis=ax)

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  flipped_img)        
   print("flipped image PSNR : ",psnr_value, "\t \t flipped image MSE : ",mse_value,  "\t \t flipped image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(flipped_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(flipped_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   flipped_im=flipped_img
   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow( flipped_im, cmap='gray')
   ax1.set_title('flipped  image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()

def Scaling_attack(watermarked_img,size):
   scaled_img=cv2.resize(watermarked_img,(size,size))
   if self_embed==True:      
      org_water=self_embedding(scaled_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(scaled_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(scaled_img, cmap='gray')
   ax1.set_title('Scaled image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()

def Translation_attack(watermarked_img,dx,dy):
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    translated_img = cv2.warpAffine(watermarked_img, M, (watermarked_img.shape[1], watermarked_img.shape[0]))

    mse_value, psnr_value, ssim_value = Image_metrics(original_img, translated_img)           
    print("Translated image PSNR : ",psnr_value, "\t \t Translated image MSE : ",mse_value ,"\t \t Translated image SSIM : ",ssim_value)
      
    if self_embed==True:      
      org_water=self_embedding(translated_img)
    else:
      org_water=org_watermark

    ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(translated_img)
    tamper=Tamper_detection(org_water,ext_watermark)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(translated_img, cmap='gray')
    ax1.set_title('Translated image')
    ax2.imshow(ext_watermark, cmap='gray')
    ax2.set_title('Attacked watermark')
    ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
    ax3.set_title('Tampering')
    plt.show()

def Zooming_attack(watermarked_img,XD, YD, sizex, sizey):
   zoomed_img=cv2.resize(watermarked_img,(sizey,sizex))
   for i in range(sizex):
      for j in range(sizey):
         zoomed_img[i][j]=watermarked_img[XD+i][YD+j]
   zoomed_img=cv2.resize(zoomed_img,(img_size,img_size))

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  zoomed_img)           
   print("Zoomed image PSNR : ",psnr_value, "\t \t Zoomed image MSE : ",mse_value ,"\t \t Zoomed image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(zoomed_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( zoomed_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(zoomed_img, cmap='gray')
   ax1.set_title('Zoomed image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()

def Scanning_attack(watermarked_img,type,size=256,rot=30):
   if type =='scalling':
      S_img=cv2.resize(watermarked_img,(size,size))
      S_img=cv2.resize(S_img,(img_size,img_size))
   elif type=='rotation':
      S_img = rotate(watermarked_img, rot,reshape=False)
      S_img = rotate(S_img, rot*-1,reshape=False)
   else:
      raise ValueError("Choose either scalling or rotation")

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,S_img)           
   print("Scanning image PSNR : ",psnr_value, "\t \t Scanning image MSE : ",mse_value ,"\t \t Scanning image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(S_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( S_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(S_img, cmap='gray')
   ax1.set_title('Scanning image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()


def Gaussian_noise(watermarked_img,mean,stddev):
   if len((np.asarray(watermarked_img)).shape)==3:
      long=3
   else:
      long=1

   image=[]
   for channel in range(long):
      if long==3:
         img=watermarked_img[:, :, channel]
      else:
         img=watermarked_img
      noise = np.random.normal(mean, stddev, img.shape)
      noisy_img = img + noise
      image.append(noisy_img)

   if long==3:
       noisy_img = np.stack([image[0], image[1], image[2]], axis=2)
   else:
       noisy_img=image[0]
   
   noisy_img = noisy_img.astype("uint8")
   mse_value, psnr_value, ssim_value = Image_metrics(original_img, noisy_img)           
   print("Noisy image PSNR : ",psnr_value, "\t \t Noisy image MSE : ",mse_value ,"\t \t Noisy image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(noisy_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(noisy_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(noisy_img, cmap='gray')
   ax1.set_title('Noisy image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def Salt_paper(watermarked_img,salt_vs_pepper,amount):
   salt_paper_image=np.copy(watermarked_img)
   # Generate salt-and-pepper noise
   salt = np.ceil(amount * salt_paper_image.size * salt_vs_pepper)
   pepper = np.ceil(amount * salt_paper_image.size * (1.0 - salt_vs_pepper))
   coords_salt = [np.random.randint(0, i - 1, int(salt)) for i in salt_paper_image.shape]
   coords_pepper = [np.random.randint(0, i - 1, int(pepper)) for i in salt_paper_image.shape]
   salt_paper_image[coords_salt[0], coords_salt[1]] = 255
   salt_paper_image[coords_pepper[0], coords_pepper[1]] = 0

   mse_value, psnr_value, ssim_value = Image_metrics(original_img, salt_paper_image)           
   print("Salt&paper image PSNR : ",psnr_value, "\t \t Salt&paper image MSE : ",mse_value ,"\t \t Salt&paper image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(salt_paper_image)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(salt_paper_image)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(salt_paper_image, cmap='gray')
   ax1.set_title('Salt&paper image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def Speckle_noise(watermarked_img):
   if len((np.asarray(watermarked_img)).shape)==3:
      long=3
   else:
      long=1

   image=[]
   for channel in range(long):
      if long==3:
         img=watermarked_img[:, :, channel]
      else:
         img=watermarked_img
      noisy_img = img / np.max(img)
      noisy_img = random_noise(noisy_img, mode='speckle', seed=None, clip=True)
      noisy_img = noisy_img * np.max(img)
      noisy_img = noisy_img.astype("uint8")
      image.append(noisy_img)

   if long==3:
       noisy_img = np.stack([image[0], image[1], image[2]], axis=2)
   else:
       noisy_img=image[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img, noisy_img)           
   print("Speckle noise image PSNR : ",psnr_value, "\t \t Speckle noise image MSE : ",mse_value ,"\t \t Speckle noise image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(noisy_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(noisy_img)
   tamper=Tamper_detection(org_water,ext_watermark)
   
   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(noisy_img, cmap='gray')
   ax1.set_title('Speckle noise')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def Median_filtring(watermarked_img,size):
   filtered_image = median_filter(watermarked_img, size=size)

   mse_value, psnr_value, ssim_value = Image_metrics(original_img, filtered_image)           
   print("Median filter image PSNR : ",psnr_value, "\t \t Median filter image MSE : ",mse_value ,"\t \t Median filter image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(filtered_image)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(filtered_image)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(filtered_image, cmap='gray')
   ax1.set_title('Median filter image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def Average_filtring(watermarked_img,size):
   if len((np.asarray(watermarked_img)).shape)==3:
      long=3
   else:
      long=1

   image=[]
   for channel in range(long):
      if long==3:
         img=watermarked_img[:, :, channel]
      else:
         img=watermarked_img
      img = uniform_filter(img, size=size)
      image.append(img)

   if long==3:
      filtered_image =  np.stack([image[0], image[1], image[2]], axis=2)
   else:
      filtered_image=image[0]
      
   mse_value, psnr_value, ssim_value = Image_metrics(original_img, filtered_image)           
   print("Average filter image PSNR : ",psnr_value, "\t \t Average filter image MSE : ",mse_value ,"\t \t Average filter image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(filtered_image)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(filtered_image)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(filtered_image, cmap='gray')
   ax1.set_title('Average filter image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def Gaussian_filtring(watermarked_img):
   filtered_image = gaussian_filter(watermarked_img, sigma=1.0, order=0)

   mse_value, psnr_value, ssim_value = Image_metrics(original_img, filtered_image)           
   print("Gaussian filter image PSNR : ",psnr_value, "\t \t Gaussian filter image MSE : ",mse_value ,"\t \t Gaussian filter image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(filtered_image)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(filtered_image)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
   ax1.imshow(filtered_image, cmap='gray')
   ax1.set_title('Gaussian filter image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def Histogram_equalization(watermarked_img):
   if len((np.asarray(watermarked_img)).shape)==3:
      long=3
   else:
      long=1

   image=[]
   for channel in range(long):
      if long==3:
         img=watermarked_img[:, :, channel]
      else:
         img=watermarked_img
      max=np.max(img)

      filtered_image = exposure.equalize_hist(img)
      filtered_image = filtered_image *max
      filtered_image = filtered_image.astype("uint8")
      image.append(filtered_image)

   if long==3:
       filtered_image = np.stack([image[0], image[1], image[2]], axis=2)
   else:
       filtered_image=image[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img, filtered_image)           
   print("His-EQ image PSNR : ",psnr_value, "\t \t His-EQ image MSE : ",mse_value ,"\t \t His-EQ image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(filtered_image)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(filtered_image)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax0,ax1, ax2,ax3) = plt.subplots(1, 4)
   ax0.imshow(filtered_image, cmap='gray')
   ax0.set_title('His-EQ image')
   ax1.imshow(org_water, cmap='gray')
   ax1.set_title('Rec watermark')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def Low_pass(watermarked_img,kernel,co):
   blurred = cv2.GaussianBlur(watermarked_img, kernel, co)

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  blurred)           
   print("Low pass image PSNR : ",psnr_value, "\t \t Low pass image MSE : ",mse_value ,"\t \t Low pass image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(blurred)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( blurred)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(blurred, cmap='gray')
   ax1.set_title('Low pass image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()

def Motion_blur(watermarked_img,size,angle):
   img=np.copy(watermarked_img)
   # Create the motion blur kernel
   kernel = np.zeros((size, size))
   kernel[int((size-1)/2), :] = np.ones(size)
   kernel = rotate(kernel, angle, mode='constant')
   # Normalize the kernel
   kernel = kernel / np.sum(kernel)

   # Apply the motion blur filter to the image
   if len((np.asarray(watermarked_img)).shape)==3:
      long=3
   else:
      long=1

   image=[]
   for channel in range(long):
      if long==3:
         img=watermarked_img[:, :, channel]
      else:
         img=watermarked_img
      blurred_img = convolve(img, kernel)
      image.append(blurred_img)

   if long==3:
       blurred_img = np.stack([image[0], image[1], image[2]], axis=2)
   else:
       blurred_img=image[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  blurred_img)           

   print("Motion blur image PSNR : ",psnr_value, "\t \t Motion blur image MSE : ",mse_value ,"\t \t Motion blur image SSIM : ",ssim_value)
   if self_embed==True:      
      org_water=self_embedding(blurred_img)
   else:
      org_water=org_watermark
   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( blurred_img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(blurred_img, cmap='gray')
   ax1.set_title('Motion blur image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()


def Cropping_attack(watermarked_img,XD, YD, sizex, sizey):
   cropped_img=np.copy(watermarked_img)
   if len((np.asarray(cropped_img)).shape)==3:
      long=3
   else:
      long=1

   image=[]
   for channel in range(long):
      if long==3:
         img=cropped_img[:, :, channel]
      else:
         img=cropped_img
      for i in range(sizex):
         for j in range(sizey):
            img[XD+i][YD+j]=0
      image.append(img)

   if long==3:
       cropped_img = np.stack([image[0], image[1], image[2]], axis=2)
   else:
       cropped_img=image[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  cropped_img)           
   print("Cropped image PSNR : ",psnr_value, "\t \t Cropped image MSE : ",mse_value ,"\t \t Cropped image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(cropped_img)
   else:
      org_water=org_watermark
      
   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( cropped_img)
   tamper=Tamper_detection(org_water,ext_watermark)
   tamper=Tamper_localization(tamper)
   recovered_img=recovery_process(cropped_img,tamper,Rec_watermark1,Rec_watermark2 )

   fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
   ax1.imshow(cropped_img, cmap='gray')
   ax1.set_title('Cropped image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   ax4.imshow(recovered_img, cmap='gray')
   ax4.set_title('Recovered')
      
   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  recovered_img)           
   print("Recovered image PSNR : ",psnr_value, "\t \t Recovered image MSE : ",mse_value ,"\t \t Recovered image SSIM : ",ssim_value)
   plt.show()

def Copy_move_attack(watermarked_img,XD, YD, XD2, YD2, sizex, sizey):
   image=np.copy(watermarked_img)
   if len((np.asarray(image)).shape)==3:
      long=3
   else:
      long=1

   imag=[]
   for channel in range(long):
      if long==3:
         img=image[:, :, channel]
      else:
         img=image
      for i in range(sizex):
         for j in range(sizey):
            img[XD+i][YD+j]=img[XD2+i][YD2+j]
      imag.append(img)

   if long==3:
       Copy_move_img = np.stack([imag[0], imag[1], imag[2]], axis=2)
   else:
       Copy_move_img=imag[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  Copy_move_img)           
   print("Copy move image PSNR : ",psnr_value, "\t \t Copy move image MSE : ",mse_value ,"\t \t Copy move image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(Copy_move_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( Copy_move_img)
   tamper=Tamper_detection(org_water,ext_watermark)
   tamper=Tamper_localization(tamper)
   recovered_img=recovery_process(Copy_move_img,tamper,Rec_watermark1,Rec_watermark2 )

   fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
   ax1.imshow(Copy_move_img, cmap='gray')
   ax1.set_title('Copy move image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   ax4.imshow(recovered_img, cmap='gray')
   ax4.set_title('Recovered')
      
   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  recovered_img)           
   print("Recovered image PSNR : ",psnr_value, "\t \t Recovered image MSE : ",mse_value ,"\t \t Recovered image SSIM : ",ssim_value)
   plt.show()

def Copy_paste_attack(watermarked_img,XD, YD,sizex, sizey):
   image=np.copy(watermarked_img)
   img = imageio.imread(os.path.join(current_directory, 'Images/Baboon.tiff'))
   img=cv2.resize(img,(img_size,img_size))
   
   if len((np.asarray(image)).shape)==3:
      long=3
   else:
      long=1
      if len((np.asarray(img)).shape)==3:
         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   imag=[]
   for channel in range(long):
      if long==3:
         img1=image[:, :, channel]
         img2=img[:, :, channel]
      else:
         img1=image
         img2=img
      for i in range(sizex):
         for j in range(sizey):
            img1[XD+i][YD+j]=img2[XD+i][YD+j]
      imag.append(img1)
   
   if long==3:
       Copy_paste_img = np.stack([imag[0], imag[1], imag[2]], axis=2)
   else:
       Copy_paste_img=imag[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  Copy_paste_img)           
   print("Copy paste image PSNR : ",psnr_value, "\t \t Copy paste image MSE : ",mse_value ,"\t \t Copy paste image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(Copy_paste_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( Copy_paste_img)
   tamper=Tamper_detection(org_water,ext_watermark)
   tamper=Tamper_localization(tamper)
   recovered_img=recovery_process(Copy_paste_img,tamper,Rec_watermark1,Rec_watermark2)

   fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
   ax1.imshow(Copy_paste_img, cmap='gray')
   ax1.set_title('Copy_paste image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   ax4.imshow(recovered_img, cmap='gray')
   ax4.set_title('Recovered')
      
   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  recovered_img)           
   print("Recovered image PSNR : ",psnr_value, "\t \t Recovered image MSE : ",mse_value ,"\t \t Recovered image SSIM : ",ssim_value)
   plt.show()

def Text_addition(watermarked_img,text,x,y,size):
   image=np.copy(watermarked_img)
   if len((np.asarray(image)).shape)==3:
      long=3
   else:
      long=1

   imag=[]
   for channel in range(long):
      if long==3:
         img=image[:, :, channel]
      else:
         img=image
      texted_image = Image.fromarray(img)
      draw = ImageDraw.Draw(texted_image)
      font = ImageFont.truetype("arial.ttf", size)
      # Draw the text on the image
      draw.text((x, y), text, font=font)
      texted_image=np.array(texted_image)
      imag.append(texted_image)

   if long==3:
       texted_image = np.stack([imag[0], imag[1], imag[2]], axis=2)
   else:
       texted_image=imag[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  texted_image)           
   print("Text addition image PSNR : ",psnr_value, "\t \t Text addition image MSE : ",mse_value ,"\t \t Text addition image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(texted_image)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( texted_image)
   tamper=Tamper_detection(org_water,ext_watermark)
   recovered_img=recovery_process(texted_image,tamper,Rec_watermark1,Rec_watermark2 )

   fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
   ax1.imshow(texted_image, cmap='gray')
   ax1.set_title('Cropped image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   ax4.imshow(recovered_img, cmap='gray')
   ax4.set_title('Recovered')
      
   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  recovered_img)           
   print("Recovered image PSNR : ",psnr_value, "\t \t Recovered image MSE : ",mse_value ,"\t \t Recovered image SSIM : ",ssim_value)
   plt.show()

def Content_addition(watermarked_img,des_x, des_y ,sizex,sizey):
   image=np.copy(watermarked_img)
   img = imageio.imread(os.path.join(current_directory, 'Images/Addition.png'))
   zoomed_img=cv2.resize(img,(sizey,sizex)) 

   if len((np.asarray(image)).shape)==3:
      long=3
   else:
      long=1
      if len((np.asarray(zoomed_img)).shape)==3:
         zoomed_img=cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2GRAY)

   imag=[]
   for channel in range(long):
      if long==3:
         img1=image[:, :, channel]
         img2=zoomed_img[:, :, channel]
      else:
         img1=image
         img2=zoomed_img
      for i in range(sizex):
         for j in range(sizey):
            img1[des_x+i][des_y+j]=img2[i][j]
      imag.append(img1)
   if long==3:
       content_addition_img = np.stack([imag[0], imag[1], imag[2]], axis=2)
   else:
       content_addition_img=imag[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  content_addition_img)        
   print("Content addition image PSNR : ",psnr_value, "\t \t Content addition image MSE : ",mse_value ,"\t \t Content addition image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(content_addition_img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2=extraction_DWT_watermark( content_addition_img)
   tamper=Tamper_detection(org_water,ext_watermark)
   tamper=Tamper_localization(tamper)
   recovered_img=recovery_process(content_addition_img,tamper,Rec_watermark1,Rec_watermark2)

   fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
   ax1.imshow(content_addition_img, cmap='gray')
   ax1.set_title('Added image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   ax4.imshow(recovered_img, cmap='gray')
   ax4.set_title('Recovered')
      
   mse_value, psnr_value, ssim_value = Image_metrics(original_img,  recovered_img)         
   print("Recovered image PSNR : ",psnr_value, "\t \t Recovered image MSE : ",mse_value ,"\t \t Recovered image SSIM : ",ssim_value)
   plt.show()


def Constant_feature_attack(type,watermarked_img,XD, YD, xsize, ysize,MSB):
   image=np.copy(watermarked_img)
   if len((np.asarray(image)).shape)==3:
      long=3
   else:
      long=1

   imag=[]
   if type=="all":
    for channel in range(long):
      if long==3:
         img=image[:, :, channel]
      else:
         img=image
      for j in range(len(img[0])):
         for k in range(len(img[0])): 
            bits=""
            for t in range(MSB):
               pixel =dec_to_bin(img[j][k]) 
               bits=bits+str(random.randint(0,1))
               pixel=bits+pixel[MSB:len(pixel)]
               img[j][k]=bin_to_dec(pixel)
      imag.append(img)  

   elif type=="zone":
    for channel in range (long):
      if long==3:
         img=image[:, :, channel]
      else:
         img=image
      for j in range(xsize):
         for k in range(ysize):
            pixel =dec_to_bin(img[XD+j][YD+k])                                  
            bits=""
            for t in range(MSB):
               bits=bits+str(random.randint(0,1))
               pixel=bits+pixel[MSB:len(pixel)]
               img[XD+j][YD+k]=bin_to_dec(pixel)  
      imag.append(img)         
   else:
      raise ValueError("Selecet either 'all' or 'zone'")

   if long==3:
       img = np.stack([imag[0], imag[1], imag[2]], axis=2)
   else:
       img=imag[0]  

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,img)           
   print("Content feature image PSNR : ",psnr_value, "\t \t Content feature image MSE : ",mse_value ,"\t \t Content feature image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(img, cmap='gray')
   ax1.set_title('Content feature image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def Content_only_attack(type,watermarked_img,XD, YD,xsize, ysize,MSB):
   image=np.copy(watermarked_img)
   pic = imageio.imread(os.path.join(current_directory, 'Images/Peppers.tiff'))
   pic=cv2.resize(pic,(img_size,img_size))

   if len((np.asarray(image)).shape)==3:
      long=3
   else:
      long=1
      if len((np.asarray(pic)).shape)==3:
         pic=cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

   if type=="all":
      max_x=len(image[0])
      max_y=len(image[0])
      XD=0
      YD=0
   elif type=="zone":
      max_x=xsize
      max_y=ysize
   else :
      raise ValueError("Error , select either 'all' or 'zone'")

   imag=[]
   for channel in range(long):
      if long==3:
         img1=image[:, :, channel]
         img2=pic[:, :, channel]
      else:
         img1=image
         img2=pic
      for j in range(max_x):
         for k in range(max_y):    
            pixel =dec_to_bin(img1[XD+j][YD+k]) 
            pixel2 =dec_to_bin(img2[XD+j][YD+k])
            pixel=pixel2[0:MSB]+pixel[MSB:len(pixel)]                  
            img1[XD+j][YD+k]=bin_to_dec(pixel)
      imag.append(img1)

   if long==3:
       img = np.stack([imag[0], imag[1], imag[2]], axis=2)
   else:
       img=imag[0]

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,img)           
   print("Content only image PSNR : ",psnr_value, "\t \t Content only image MSE : ",mse_value ,"\t \t Content only image SSIM : ",ssim_value)
   
   if self_embed==True:      
      org_water=self_embedding(img)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( img)
   tamper=Tamper_detection(org_water,ext_watermark)

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(img, cmap='gray')
   ax1.set_title('Content only image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')      
   plt.show()

def VQ_attack(watermarked_img):
   # Convert image to a 2D array of pixels
   if len((np.asarray(watermarked_img)).shape)==3:
      long=3
      VQ_img = np.reshape(watermarked_img, (img_size*img_size, 3))
   else:
      long=1
      VQ_img = np.reshape(watermarked_img, (img_size*img_size,1))

   # Apply k-means clustering to the pixel data
   kmeans = KMeans(n_clusters=16, random_state=0,n_init=10).fit(VQ_img)
   codebook = kmeans.cluster_centers_

   # Replace each pixel with its nearest codevector in the codebook
   quantized = codebook[kmeans.labels_]
   quantized = np.reshape(quantized, (VQ_img.shape[0], VQ_img.shape[1]))
   if long==3:
      quantized=quantized.reshape(img_size,img_size,3)
      quantized = quantized.astype("uint8")
   else:
      quantized=quantized.reshape(img_size,img_size)
      quantized = quantized.astype("uint8")

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,quantized)           
   print("VQ image PSNR : ",psnr_value, "\t \t VQ image MSE : ",mse_value ,"\t \t VQ image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(quantized)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark( quantized)
   tamper=Tamper_detection(org_water,ext_watermark)
   
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(quantized, cmap='gray')
   ax1.set_title('VQ image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   plt.show()

def Constant_small_noise_addition(watermarked_img, num_steps, delta):   #add a number to all the pixels , tamper is detected only when embedding in LL , not detected in high fresquency beacuse all pixels are incresed by the same value
   image=np.copy(watermarked_img)

   for i in range(num_steps):
      # Modify the image by adding the delta to each pixel value
      image = np.clip(image + delta, 0, 255).astype(np.uint8)

   mse_value, psnr_value, ssim_value = Image_metrics(original_img,image)         
   print("Timeline image PSNR : ",psnr_value, "\t \t Timeline image MSE : ",mse_value ,"\t \t Timeline image SSIM : ",ssim_value)

   if self_embed==True:      
      org_water=self_embedding(image)
   else:
      org_water=org_watermark

   ext_watermark,Rec_watermark1,Rec_watermark2 =extraction_DWT_watermark(image)
   tamper=Tamper_detection(org_water,ext_watermark)

   """if len((np.asarray(watermarked_img)).shape)==3:
      max1=np.max(image[:][:][0])
      max2=np.max(image[:][:][1])
      max3=np.max(image[:][:][2])
      max_v=max(max1,max2,max3)
      imag=image/max_v
   else:
      maxi=np.max(image[:][:])
      imag=image/maxi"""

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
   ax1.imshow(image, cmap='gray')
   ax1.set_title('Timeline image')
   ax2.imshow(ext_watermark, cmap='gray')
   ax2.set_title('Attacked watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')

   mse_value, psnr_value, ssim_value = Image_metrics(org_water, ext_watermark)           
   print("Extracted Watermark PSNR : ",psnr_value, "\t \t Extracted Watermark MSE : ",mse_value ,"\t \t Extracted Watermark SSIM : ",ssim_value)
   plt.show()

