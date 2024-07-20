import numpy as np
import pywt
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from config_variables import *


#Arnold transform
def scramble_image(image, iterations):
    #image=np.array(image)
    n, m = image.shape
    transformed_image = np.zeros((n, m),dtype="uint8")
    for x in range(n):
        for y in range(m):
            transformed_image[(2*x + y) % n][(x + y) % m] = image[x][y]
    if iterations > 1:
        return scramble_image(transformed_image, iterations - 1)
    else:
        return transformed_image

def unscramble_image(image, iterations):
    #image=np.array(image)
    n, m = image.shape
    transformed_image = np.zeros((n, m),dtype="uint8")
    for x in range(n):
        for y in range(m):
            transformed_image[x][y]= image[(2*x + y) % n][(x + y) % m]
    if iterations > 1:
        return unscramble_image(transformed_image, iterations - 1)
    else:
        return transformed_image

#Henon map
def encrypt_image(image, key):
    encrypted_image = np.copy(image)
    height, width = image.shape
    
    # Initialize the Hanon map with the given key
    x, y = key[0], key[1]
    
    for i in range(height):
        for j in range(width):
            # Generate pseudo-random numbers using the Hanon map
            x = 1 - key[0] * x**2 +  y
            y = key[1] *x
            
            # Encrypt the pixel value using XOR operation
            encrypted_image[i, j] = image[i, j] ^ int(x * 255)
    
    return encrypted_image

def decrypt_image(encrypted_image, key):
    decrypted_image = np.copy(encrypted_image)
    height, width = encrypted_image.shape
    
    # Initialize the Hanon map with the given key
    x, y = key[0], key[1]
    
    for i in range(height):
        for j in range(width):
            # Generate pseudo-random numbers using the Hanon map
            x = 1 - key[0] * x**2 +  y
            y = key[1] *x
            
            # Decrypt the pixel value using XOR operation
            decrypted_image[i, j] = encrypted_image[i, j] ^ int(x * 255)
    
    return decrypted_image


def Image_metrics(img1, img2):
   if len((np.asarray(img1)).shape)==3:
      long=3
   else:
      long=1

   result1=[]
   result2=[]
   result3=[]
   for channel in range (long):
      if long==3:
         im1=img1[:, :, channel]
         im2=img2[:, :, channel]
      else:
         im1=img1
         im2=img2
      
      msee=0
      for i in range(len(im1)):
         for j in range(len(im1)):
            if im1[i][j] >= im2[i][j]:
               msee = msee+int(((im1[i][j] - im2[i][j]) ** 2))
            else:
               msee = msee+int(((im2[i][j] - im1[i][j]) ** 2))
      mse=msee/(len(im1)*len(im1))
      result1.append(mse)
      
      if mse!=0:     
         psnr = 20 * math.log10(255 / math.sqrt(mse))
      else:
         psnr=400
      result2.append(psnr)

      ssim_value = ssim(im1,im2,data_range=255.0) 
      result3.append(ssim_value)

   if long==3:
      mse=(result1[0]+result1[1]+result1[2])/3
      psnr=(result2[0]+result2[1]+result2[2])/3
      ssim_v=(result3[0]+result3[1]+result3[2])/3
   else:
      mse=result1[0]
      psnr=result2[0]
      ssim_v=result3[0]
   
   if psnr==400:
      psnr="Inf"
   return mse, psnr, ssim_v

def dec_to_bin(n):
    x=bin(n).replace("0b", "")
    while len(x)<8 :
      x="0"+x 
    return (x)
    
def bin_to_dec(n):
    return int(n, 2)

def watermark_to_digit(wat):
  #Transform wat to 1D array
  wat = np.array(wat)
  wat = wat.flatten()
  #Put all the watermark bits in one sequence
  watermark=""
  for i in range (len(wat)):
         bi=dec_to_bin(wat[i])   
         watermark=watermark+bi
  return(watermark)


def embedding_DWT_watermark(cover,org_watermark) :
  if len((np.asarray(cover)).shape)==3:
     long=3
  else:
     long=1

  lis=[]
  for channel in range (long):      #Normalize the cover image to the range of [4,251] to avoid overfloaw and underflow problems
      if long==3:
        img=cover[:, :, channel]
      else:
        img=cover
      for i in range (img_size):
         for j in range (img_size):
            if img[i][j]<4:
               img[i][j]=4
            elif img[i][j]>251:
               img[i][j]=251
      lis.append(img)
  if long==3:
       cover = np.stack([lis[0], lis[1], lis[2]], axis=2)
  else:
       cover=lis[0]

  # Load the watermarks
  if self_embed==True:
      Auth_wat=self_embedding(cover)
  else:
      Auth_wat=org_watermark

  Rec_wat=Recovery_watermark_construction(cover,long)

  Auth_arr=[]
  Rec_arr1=[]
  Rec_arr2=[]
  #Prepare the watermarks and transform them to long binary digits
  for channel in range(long):
    if long==3:
        Auth=Auth_wat[:, :, channel]
        Rec=Rec_wat[:, :, channel]
    else:
        Auth=Auth_wat
        Rec=Rec_wat
    if Auth_encryption==True:
        Auth = encrypt_image(Auth, key)
    if Rec_scrambling==True:
        Rec1 = scramble_image(Rec, key1)
        Rec2 = scramble_image(Rec, key2)   
    else:
        Rec1 = Rec
        Rec2 = Rec   
    Auth = watermark_to_digit(Auth)
    Rec1 = watermark_to_digit(Rec1)
    Rec2 = watermark_to_digit(Rec2)
    Auth_arr.append(Auth)
    Rec_arr1.append(Rec1)
    Rec_arr2.append(Rec2)

  
  w_comp_arr=[]
  #Loop on the RGB channels of the cover image
  for channel in range (long): 
   if long==3:
      watermarked_img=cover[:, :, channel]
   else:
      watermarked_img=np.copy(cover)
   Auth_watermark=Auth_arr[channel]
   Rec_watermark1=Rec_arr1[channel]
   Rec_watermark2=Rec_arr2[channel]

   # Apply Discrete wavelet transform to the cover image
   coeffs = pywt.dwt2(watermarked_img, 'haar')             
   LL, (LH, HL, HH) = coeffs
   lis=["LH","HL","HH"]
   
   #Loop on the frequency subbands of the image
   for subb in lis:                
    if subb=="LH" : subband=LH; watermark=Rec_watermark1
    elif subb=="HL" : subband=HL; watermark=Rec_watermark2
    elif subb=="HH" : subband=HH; watermark=Auth_watermark
    a=0
    #Round the coefficient values to 5 numbers after the decimal point to avoid problems caused by DWT-IDWT 
    subband=np.round(subband,5)
    #Loop on each subband  
    while a+bloc_size<=int(len(subband)): 
     b=0
     while b+bloc_size<=int(len(subband)):               
         v=0   
         #Loop on each bloc    
         for j in range(bloc_size):
            for k in range(bloc_size):
               if  v*BPP==8: break        # Ensure that 8 bits are embedded in each block to provide tamper localization           
               neg=False  
               # Given that the coefficients values can be negative, transform them to positive values for binary transformation                                      
               if subband[a+j][b+k]<0 :                                      
                  subband[a+j][b+k]=subband[a+j][b+k]*-1    
                  neg=True                                      
               dec_part=subband[a+j][b+k]% 1
               int_part =int(subband[a+j][b+k])
               pixel =dec_to_bin(int_part) 

               #Watermark bits embedding
               bits=str(watermark[0:BPP])
               watermark=watermark[BPP:len(watermark)]          
               pixel=pixel[0:len(pixel)-BPP]+bits  

               #Pixel quality adjustement fo better watermarked image quality
               if BPP>1:
                  qu=pixel[len(pixel)-1-BPP]
                  if qu=='1':
                        qu='0'
                  else:
                        qu='1'
                  pixel_qu=pixel[0:len(pixel)-BPP-1]+qu+pixel[len(pixel)-BPP:len(pixel)]
                  pixel=bin_to_dec(pixel)
                  pixel2=bin_to_dec(pixel_qu)
                  if abs(int_part-pixel2)<abs(int_part-pixel):
                     pixel=pixel2
               else:
                  pixel=bin_to_dec(pixel)

               #Update the watermarked coefficient
               subband[a+j][b+k]=pixel+dec_part
               if neg==True : 
                  subband[a+j][b+k]=subband[a+j][b+k]*-1   
               v=v+1
         b=b+bloc_size
     a=a+bloc_size

    if subb=="LL" : LL=subband
    elif subb=="LH" : LH=subband
    elif subb=="HL" : HL=subband
    elif subb=="HH" : HH=subband
   #Apply inverse DWT to the watermarked subbands
   watermarked_coeffs = LL, (LH, HL, HH)
   watermarked = pywt.idwt2(watermarked_coeffs, 'haar') 
   
   #Convert the watermarked channel to integer values
   for i in range (img_size):
      for j in range (img_size):
         p=watermarked[i][j] % 1
         if  p >0.6 :                                           
            watermarked_img[i][j]=int(watermarked[i][j])+1
         else :
            watermarked_img[i][j]=int(watermarked[i][j])   
   w_comp_arr.append(watermarked_img)
   #print(np.min(watermarked_img),np.max(watermarked_img))

  if long==3:
      watermarked_img =np.stack([w_comp_arr[0], w_comp_arr[1], w_comp_arr[2]], axis=2)
  else:
      watermarked_img=w_comp_arr[0]

  return(watermarked_img) 
 

def extraction_DWT_watermark(imagex):
    if len((np.asarray(imagex)).shape)==3:
       long=3
    else:
       long=1
    image=np.copy(imagex)

    FAuth_watermark=[]
    FRec_watermark1=[]
    FRec_watermark2=[]

    #Loop on the watermarked image channels
    for channel in range (long): 
     if long==3:
         image=imagex[:, :, channel]
     else:
        image=imagex
     Auth_watermark=[]
     Rec_watermark1=[]
     Rec_watermark2=[]
     # Apply Discrete wavelet transform to the channel
     coeffs = pywt.dwt2(image, 'haar')             
     LL, (LH, HL, HH) = coeffs
     lis=["LH","HL","HH"]

     #Loop on the image subbands
     for subb in lis:                  
      if subb=="LL" : subband=LL
      elif subb=="LH" : subband=LH
      elif subb=="HL" : subband=HL
      elif subb=="HH" : subband=HH
      #Round the coefficient values to 5 numbers after the decimal point to avoid problems caused by DWT-IDWT 
      subband=np.round(subband,5)
      #Loop on each frequency subband
      a=0
      while a+bloc_size<=len(subband):       
        b=0
        while b+bloc_size<=len(subband):
            wat=""
            v=0
            #Loop on each block
            for j in range(bloc_size):
               for k in range(bloc_size):
                  #Stop if 8 bits are extracted from the current block or if all the watermark bits have been extracted
                  if v*BPP==8 or len(Auth_watermark)==wat_size*wat_size:  break             
                  # Given that the coefficients values can be negative, transform them to positive values for binary transformation 
                  if subband[a+j][b+k]<0 :                                     
                        subband[a+j][b+k]=subband[a+j][b+k]*-1                                                
                  #Watermark bits extraction
                  int_part =int(subband[a+j][b+k])
                  pixel =dec_to_bin(int_part)  
                  wat=wat+pixel[len(pixel)-BPP:len(pixel)]
 
                  # if 8 bits have been extracted from the current block, append them to their corresponding subband
                  if len(wat)==8:          
                     wat=bin_to_dec(wat)  
                     if subb=="LH":
                        Rec_watermark1.append(wat)
                     elif subb=="HL":
                        Rec_watermark2.append(wat)
                     elif subb=="HH":
                        Auth_watermark.append(wat)
                     wat=""
                  v=v+1        
            b=b+bloc_size
        a=a+bloc_size

     #Reconstruct and decrypt the extracted watermarks 
     Auth_watermark=np.array(Auth_watermark)
     Auth_watermark=Auth_watermark.reshape(wat_size,wat_size)
     Auth_watermark = np.array(Auth_watermark.astype("uint8"))
     Rec_watermark1=np.array(Rec_watermark1)
     Rec_watermark1=Rec_watermark1.reshape(wat_size,wat_size)
     Rec_watermark1 = Rec_watermark1.astype("uint8")
     Rec_watermark2=np.array(Rec_watermark2)
     Rec_watermark2=Rec_watermark2.reshape(wat_size,wat_size)
     Rec_watermark2 = Rec_watermark2.astype("uint8")
     if Auth_encryption==True:
        Auth_watermark = decrypt_image(Auth_watermark, key)
     if Rec_scrambling==True:
        Rec_watermark1 = unscramble_image(Rec_watermark1, key1)
        Rec_watermark2 = unscramble_image(Rec_watermark2, key2)
     FAuth_watermark.append(Auth_watermark)
     FRec_watermark1.append(Rec_watermark1)
     FRec_watermark2.append(Rec_watermark2)

    if long==3:
      Auth_watermark =  np.stack([FAuth_watermark[0], FAuth_watermark[1], FAuth_watermark[2]], axis=2)
      Rec_watermark1 =  np.stack([ FRec_watermark1[0],  FRec_watermark1[1], FRec_watermark1[2]], axis=2)
      Rec_watermark2 =  np.stack([ FRec_watermark2[0],  FRec_watermark2[1], FRec_watermark2[2]], axis=2)
    else:
      Auth_watermark =  FAuth_watermark[0]
      Rec_watermark1 =  FRec_watermark1[0]
      Rec_watermark2 =  FRec_watermark2[0]
    return(Auth_watermark,Rec_watermark1,Rec_watermark2)
   

def self_embedding(imagex):
   img=np.copy(imagex)
   if len((np.asarray(img)).shape)==3:
     long=3
   else:
     long=1
   ww_arr=[]
   for channel in range (long): 
      if long==3:
         image=img[:, :, channel]
      else:
         image=img
      #In case of 12bit or 16bit image, normalize the image to to an 8-bit image
      if np.max(np.abs(image))>256:
         if np.max(np.abs(image))<4096:
            maaax=4095
         else: 
            maaax=65535
         img_norm = (image/ maaax) * 255
         image=np.round(img_norm,0)   
         if np.min(image)<0:
            image = (image + 255) /2 
  
      if embedding_type=='DWT':
         coeffs = pywt.dwt2(image, 'haar')
         LL, (LH, HL, HH) = coeffs
         LL = (LL /600) * 255
         image=LL

      down_size=int(len(image[0])/wat_size)
      watermark = np.array([[0 for j in range(wat_size)] for i in range(wat_size)], dtype='uint8')
      ii=0
      i=0 
      while i+down_size < len(image):
         j=0
         jj=0
         while j+down_size <len(image):
             s=0
             for k in range(down_size):
              for m in range(down_size):
                 s=s+image[i+k][j+m]
             sum=s/(down_size*down_size)
             watermark[ii][jj]=int(round(sum,0))
             jj=jj+1
             j=j+down_size
         i=i+down_size
         ii=ii+1
      ww_arr.append(watermark)

   if long==3:
      watermark =  np.stack([ww_arr[0], ww_arr[1], ww_arr[2]], axis=2)
   else:
      watermark =  ww_arr[0]
   return(watermark)

def Recovery_watermark_construction(img,long):   
    comp_arr=[]
    for channel in range (long):
        if long==3:
            image=img[:, :, channel]
        else:
            image=img
        down_s=int(len(image[0])/wat_size)
        bloc = np.array([[0 for j in range(wat_size)] for i in range(wat_size)], dtype='uint8')
        ii=0
        i=0 
        while i+down_s <= len(image[0]):
            j=0
            jj=0
            while j+down_s <=len(image[0]):
                s=0
                for k in range(down_s):
                    for m in range(down_s):
                        s=s+image[i+k][j+m]
                sum=s/(down_s*down_s)
                bloc[ii][jj]=int(round(sum,0))
                jj=jj+1
                j=j+down_s
            i=i+down_s
            ii=ii+1 
        comp_arr.append(bloc)
    if long==3:
       watermarked =  np.stack([comp_arr[0], comp_arr[1], comp_arr[2]], axis=2)
    else:
       watermarked=comp_arr[0]
    return(watermarked)
   
 
def Tamper_detection(org_watermar,ext_watermar): 
   #Given that the embedding is done bit-by-bit, val represents the number of different bits we tolerate between the binary reprentation of two pixels
   val=0  
   total=0
   if len((np.asarray(org_watermar)).shape)==3:
     long=3
   else:
     long=1

   t_arr=[]
   # 0 for altered pixels and 1 for unaltered ones
   tamper = np.array([[0 for j in range(wat_size)] for i in range(wat_size)], dtype='uint8')
   for channel in range (long): 
      if long==3:
         og_watermark=org_watermar[:, :, channel]
         ex_watermark=ext_watermar[:, :, channel]
      else:
         og_watermark=org_watermar
         ex_watermark=ext_watermar
      for i in range(wat_size):
         for j in range(wat_size):
            if og_watermark[i][j]>ex_watermark[i][j]:
               diff=og_watermark[i][j]-ex_watermark[i][j]
            else:
               diff=ex_watermark[i][j]-og_watermark[i][j]
            #We use this thereshold only when the authentication watermark is genrated from the cover image, otherwise no need to use a threshold of 3
            if (diff<3) and (self_embed==True):  
                  tamper[i][j]=1 
            else : 
                  pixel=dec_to_bin(int(og_watermark[i][j]))
                  pixel2=dec_to_bin(int(ex_watermark[i][j]))
                  sum=0
                  for k in range(len(pixel)):
                     if pixel[k]!=pixel2[k]:
                        sum=sum+1
                  if sum>val:
                     tamper[i][j]=0
                     total=total+sum
                  else:
                     tamper[i][j]=1
      t_arr.append(tamper)
   
   final_tamper = np.copy(tamper)
   if long==3:
      for i in range (wat_size):
         for j in range (wat_size):
            if t_arr[0][i][j]==1 and t_arr[1][i][j]==1 and t_arr[2][i][j]==1:
               final_tamper[i][j]=1
            else:
               final_tamper[i][j]=0
   else:
      final_tamper=t_arr[0]

   BER=total/(wat_size*wat_size*8)/long*100
   print("Bit error rate BER: ",BER,"%")
   return(final_tamper)

def Tamper_localization(tamper):
   #Perform a mojority vote between neighboords, if a pixel is unaltered by 4 or more of its neighboords are altered, the pixel is considered altered
   tamperx=np.copy(tamper)
   for i in range(wat_size):
      for j in range(wat_size):
         #Ensure that the pixel is not an edge pixel to perform majority vote
         if tamperx[i][j]==1 and i>0 and i<wat_size-1 and j>0 and j<wat_size-1:
            som=0
            ii=-1
            while ii<=1:
               jj=-1
               while jj<=1:
                  if tamperx[i+ii][j+jj]==0:
                     som=som+1
                  jj=jj+1  
               ii=ii+1
            if som>=4:
               tamper[i][j]=0

   #Can use dilatation and erosion operations, but the accuracy is not optimal
   #tamper = cv2.dilate(tamper, (3,3), iterations=1)
   #tamper = cv2.erode(tamper, (3,3), iterations=1)
   return(tamper)

def recovery_process(imagex,tamper,Rec_watermark1,Rec_watermark2):   
   if not any(0 in row for row in tamper):
      raise ValueError("No Tampering detected, Recovery impossible")
   
   image=np.copy(imagex)
   posttamp=np.copy(tamper)
   if len((np.asarray(image)).shape)==3:
      long=3
   else :
      long=1
   
   det1 = unscramble_image(posttamp, key1)
   det2 = unscramble_image(posttamp, key2)
   numb=int(img_size/wat_size)

   for i in range(wat_size):
      for j in range(wat_size):
         if posttamp[i][j]==0:
            if det1[i][j]==1 or det2[i][j]==1:
               if det1[i][j]==1:
                  pixel=Rec_watermark1[i][j]
               elif det2[i][j]==1:
                  pixel=Rec_watermark2[i][j]

               
               for a in range (numb):
                  for b in range (numb):      
                     image[i*numb+a][j*numb+b]=pixel
               posttamp[i][j]=1
   rec_img=np.copy(image)
   
   #Impainting method
   while any(0 in row for row in posttamp):
    image=np.copy(rec_img)
    for i in range(wat_size):
      for j in range(wat_size):
         if posttamp[i][j]==0:
            som1=0; som2=0; som3=0; n=0
            a=-1 
            while a<2:
               b=-1
               while b<2:
                  if a==0 and b==0: b=1     #Skip the current pixel
                  if posttamp[i+a][j+b]==1:
                     sum11=0; sum22=0; sum33=0
                     if long==3:
                        for aa in range(numb):
                           for bb in range(numb):
                              sum11=sum11+image[(i+a)*numb+aa][(j+b)*numb+bb][0]
                              sum22=sum22+image[(i+a)*numb+aa][(j+b)*numb+bb][1]
                              sum33=sum33+image[(i+a)*numb+aa][(j+b)*numb+bb][2]
                        som1=som1+sum11/(numb*numb)
                        som2=som2+sum22/(numb*numb)
                        som3=som3+sum33/(numb*numb)
                     else:
                        for aa in range(numb):
                           for bb in range(numb):
                              sum11=sum11+image[(i+a)*numb+aa][(j+b)*numb+bb]
                        som1=som1+sum11/(numb*numb)
                     n=n+1
                  b=b+1
               a=a+1  
            if n>=3:
               posttamp[i][j]=1
               for d in range (numb):
                  for f in range (numb): 
                     if long==3:    
                        image[i*numb+d][j*numb+f][0]=som1/n
                        image[i*numb+d][j*numb+f][1]=som2/n
                        image[i*numb+d][j*numb+f][2]=som3/n
                     else:
                        image[i*numb+d][j*numb+f]=som1/n
    rec_img=np.copy(image)  
   return(rec_img)            

def Display_watermarked_image(original_img,watermarked_img):
   fig, (ax1, ax2,) = plt.subplots(1, 2)
   ax1.imshow(original_img,cmap='gray')
   ax1.set_title('Original image')
   ax2.imshow(watermarked_img,cmap='gray')
   ax2.set_title('Watermarked image')
   plt.show()
   
   mse_value, psnr_value, ssim_value = Image_metrics(original_img, watermarked_img)
   print("Watermarked image PSNR : ",psnr_value,"\t \t Watermarked image MSE : ",mse_value,"\t \t Watermarked image SSIM : ",ssim_value)

def Display_watermark(org_water,Auth_watermark, Rec_watermark1,Rec_watermark2):
   tamper=Tamper_detection(org_water,Auth_watermark)

   fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
   ax1.imshow(org_water,cmap='gray')
   ax1.set_title('Original watermark')
   ax2.imshow(Auth_watermark,cmap='gray')
   ax2.set_title('Extracted watermark')
   ax3.imshow(tamper, cmap='binary',vmin=0, vmax=1)
   ax3.set_title('Tampering')
   ax4.imshow(Rec_watermark1,cmap='gray')
   ax4.set_title('Recovery 1')
   ax5.imshow(Rec_watermark2,cmap='gray')
   ax5.set_title('Recovery 2')
   
   mse_value, psnr_value, ssim_value = Image_metrics(org_water, Auth_watermark)
   print("Extracted Watermark PSNR : ",psnr_value, "\t \t Extracted Watermark MSE : ",mse_value ,"\t \t Extracted Watermark SSIM : ",ssim_value)
   plt.show()

