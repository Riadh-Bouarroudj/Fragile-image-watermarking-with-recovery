from functions import *
from attacks import *
from config_variables import *

def main():
      if len((np.asarray(original_img)).shape)==3:
            long=3
      elif len((np.asarray(original_img)).shape)==2:
            long=1
      print("Number of channels:", long)
      watermarked_img=embedding_DWT_watermark(original_img,org_watermark)
      Display_watermarked_image(original_img,watermarked_img)

      if self_embed==True:      
            org_water=self_embedding(watermarked_img)
      else:
            org_water=np.copy(org_watermark)
      ext_watermark,Rec_watermark1,Rec_watermark2=extraction_DWT_watermark(watermarked_img)
      Display_watermark(org_water,ext_watermark,Rec_watermark1,Rec_watermark2)


      """ Attacks : """
      Text_addition(watermarked_img,'Attacked image',50,220,60)
      Cropping_attack(watermarked_img,180,180,sizex=160,sizey=160)  
      Copy_move_attack(watermarked_img,160,260, 140,140,sizex=100,sizey=160)
      Copy_paste_attack(watermarked_img,200,280,sizex=100,sizey=88)
      Content_addition(watermarked_img,80,200,60,100)


      TIFF_save(watermarked_img)
      PNG_save(watermarked_img)  
      JPEG_save(watermarked_img, 80) 

      Rotation_attack(watermarked_img,30)
      Flip_direction_attack(watermarked_img,0)
      Translation_attack(watermarked_img,-20,-100)   
      Zooming_attack(watermarked_img,100,115,sizex=315,sizey=335)
      Scanning_attack(watermarked_img,"rotation",rot=30)
      Scanning_attack(watermarked_img,"scalling",size=256)

      Gaussian_noise(watermarked_img,0,10)
      Salt_paper(watermarked_img,0.5,0.05)
      Speckle_noise(watermarked_img)
      Median_filtring(watermarked_img,3)
      Average_filtring(watermarked_img,3)
      Gaussian_filtring(watermarked_img)
      Histogram_equalization(watermarked_img)                          
      Low_pass(watermarked_img,(9,9),0)
      Motion_blur(watermarked_img,size=15,angle=45)      

      Constant_feature_attack("all",watermarked_img,400,0,xsize=50,ysize=50,MSB=1)     
      Constant_feature_attack("zone",watermarked_img,400,0,xsize=50,ysize=50,MSB=1)
      Content_only_attack("all",watermarked_img,100,100,xsize=100, ysize=100,MSB=1)
      Content_only_attack("zone",watermarked_img,100,100,xsize=100, ysize=100,MSB=1)
      VQ_attack(watermarked_img)
      Constant_small_noise_addition(watermarked_img,2, 3)    

if __name__ == "__main__":
    main()
   
 






   


