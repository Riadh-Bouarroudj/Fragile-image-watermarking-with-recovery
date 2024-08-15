# Fragile-image-watermarking-with-recovery
# Description and explanation
With the widespread availability of powerful image manipulation software, achieving authentication and anti-counterfeiting for digital images is increasingly important. This code provides a solution for digital image authentication, tamper localization, and content recovery using fragile watermarking.

Fragile watermarking is a data-hiding technique that conceals information, known as the watermark within an image (referred to as the host or cover image) without causing significant distortion to the original image. The resulting image, termed the watermarked image, is then transmitted to the receiver, who can extract the watermark to determine whether the image has undergone any alteration. If the watermarked image is found to be altered, the tampered areas will be accurately highlighted, and the recovery process will take place to restore the altered regions, resulting in a recovered image free from significant tampering despite the encountered alterations.

![Untitled Diagram (18)](https://github.com/user-attachments/assets/9e26cc86-c5e9-440e-b944-16357015ddda)

In the proposed solution, one authentication watermark for tamper localization and two recovery watermarks for content recovery are embedded within the cover image using the Least Significant Bit (LSB) substitution technique. These three watermarks are generated from the cover image to keep the process fully blind, meaning that no information need to be stored on the receiver side (except for the decryption keys). Moreover, to enhance the model’s security, the authentication watermark is encrypted using the Henon map, while the two recovery watermarks are scrambled using the Arnold transformation. This scrambling operation allows the recovery information of a block to be stored in another block to perform recovery in case of tampering.

To assess the quality of the watermarked and recovered images produced by the model, three image quality metrics are employed: Mean Square Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index Measure (SSIM). Additionally, the Bit Error Rate (BER) metric is used to evaluate the accuracy of the watermark extraction procedure.

# Structure and important details
- This code works with both color and grayscale images with a bit depth of 8 bits (pixel values ranging between 0 and 255). If you want to work with images with a bit depth greater than 8 or containing negative values, the code will need some adjustments.
- The file "Attacks.py" contains the attacks used to evaluate the proposed method robustness, and two functions ("TIFF_save" and "PNG_save") for saving the watermarked image. It is worth noting that using "JPEG_save" is considered an attack since the watermarked image will undergo compression, which is not possible when using Fragile watermarking models.
- The file "config_variables.py" contains all the inputs that can be changed, such as the cover image path, the image size, the block size, the embedding capacity (BPP), etc. I recommend using (bloc_size=2 and BPP=2) for better recovered image quality, or (bloc_size=3 and BPP=1) for better watermarked image quality.
- "Functions.py" contains all the essential functions used in this method, including embedding, extraction recovery, etc.
- The method can be launched from "Main.py", which contains an example of the embedding and extraction procedures, as well as the attacks performed to evaluate the model’s robustness.

# Data citation
If you find this code to be useful for your scientific research, please cite some of the following papers associated with this code:
- Riadh Bouarroudj, Feryel Souami, Fatma Zohra Bellala, Nabil Zerrouki, A reversible fragile watermarking technique using fourier transform and Fibonacci Q-matrix for medical image authentication, Biomedical Signal Processing and Control, Volume 92, 2024, 105967. https://doi.org/10.1016/j.bspc.2024.105967
- R. Bouarroudj, F. Souami and F. Z. Belalla, "Reversible Fragile Watermarking for Medical Image Authentication in the Frequency Domain," 2023 2nd International Conference on Electronics, Energy and Measurement (IC2EM), Medea, Algeria, 2023, pp. 1-6. https://doi.org/10.1109/IC2EM59347.2023.10419699
- R. Bouarroudj, F. Souami and F. Z. Bellala, "Fragile watermarking for medical image authentication based on DCT technique," 2023 5th International Conference on Pattern Analysis and Intelligent Systems (PAIS), Sétif, Algeria, 2023, pp. 1-6. https://doi.org/10.1109/PAIS60821.2023.10322029

# Image Sources
The Datasets used to evaluate the proposed method’s performance can be accessed via the following links:
- The USC-SIPI Image Database: https://sipi.usc.edu/database/database.php
- STructured Analysis of the Retina: https://cecas.clemson.edu/~ahoover/stare/
- Chest CT-Scan images Dataset: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
- Brain Tumor MRI Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- COVID-19 Radiography Database: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Ultrasound image database: http://splab.cz/en/download/databaze/ultrasound
