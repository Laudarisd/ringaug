@echo off
setlocal

REM ================= User Settings =================
REM Input paths
set ORIGINAL_IMG_DIR=.\seg-topo-augment\images
set ORIGINAL_JSON_DIR=.\seg-topo-augment\json

REM Output paths
set OUTPUT_DIR=.\seg-topo-augment\augmented

REM Number of augmented images per source image
set NUM_PER_IMAGE=2

REM ----------- Augmentation Ranges -----------
REM crop scale range: 0.1 to 1.0
set CROP_SCALE_MIN=0.80
set CROP_SCALE_MAX=0.90

REM rotation angle range in degrees: -180 to 180
set ANGLE_MIN=-30
set ANGLE_MAX=30

REM affine scale range: >0
set SCALE_MIN=0.70
set SCALE_MAX=1.30

REM translation percent range: -1.0 to 1.0
set TRANSLATE_MIN=-0.10
set TRANSLATE_MAX=0.10

REM brightness/contrast delta range: usually -1.0 to 1.0
set BRIGHTNESS_MIN=-0.10
set BRIGHTNESS_MAX=0.10
set CONTRAST_MIN=-0.10
set CONTRAST_MAX=0.10

REM ----------- Probabilities (0.0 to 1.0) -----------
set P_ROTATE=0.90
set P_FLIP_H=0.20
set P_FLIP_V=0.10
set P_AFFINE=0.80
set P_CROP=0.70
set P_BRIGHTNESS=0.40

REM ----------- Topology/Contour Controls -----------
set CONTOUR_SIMPLIFY_EPSILON=1.5
set MIN_COMPONENT_AREA=12.0
set MIN_MASK_PIXEL_AREA=12.0
set RANDOM_AUG_PER_IMAGE=3

REM Debug mode: set to --debug to enable, keep empty to disable
set DEBUG_FLAG=
REM ================= End Settings =================

echo ========================================
echo Running Seg-TOPO augmentation
echo Input images: %ORIGINAL_IMG_DIR%
echo Input json:   %ORIGINAL_JSON_DIR%
echo Output root:  %OUTPUT_DIR%
echo Num/image:    %NUM_PER_IMAGE%
echo ========================================

python main.py ^
  --original-img-dir "%ORIGINAL_IMG_DIR%" ^
  --original-json-dir "%ORIGINAL_JSON_DIR%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --num-per-image %NUM_PER_IMAGE% ^
  --crop-scale-min %CROP_SCALE_MIN% ^
  --crop-scale-max %CROP_SCALE_MAX% ^
  --angle-min %ANGLE_MIN% ^
  --angle-max %ANGLE_MAX% ^
  --scale-min %SCALE_MIN% ^
  --scale-max %SCALE_MAX% ^
  --translate-min %TRANSLATE_MIN% ^
  --translate-max %TRANSLATE_MAX% ^
  --brightness-min %BRIGHTNESS_MIN% ^
  --brightness-max %BRIGHTNESS_MAX% ^
  --contrast-min %CONTRAST_MIN% ^
  --contrast-max %CONTRAST_MAX% ^
  --p-rotate %P_ROTATE% ^
  --p-flip-h %P_FLIP_H% ^
  --p-flip-v %P_FLIP_V% ^
  --p-affine %P_AFFINE% ^
  --p-crop %P_CROP% ^
  --p-brightness %P_BRIGHTNESS% ^
  --contour-simplify-epsilon %CONTOUR_SIMPLIFY_EPSILON% ^
  --min-component-area %MIN_COMPONENT_AREA% ^
  --min-mask-pixel-area %MIN_MASK_PIXEL_AREA% ^
  --random-aug-per-image %RANDOM_AUG_PER_IMAGE% ^
  %DEBUG_FLAG%

echo ========================================
echo Done.
echo ========================================

endlocal
