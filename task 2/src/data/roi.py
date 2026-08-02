import cv2
import numpy as np
import random

def get_smart_roi_crops(img_before: np.ndarray, img_after: np.ndarray, tile_size: int = 512, num_crops: int = 20):
    """
    Finds the common area without black borders (NoData) on both images
    and returns a list of safe coordinates (x, y) for cropping tiles.
    """
    h, w = img_before.shape
    
    # 1. Create binary masks: where any data exists (brightness > 0)
    # Sentinel-2 often fills NoData with absolute zeros.
    mask_before = (img_before > 0).astype(np.uint8)
    mask_after = (img_after > 0).astype(np.uint8)
    
    # 2. Find the intersection (logical AND). We need an area that exists in BOTH images.
    joint_mask = cv2.bitwise_and(mask_before, mask_after)
    
    # 3. Find contours (the same curved boundaries of the satellite image)
    contours, _ = cv2.findContours(joint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No shared valid data region exists between the images.")
        return []
        
    # Take the largest contour (main land mass)
    main_contour = max(contours, key=cv2.contourArea)
    
    # 4. Describe a bounding rectangle around the complex curve (Bounding Box)
    # to narrow the search area and avoid generating points across the whole canvas.
    x_min, y_min, box_w, box_h = cv2.boundingRect(main_contour)
    
    # Check: does at least one tile even fit in this land patch?
    if box_w < tile_size or box_h < tile_size:
        print("Error: Shared region is too small for tile size", tile_size)
        return []

    valid_crops = []
    max_attempts = 1000 # Safety guard against an infinite loop
    
    # 5. Generate safe coordinates
    for _ in range(max_attempts):
        if len(valid_crops) >= num_crops:
            break
            
        # Sample a point only within the found bounding box
        rand_x = random.randint(x_min, x_min + box_w - tile_size)
        rand_y = random.randint(y_min, y_min + box_h - tile_size)
        
        # CRITICAL CHECK: did our 512x512 square touch the black border of the contour?
        # Crop a patch from the shared mask
        crop_mask = joint_mask[rand_y : rand_y + tile_size, rand_x : rand_x + tile_size]
        
        # If ALL pixels in the square are 1 (i.e., they lie inside the contour)
        if np.all(crop_mask == 1):
            valid_crops.append((rand_x, rand_y))
            
    return valid_crops