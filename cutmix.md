# Generate annotations from existing dataset

0. find the largest image dims in dataset (Hmax, Wmax)
1. for each image in dataset:
    - set STEP = 10 (pixels)
    1. if image.w < Wmax/2 and image.h >= Hmax/2:
        1. create a window of size (Hmax/2, image.w)
        2. x1 = 0
        3. x2 = image.h
        4. y1 = 0
        5. y2 = Hmax/2
        6. while y2 < image.h:
            - window <-- x1, y1, x2, y2
            - filter out boxes from the original image that have zero overlap with the window. Let's call them `filtered_level_1_bboxes`
            1. use one_image_subarray.crop_boxes with a high threshold (like 0.9) to filter_and_crop the `filtered_level_1_bboxes`. Let's call them `filtered_level_2_bboxes`.
            2. if `filtered_level_1_bboxes.count == filtered_level_2_bboxes.count`:
                1. create annotation for that image and store in `cropped_vertically`.
                2. store image for that annotation in `cropped_vertically`
            3. y1 = y1 + STEP
            4. y2 = y2 + STEP
    2. else if image.w >= Wmax/2 and image.h < Hmax/2:
        1. create a window of size (image.h, Wmax/2)
        2. x1 = 0
        3. x2 = Wmax/2
        4. y1 = 0
        5. y2 = image.h
        6. while x2 < image.w:
            - window <-- x1, y1, x2, y2
            - filter out boxes from the original image that have zero overlap with the window. Let's call them `filtered_level_1_bboxes`
            1. use one_image_subarray.crop_boxes with a high threshold (like 0.9) to filter_and_crop the `filtered_level_1_bboxes`. Let's call them `filtered_level_2_bboxes`.
            2. if `filtered_level_1_bboxes.count == filtered_level_2_bboxes.count`:
                1. create annotation for that image and store in `cropped_horizontally`.
                2. store image for that annotation in `cropped_horizontally`
            3. x1 = x1 + STEP
            4. x2 = x2 + STEP

    3. else if image.w >= Wmax/2 and image.h >= Hmax/2:
        1. y1 = 0
        2. y2 = Hmax/2
        3. while y2 < image.h:
            - x1 = 0
            1. x2 = Wmax/2
            2. while x2 < image.w:
                - window <-- x1, y1, x2, y2
                - filter out boxes from the original image that have zero overlap with the window. Let's call them `filtered_level_1_bboxes`
                1. use one_image_subarray.crop_boxes with a high threshold (like 0.9) to filter_and_crop the `filtered_level_1_bboxes`. Let's call them `filtered_level_2_bboxes`.
                2. if `filtered_level_1_bboxes.count == filtered_level_2_bboxes.count`:
                    1. create annotation for that image and store in `cropped_square`.
                    2. store image for that annotation in `cropped_square`
                3. x1 = x1 + STEP
                4. x2 = x2 + STEP
            2. y1 = y1 + STEP
            3. y2 = y2 + STEP
    4. else:
        1. do nothing (too small image)

## Notes:
- do not use arbitrary windows like (h, w) where h > Hmax/2 and w > Wmax/2, because having fixed Hmax/2 or Wmax/2 shapes (or both) will help with stitching images together
- do not consider Hmax, Wmax to match the hyperparameter input_size. You will create a dependency on the experiment. The new generated dataset should be independent on the experiment, but dependent on the original dataset.