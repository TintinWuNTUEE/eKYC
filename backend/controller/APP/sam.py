from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
class SAMSegmentor:
    def __init__(self,input_box,device,chkpt):
        self.sam_checkpoint = chkpt
        self.model_type ="vit_h"
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.device = device
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.input_box = input_box
    def __crop_image(self,img,seg_mask_np):
        contours, _ = cv2.findContours(np.uint8(seg_mask_np), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(largest_contour, epsilon=0.1, closed=True)
        x,y,w,h = cv2.boundingRect(polygon)
        object = img[y:y+h, x:x+w]
        rect =cv2.minAreaRect(polygon)
        angle = rect[-1]
        width, height = rect[1]
        if width>height:
            angle = angle
        else:
            angle = - (90-angle)
        center = (w//2, h//2)
        rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(object, rotation_mat, (w, h), flags=cv2.INTER_LANCZOS4)
        bbox = self.__seg2bbox(rotated>0)
        rotated = rotated[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        return rotated
    def __seg2bbox(self,seg_mask_np):
        rows = np.any(seg_mask_np, axis=1)
        cols = np.any(seg_mask_np, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bbox = np.array([xmin, ymin, xmax, ymax])
        return bbox
    def segment(self,image):
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
        point_coords=None,
        point_labels=None,
        box=self.input_box[None,:],
        multimask_output=False,
        )
        image = image * masks[0][..., np.newaxis]
        segmented_image = self.__crop_image(image,masks[0])
        return segmented_image
