from yacs.config import CfgNode as CN

__C = CN()
__C.device = "cuda"
# Threshold for each score
__C.THRES = CN()
__C.THRES.PAD = 0.5
__C.THRES.FV = 0.64
__C.THRES.IDFD = 0
__C.THRES.OCR = 0
__C.THRES.FIDD = 200

# Weights for each score
__C.WEIGHTS = CN()
__C.WEIGHTS.PAD = 0.3
__C.WEIGHTS.FV = 0.3
__C.WEIGHTS.IDFD = 0
__C.WEIGHTS.OCR = 0.2
__C.WEIGHTS.FIDD = 0.2

__C.SEGMENT = CN()
__C.SEGMENT.INPUTBOX = [88,250,950,800]
__C.SEGMENT.CHKPT = '/home/tintinw/eKYC/backend/controller/APP/sam_vit_h_4b8939.pth'
def get_cfg():
    '''
    Return a clone of the config 
    Call by value (not calling by reference)
    '''
    return __C.clone()

if __name__ == "__main__":
    print(__C)