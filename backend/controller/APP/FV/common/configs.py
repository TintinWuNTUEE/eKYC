def merge_configs(cfgs,new_cfgs):
    if hasattr(cfgs, 'img_IDcard_path'):
        new_cfgs['img']['img_IDcard_path']=cfgs.img_IDcard_path
    if hasattr(cfgs, 'img_Selfie_path'):
        new_cfgs['img']['img_Selfie_path']=cfgs.img_Selfie_path
    return new_cfgs
