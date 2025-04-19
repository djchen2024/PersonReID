# CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_tiny.yml TEST.WEIGHT './swin_tiny_market.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
# CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_small.yml TEST.WEIGHT './swin_small_market.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
# CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_base.yml TEST.WEIGHT './swin_base_market.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2



# CUDA_VISIBLE_DEVICES=6 python test.py --config_file configs/msmt17/swin_tiny.yml TEST.WEIGHT './swin_tiny_msmt17.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
# CUDA_VISIBLE_DEVICES=6 python test.py --config_file configs/msmt17/swin_samll.yml TEST.WEIGHT './swin_small_msmt17.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
# CUDA_VISIBLE_DEVICES=6 python test.py --config_file configs/msmt17/swin_base.yml TEST.WEIGHT './swin_base_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2



# CUDA_VISIBLE_DEVICES=6 python test.py --config_file configs/reid_CUSTOM_v1/swin_base.yml TEST.WEIGHT './swin_base_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2
# CUDA_VISIBLE_DEVICES=6 python test.py --config_file configs/reid_CUSTOM_v2/swin_base.yml TEST.WEIGHT './swin_base_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2





# Dynamic Gallery ---------------------
# CUDA_VISIBLE_DEVICES=6 python streamed_test.py --config_file configs/reid_CUSTOM_v1/swin_base.yml TEST.WEIGHT './swin_base_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2
# CUDA_VISIBLE_DEVICES=6 python streamed_test.py --config_file configs/reid_CUSTOM_v2/swin_base.yml TEST.WEIGHT './swin_base_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2




# Dynamic Gallery on Custom videos ---------------------
CUDA_VISIBLE_DEVICES=6 python streamed_test.py --config_file configs/reid_CUSTOM_VIDEO/swin_base.yml TEST.WEIGHT './swin_base_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2
