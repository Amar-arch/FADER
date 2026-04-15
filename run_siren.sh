MODELTYPE=siren
EXPNAME=SIREN
LR=0.0005
FIRSTOMEGA=20
HIDDENOMEGA=30

GPUID=0
IMGID=14

## train_siren_gradiet_loss

#  CUDA_VISIBLE_DEVICES=$GPUID python train_image.py \
#     --model_type $MODELTYPE --exp_name $EXPNAME \
#     --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
#     --lr $LR \
#     --img_id $IMGID 


## train_siren_PE

# CUDA_VISIBLE_DEVICES=$GPUID python train_image_1.py \
#     --model_type $MODELTYPE --exp_name $EXPNAME \
#     --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
#     --lr $LR \
#     --img_id $IMGID 


## train_gelu_PE

#  CUDA_VISIBLE_DEVICES=0 python train_image_2.py \
#     --model_type $MODELTYPE --exp_name $EXPNAME \
#     --lr 0.003 \
#     --img_id $IMGID 


## train_siren

CUDA_VISIBLE_DEVICES=0 python train_image_4.py \
    --model_type $MODELTYPE --exp_name $EXPNAME \
    --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
    --lr $LR \
    --img_id $IMGID 





