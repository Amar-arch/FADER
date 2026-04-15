MODELTYPE=siren_1
EXPNAME=SIREN
LR=0.0005
FIRSTOMEGA=1
HIDDENOMEGA=30

GPUID=0
IMGID=14
first_bias_scale=20


## train





CUDA_VISIBLE_DEVICES=$GPUID python train_image_4.py \
     --model_type $MODELTYPE --exp_name $EXPNAME \
     --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
     --lr $LR \
     --img_id $IMGID \
     --first_bias_scale $first_bias_scale