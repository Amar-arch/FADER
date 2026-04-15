MODELTYPE=finer
EXPNAME=FINER
LR=0.0005
FIRSTOMEGA=1
HIDDENOMEGA=30
HIDDENLAYERS=3
HIDDENFEATURES=256
first_bias_scale=20





### train single image
CUDA_VISIBLE_DEVICES=1 python train_image_4.py \
    --model_type $MODELTYPE --exp_name $EXPNAME \
    --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
    --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES \
    --lr $LR \
    --img_id 14 \
    --first_bias_scale $first_bias_scale


### train multi image
# GPUID_TUPLE=(0 1 3 4)
# ID=0
# for IMGID in {0..15}
# do
#     GPUID=${GPUID_TUPLE[$ID]}
#     echo "GPUID:" $GPUID  "ImageId:" $IMGID
#     ((ID++))
#     ID=$((ID % 4))

#     ## train
#     CUDA_VISIBLE_DEVICES=$GPUID python train_image.py \
#         --model_type $MODELTYPE --exp_name $EXPNAME \
#         --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
#         --hidden_layers $HIDDENLAYERS --hidden_features $HIDDENFEATURES \
#         --lr $LR \
#         --img_id $IMGID &
# done
# wait