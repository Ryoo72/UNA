# voc_down.sh
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget http://pjreddie.com/media/files/VOC2012test.tar
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOC2012test.tar

wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar

cp -r VOCdevkit data/

python tools/dataset_converters/pascal_voc.py ./data/VOCdevkit -o ./pas_cocofmt --out-format coco
mkdir pas_cocofmt_noise