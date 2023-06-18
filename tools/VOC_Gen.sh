# VOC_Gen.sh
# You can use this script to convert the noise-injected PASCAL VOC dataset, which was created using 'noise_inj.sh', into a convenient format.

mkdir ../../pas_noise_vocfmt

noise_ratio="5 10 15 20 25 30 35 40"
for var in ${noise_ratio}
do
    # create directory
    mkdir ../../pas_noise_vocfmt/pas_noise_vocfmt${var}
    mkdir ../../pas_noise_vocfmt/pas_noise_vocfmt${var}/VOC2007
    mkdir ../../pas_noise_vocfmt/pas_noise_vocfmt${var}/VOC2007/JPEGImages
    mkdir ../../pas_noise_vocfmt/pas_noise_vocfmt${var}
    mkdir ../../pas_noise_vocfmt/pas_noise_vocfmt${var}/VOC2012
    mkdir ../../pas_noise_vocfmt/pas_noise_vocfmt${var}/VOC2012/JPEGImages

    # coco fmt noise 2 voc fmt
    python coco2voc.py --ann_file ../../pas_cocofmt_noise/newmixnoisy${var}voc_trainval07.json --output_dir ../../pas_noise_vocfmt/pas_noise_vocfmt${var}
    python coco2voc.py --ann_file ../../pas_cocofmt_noise/newmixnoisy${var}voc_trainval12.json --output_dir ../../pas_noise_vocfmt/pas_noise_vocfmt${var}

    # copy devkit
    cp -r ../../data/VOCdevkit ../../data/VOCdevkit${var}

    # file overwrite
    cp -r ../../pas_noise_vocfmt/pas_noise_vocfmt${var}/VOC2007/JPEGImages/* ../../data/VOCdevkit${var}/VOC2007/Annotations
    cp -r ../../pas_noise_vocfmt/pas_noise_vocfmt${var}/VOC2012/JPEGImages/* ../../data/VOCdevkit${var}/VOC2012/Annotations
    
    echo noise ratio ${var} done
done
