# CODE FROM : https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9
mkdir coco
cd coco
mkdir images
cd images

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip train2017.zip
unzip val2017.zip

rm train2017.zip
rm val2017.zip

cd ../
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip

rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip