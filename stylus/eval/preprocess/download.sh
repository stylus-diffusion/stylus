mkdir -p ~/coco
cd ~/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip annotations_trainval2014.zip
unzip val2014.zip
rm annotations_trainval2014.zip
rm val2014.zip
