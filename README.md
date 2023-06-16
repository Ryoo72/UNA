<p align="center">
 <img width="80%" src="./figures/UNA_logo.png"/>
</p>


# UNA
This repository contains the code for generating **U**niversal-**N**oise **A**nnotation (UNA), which is a more practical setting that encompasses all types of noise that can occur in object detection. Additionally, experiment configuration files, log files, and links to download pre-trained weights are included.

You can use this code to simulate various types of noise and evaluate the performance of object detection models under different noise settings. It provides a comprehensive framework for studying the impact of noise on object detection algorithms.

# ⚡️ Quick Start

UNA dataset can be generated by una_inj.py.

~~~bash
python una_inj.py --ratio 0.1
~~~

una_inj.py takes following arguments

- `--path` : The file path to the COCO annotation JSON file (e.g., ./instances_train2017.json).
- `--target` : The file path where the UNA dataset will be stored.
- `--output` : Prefix for the output file.
- `--ratio` :  Intensity of the synthesized noise.
- `--class_type` : Select either 'coco' or 'voc'.

> **NOTE** : Currently, una_inj.py only supports the COCO format.


### COCO Experiments

1. download cocodata
우선 coco dataset 이나 pascal voc를 다운 받아라. 여기서 받으면 편하다.
https://cocodataset.org/#home


2. 아래의 스크립트로 UNA dataset을 생성해라.

~~~bash
git clone https://github.com/Ryoo72/UNA.git
cd UNA
bash una_inj.sh
~~~


3. mmdetection이나 detectron을 이용해서 당신의 실험을 enjoy해라.

## PASCAL VOC로 실험하기

1. download dataset
여기서 받아라.

2. COCO 형식으로 변환해라.
변환 툴은 mmdetection에 있다.

3. UNA 생성
~~~bash
git clone https://github.com/Ryoo72/UNA.git
cd UNA
python una_inj.py --ratio 0.1 --class_type voc
python una_inj.py --ratio 0.2 --class_type voc
python una_inj.py --ratio 0.3 --class_type voc
python una_inj.py --ratio 0.4 --class_type voc
~~~

4. mmdetection이나 detectron을 이용해서 당신의 실험을 enjoy해라. VOC포맷으로 다시 변경하고 싶다면, 이 레포를 참고해라.

# 📄 License

Distributed under the MIT License. LICENSE contains more information.

# ✉️ Contact

If you have any questions or inquiries regarding the usage of this repository, please feel free to reach out to us at kwangrok.ryoo@lgresearch.ai. Your feedback and engagement are highly valued, and we look forward to hearing from you.