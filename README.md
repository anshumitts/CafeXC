# IMAGEXC
## [Will think of something]()
```bib
@InProceedings{Mittal21b,
    author       = "XC",
    title        = "VisualXC",
    booktitle    = "No Clue",
    month = "Sometime",
    year = "this year maybe",
    }
```

#### SETUP WORKSPACE
```bash
mkdir -p ${HOME}/scratch/XC/data 
mkdir -p ${HOME}/scratch/XC/programs
mkdir -p ${HOME}/scratch/XC/Corpus
mkdir -p ${HOME}/scratch/XC/RawData
```

#### SETUP IMAGEXC
```bash
cd ${HOME}/scratch/XC/programs
git clone https://github.com/anshumitts/ImageXC.git
conda create -f ImageXC/imagexc.yml
conda activate imagexc
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python setup.py install
cd ../ImageXC
```

#### DOWNLOAD DATASET
```bash
cd ${HOME}/scratch/XC
curl -J https://owncloud.iitd.ac.in/nextcloud/index.php/s/t4XYkStWt3Cg73S/download --output RawData.zip
unzip RawData.zip
```

#### SETUP DATASET
```bash
cd ${HOME}/scratch/XC/programs/ImageXC/dataset_scripts
chmod +x setup_dataset.sh
./setup_dataset.sh <ALL_GPU_IDS> <WORK_DIR> <CORPUS_NAME> <RAW_DATASET_NAME> <DATASET_NAME> <IMG_ENCODER> <FLAG1> <FLAG2> <FLAG3>
# FLAG1 : [0->disabled | 1->enabled]] Converts MSR Data format to XC format
# FLAG2 : [0->disabled | 1->enabled]] Setup images for documents (and labels [if exists])
# FLAG3 : [0->disabled | 1->enabled]] Build pre-trained features for document (and labels [if exists])
e.g.
./setup_dataset.sh 0,1 ${HOME}/scratch/XC SAMPLE-Amazon-1K SAMPLE-IMG-Amazon-1K SAMPLE-IMG-AmazonTitles-1K ViT 1 1 1
```

#### RUNNING IMAGEXC
```bash
cd ${HOME}/scratch/XC/programs/IMAGEXC/code_base
chmod +x run.sh
./run.sh <ALL_GPU_IDS> <TYPE> <DATASET> <FOLDER_NAME> <RANKER> <IMG_ENCODER> <TXT_ENCODER>
# TYPE          :	MultiModalSiameseXC SiameseTextXML
# DATASET       :	SAMPLE-IMG-AmazonTitles-1K (or your dataset name and configuration file in configs)
# FOLDER_NAME   :	USER's choice
# RANKER        :	NGAME XAttnRanker XAttnRanker-hybrid
# IMG_ENCODER   :	resnet18 vgg11 inception_v3 ViT resnet50FPN
# TXT_ENCODER   :	BoW Seq VisualBert sentencebert
e.g.
./run.sh 0,1 MultiModalSiameseXC SAMPLE-IMG-AmazonTitles-1K TEST XAttnRanker-hybrid ViT sentencebert

```
