# S1000-transformer-tagger
S1000 Transformer based NER tagger for literature

Code for paper:

## Environment setup:
This code is tested with Python 3.9 installed with conda and the packages from requirements.txt installed in that environment. Running setup.sh will download a NER model finetuned with S1000 dataset, example data and install the needed packages. You can substitute the NER model with a finetuned model trained with the accompanying repo meant for model finetunign https://github.com/jouniluoma/S1000-transformer-ner  

Quickstart
```
conda create -n s1000-env python=3.9
conda activate s1000-env
./setup.sh
./scripts/run-bio-tagger.sh
```
These create enviroment, installs required packages and runs tagging on example data
