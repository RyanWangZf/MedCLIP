# ReportGenerationMeetsGraph
Code base for our AAAI 2020 paper "When Report Generation Meets Knowledge Graph". Please cite our paper if you find the code useful for your research.

The proposed model is defined in 'sentgcn.py'. Note that a pre-defined disease observation graph is hard-coded in 'train_gcnclassifier.py' and 'train_sentgcn.py'. Baseline models are defined in their specific model files. To train a model, just run the corresponding 'run_*.sh' file.

The dataset splits, class keywords and vocabs can be found in the data folder.

In our setting, we first train the classifier, then train the report generation decoder.
