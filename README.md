# cxr-gen-report-CLIP

You can see our report for details: [here](./final_report.pdf)

* Dataset: [IU X-ray](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university), [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) (Actually, we used [pretrained CLIP model](https://stanfordmedicine.app.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml) with the dataset)
* Model architecture for training: ResNet18, EfficientNet-b4
* Loss functions for training: Cross Entropy Loss, CLIP directional Loss
* AdamW optimizer and 2e-3 learning rate
* Result:
    * Train Loss Plots(CE, CLIP): ![ResNet](./plot/ResNet%20%2B%20CLIP%20Loss.png), ![efficientNet](./plot/EfficientNet%20%2B%20CLIP%20Loss.png)

* Reference:

    https://github.com/atimashov/cxr-report-generation (our baseline),

    https://github.com/rajpurkarlab/CXR-RePaiR (CLIP model),

    https://github.com/rinongal/StyleGAN-nada (CLIP directional Loss)
