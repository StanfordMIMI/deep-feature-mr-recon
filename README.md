# Using Deep Feature Distances for Evaluating MR Image Reconstruction Quality

- **Title:** [Using Deep Feature Distances for Evaluating MR Image Reconstruction Quality](https://openreview.net/forum?id=AUiZyqYiGb)
- **Authors:** [Philip M. Adamson](https://www.linkedin.com/in/philipadamson/), Arjun D Desai, Jeffrey Dominic, Christian Bluethgen, Jeff P. Wood, Ali B Syed, Robert D. Boutin, Kathryn J. Stevens, Shreyas Vasanawala, John M. Pauly, Akshay S Chaudhari, Beliz Gunel
- **Project Website:** https://stanfordmimi.github.io/deep-feature-mr-recon/
- **Contact:** {padamson} [at] stanford [dot] edu

![Deep Feature Distances Methods](data/DFD_Methods.png)

## Datasets
All reader study MR reconstructions and radiologist reader study scores can be downloaded [here](https://drive.google.com/drive/folders/1REr4R_geovFPpz1aYYX-P2GDBNTxosgc?usp=share_link).

## Code

### Set-up
Create a conda environment for this project:

```bash
conda create -n dfd_env python=3.9
conda activate dfd_env
```

Install torch
```bash
conda install pytorch=2.1.1 torchvision=0.16.1 cudatoolkit=10.1 -c pytorch
```

Finally install dependencies from the requirements.txt file

```bash
pip install -r requirements.txt
```

### Basic Usage
To compute metrics on the MR reconstruction reader study dataset, run the following command:

```bash
cd deep-feature-mr-recon # Navigate to your cloned repo
python deep-feature-mr-recon/reader_study_metrics.py --img_dir [path_to_image_folder] --results_dir [path_to_save_results]

```

The Jupyter Notebook ReaderStudy_vs_Metrics.ipynb can then be used to analyze correlations between the computed metrics and radiologist reader study scores.

### Advanced Usage
The deep-feature-mr-recon project is built on top of [meddlr](https://github.com/ad12/meddlr), a config-driven an ML framework built to simplify medical image reconstruction and analysis problems.
Deep Feature Metrics such as LPIPS and SSFD have been incorporated into meddlr to use as both an evaluation and optimization metric for any MR reconstruction task. 
Refer to the meddlr documentation for more details.

## Citation

```
@inproceedings{adamson2023using,
  title={Using Deep Feature Distances for Evaluating MR Image Reconstruction Quality},
  author={Adamson, Philip M and others},
  booktitle={NeurIPS 2023 Workshop on Deep Learning and Inverse Problems},
  year={2023}
}
```

