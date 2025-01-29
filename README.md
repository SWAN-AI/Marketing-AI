# Marketing-AI  
## Multimodal Marketing Success Prediction  

This project aims to predict the success of marketing campaigns using **multimodal data**. By combining various types of data, such as **text, images, and numerical features**, we build a predictive model that helps identify the most effective marketing strategies.  

📄 **Related Paper:**  
Our research on this topic has been published on **arXiv**. You can read the full paper here:  
[🔗 Enhancing Cross-Modal Contextual Congruence for Crowdfunding Success using Knowledge-infused Learning](https://arxiv.org/abs/2402.03607)

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In this project, we leverage multimodal data to predict the success of marketing campaigns. By analyzing textual content, images, and other relevant features, we aim to build a model that can accurately predict the performance of different marketing strategies.

## Data
The dataset used in this project consists of a combination of textual data, image data, and numerical features. The dataset is collected from various marketing campaigns and contains information such as campaign text, campaign images, target demographics, and campaign success metrics.

## Project Structure
The project has the following structure:


- `data/`: Directory containing the sample dataset files and images.
- `src/`: Directory containing the different models used in the project, such as text models, image models, and the multimodal model.
- `utils/`: Directory containing utility functions for data preprocessing and evaluation.
- `notebooks/`: Directory containing Jupyter notebooks for data analysis, text modeling, image processing, and multimodal modeling.
- `README.md`: This file you're currently reading.
- `requirements.txt`: File specifying the project dependencies.

## Setup
To set up the project, follow these steps:

1. Clone the repository:

2. Install the required dependencies:

3. Download the necessary dataset files and place them in the `data/` directory.

## Usage
To run the project, you can use the provided Jupyter notebooks in the `notebooks/` directory. Each notebook focuses on a specific aspect of the project, such as data analysis, text modeling, image processing, and multimodal modeling. Follow the instructions in the notebooks to execute the code and reproduce the results.

To run a file please use this :
`python mmbt/train.py --batch_sz 4 --gradient_accumulation_steps 40 --savedir/result /path/to/savedir/ --name mmbt_model_run --data_path /path/to/datasets --task food101 --task_type classification --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1`

<!-- """python train_MMBT_ConceptNet_cuda.py --batch_sz 4 --gradient_accumulation_steps 40 --savedir results_9_6/ --name mmbt_model_run 
--data_path kickstarter_data --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --max_epochs 5 """

# for windows machine gpu 6 - I4I
# python train_MMBT_cuda.py --batch_sz 64 --gradient_accumulation_steps 40 --savedir results_mmbt_12_9/ --name mmbt_model_run --data_path C:\Users\tpadhi1\Desktop\Adobe\mmbt-kg\kickstarter_dataset_processed --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --max_epochs 50

# for vision transformer
# python train_MMBT_ViT_Bert.py --batch_sz 32 --img_hidden_sz 768 --gradient_acuumulation_steps 40 --gradient_accumulation_steps 40 --savedir test --name mmbt_model_run --data_path C:\Users\tpadhi1\Desktop\Adobe\mmbt-kg\data_prep_codes\Experiments\Transe --model mmbt --num_image_embeds 197 --freeze_txt 5 --freeze_img 3 --max_epochs 50
# python train_BLIP.py --batch_sz 16  --gradient_accumulation_steps 40 --savedir test --name mmbt_model_run --data_path C:\Users\tpadhi1\Desktop\Adobe\mmbt-kg\data_prep_codes\Experiments\Transe --model mmbt --max_epochs 50 -->

## Results
The project aims to achieve accurate predictions of marketing campaign success using multimodal data. The final model's performance is evaluated using appropriate metrics, and the results are presented in the notebooks or in a separate evaluation report.

## Contributing
Contributions to this project are welcome! If you have any ideas, suggestions, or improvements, please create an issue or submit a pull request.

## Citation 📖  

If you find our work useful, please consider citing our paper:  

```bibtex
@inproceedings{padhi2024enhancing,
  author    = {Trilok Padhi and others},
  title     = {Enhancing Cross-Modal Contextual Congruence for Crowdfunding Success using Knowledge-infused Learning},
  booktitle = {2024 IEEE International Conference on Big Data (BigData)},
  publisher = {IEEE},
  year      = {2024}
}
```


## License
This project is licensed under the [MIT License](LICENSE).
