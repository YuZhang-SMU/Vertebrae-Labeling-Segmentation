# Semantics and Instance Interactive Learning for Labeling and Segmentation of Vertebrae in CT Images [MIA'25] 

[**Link to the Paper**](https://www.sciencedirect.com/science/article/abs/pii/S1361841524003050)  

---

## Framework  

![Framework](framework.png)

---

## Contributions  

The main contributions of this paper are as follows:  

1. **New Learning Paradigm**  
   We introduce a new paradigm based on semantics and instance in-teractive learning, namely SIIL, for synchronous labeling and seg-mentation of vertebrae in CT images. Feature interaction embod-ies to help learn position and contour information, and improvethe separability of vertebral instances.  

2. **MILL Module**  
   We propose an MILL module to facilitate the interaction betweensemantic and instance features, which introduces an instancelocalization matrix to filter out absent vertebrae and alleviatetheir interference.

3. **OCPL Module**  
   We design an OCPL module to mitigate the high similarity of adja-cent vertebrae by modeling the intrinsic sequential relationship ofinstance features via Bi-GRU, and enhance inter-class separabilityand intra-class consistency via cross-image contrastive learning. 

4. **Extensive Experiments**  
   We conduct extensive experiments to demonstrate the effective-ness of the proposed method, and it achieves optimal performancein three public datasets.

---

## Data Organization  

Below is the recommended structure for organizing your data:  

```plaintext
Project/  
│  
├── data/  
│   ├── img/  
│   │   ├── 1_3_6_1_4_1_9328_50_4_0001.nii.gz
│   │   └── ...
│   ├── msk/  
│   │   ├── 1_3_6_1_4_1_9328_50_4_0001.nii.gz 
│   │   └── ...
│   ├── pap/  
│   │   ├── 1_3_6_1_4_1_9328_50_4_0001.nii.gz 
│   │   └── ...
│   ├── training.txt
│   └── testing.txt
└── SIIL/...
```

---

### File Descriptions  

- **`data/img`**: Directory containing processed image data in `.nii.gz` format.  
- **`data/msk`**: Directory containing processed mask data in `.nii.gz` format.
- **`data/pap`**: Directory containing processed probability map in `.nii.gz` format. (only used for training)  
- **`training.txt`**: A text file containing patient names used for the training data.  
- **`testing.txt`**: A text file containing patient names used for the testing data.

---

## Code Execution Steps

Preprocess the data and organize files according to the **Data Organization** section.

Install the dependencies:  
   ```bash  
   conda install -c conda-forge --file requirements.txt
   ```
Train the model:
   ```bash  
   python SIIL.trainer.py  
   ```
Test the model:
   ```bash  
   python SIIL.tester.py  
   ```
To modify parameters, edit the file:  **`SIIL.get_config.py`**
