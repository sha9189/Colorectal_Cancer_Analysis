# Colorectal Adenocarcinoma Cancer Clinical and Genomic Data Analysis

In this project, we aimed to identify key features for predicting cancer recurrence in Colorectal Adenocarcinoma patients. Our approach involved data collection, cleaning, and feature extraction from both clinical and genomic datasets. Below is a detailed breakdown of our findings and methodologies.

## 1. Data Source Details:
**Name of Cancer Study**: Colorectal Adenocarcinoma (TCGA, PanCancer Atlas)    
**Data Source**: [cBioPortal](https://www.cbioportal.org/study/summary?id=coadread_tcga_pan_can_atlas_2018)   

### Probability Distribution of Cancer Recurrence Time 
<p align="center">
  <img src="images/recurrence%20time%20density%20plot.png" alt="Probability distribution of cancer recurrence time" width="400">
</p>

## 2. Clinical Data Analysis
We meticulously cleaned the clinical data by removing columns with low variability, irrelevant information (e.g., data collection techniques), or excessive missing values. We then utilized a Random Forest classifier to predict the cancer recurrence time range. The trained model provided the following feature importance plot:

<p align="center">
  <img src="images/clinical feature importance on two class.png" alt="Clinical Feature Importance" width="450">
</p>

## 3. Genomic Data Analysis
The genomic dataset included every gene mutation observed in each patient, with the number of mutations per patient ranging from over 17,000 to just 13. We narrowed our focus to 13 gene signatures as guided by Chapter 25 of Reference [2]. The distribution of these gene signatures among the patients is depicted below:

<p align="center">
  <img src="images/genomic data distribution.png" alt="Genomic Data Distribution" width="450">
</p>

Using the Random Forest Classifier, we obtained the following feature importance plot for the genomic data:

<p align="center">
  <img src="images/genomic feature importance on three class.png" alt="Text" width="450">
</p>



## References: 

[1] [Colorectal Adenocarcinoma (TCGA, PanCancer Atlas)](https://www.cbioportal.org/study/summary?id=coadread_tcga_pan_can_atlas_2018)

[2] Diagnostic Molecular Pathology : A Guide to Applied Molecular Testing, edited by William B. Coleman, and Gregory J. Tsongalis, Elsevier Science & Technology, 2023. ProQuest Ebook Central, https://ebookcentral-proquest-com.gate.lib.buffalo.edu/lib/buffalo/detail.action?docID=30795817.