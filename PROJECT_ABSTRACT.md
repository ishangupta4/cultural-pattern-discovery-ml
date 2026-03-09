# Project Abstract

## Discovering Cultural Periods from Museum Metadata with Classical Machine Learning

**Course:** CS 6140 — Machine Learning
**Instructor:** Prof. Smruthi Mukund
Summary 

The Metropolitan Museum of Art in New York has one of the largest and most varied collections of art in the world, covering more than 5000 years of the creative history of the human race. In 2017, the museum made metadata of all its collection available as a free to use csv file under creative commons zero license. This dataset is composed of over 470,000 records of artworks which are described by 43 columns, including such characteristics as the medium used (e.g. "Oil on canvas" or "Bronze"), culture of origin, artist name, date of creation, department classification and descriptive tags.

Almost all of the machine learning work on museum collections has been concerned with analyzing images of the artworks, and the tabular metadata has almost completely unexplored. The aim of this project is to address this gap by building supervised and unsupervised models which work purely from the columns of the Met's metadata and without the use of any images. Specifically, we aim to (a) identify the curatorial department that an artwork belongs to using the metadata features, and (b) use clustering algorithms to identify groupings in the collection and compare the outcomes of the machine discovered groupings with art historical period labels established.

Data Challenges

What makes this project particularly good to demonstrate some of the fundamental concepts of machine learning is the nature of the data itself. Eighteen of the 43 columns have missing in more than 75 percent of records, so missing data handling is a major part of the work. The target variable for classification (Department) is highly imbalanced where the number of records in the Drawings and Prints department is higher than 154,000 and the number of records in some department is in the hundreds range, hence the choice of metrics and resampling techniques are important. Several columns contain a free text (Medium, Tags, Classification) that needs to be transformed into numbers using, for example, TF IDF or Count Vectorization. Others are categorical variables for which there are many unique values (Culture has over 2,000) and creative encoding strategies are required.

Approach

We plan to organize our investigation in the form of a comparison between families of models: baseline (logistic regression), then ensemble tree methods (Random Forest, XGBoost), then SHAP based feature importance analysis to explain which metadata attributes are most predictive. On the unsupervised side, we will apply K Means and hierarchical clustering on engineered representation of features, visualize the findings using (UMAP), and determine if the clusters represent known art historical periods. Our key idea is that the combination of text features, engineered numerical attributes (e.g. age and complexity of the medium of artworks), and careful treatment of missing data, will substantially beat naive baselines and find interpretable patterns in how the Met organizes its collection.

Motivation

Our research material contributes to an emerging, small body of research on the intersection of machine learning and cultural heritage data. Prior surveys have found that adoption of ML in this area is still limited with most work being focused on image classification and not structured metadata analysis. Recent studies have shown that it is possible to effectively curate art exhibitions using simple text features extracted from metadata columns, and that tabular metadata can be in the same ballpark as approaches that also use images and text embeddings. These works suggest that structured museum data can be used for classical ML and is understudied, and there is much value to extract from metadata features using careful feature engineering.
