# Artificial Intelligence Project

This repository contains the project for artificial intelligence course taken at Unibo a.y. 2021/2022. 
The work aims to predict if a student enrolled at university will give up the studies. Being able to predict such an event, one could prevent the problem by assisting the students and also try to understand the reasons behind the dropout. 

## Model and Data

The problem has been approached by exploiting articial intelligence techniques, in particular K-Nearest Neighbours, Decision Trees and Random Forests. Those algorithms are applied to data coming from University of Bologna, concerning the students enrolled during the years 2016-2018. The dataset is not included in the repository as it contains private, sensible information.
A first attempt to understand which features combination makes the predictions more effective can be observed in 'src/feature_analysis.py'. Then the final tests are in 'src/problems.py'.

### Conclusions

The results are satisfying as an accuracy of the 80% is reached in almost all the cases, the same is for specificity. While with regard to sensitivity, mainly for bachelor’s and unique cycle degrees, the results show a percentage of more than the 90%; that is the most interesting result as this evaluation metric deals with the prediction of just the dropouts 
