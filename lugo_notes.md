barplot:


fixed_acidity vs. quality: all the bars look very similar except for 9 

volatile_acidity vs. quality: all the bars look to have the same height, around 3.0, except for quality 4 and 5, which are slightly above .35

* citric_acid vs. quality: the bars have a trend from low to higher, and Quality 3 is the odd one here as it's almost the same height as the highest quality of 9. 

* residual_sugar vs. quality: no real trend here as 5 and 6 have a higher residual sugar compared to 7 and 8

* chlorides vs. quality: it has a better trend as the higher the chlorides, the lower the quality.

** free_sulfur_dioxide vs. quality: has a trend that the lower the free sulfur dioxide, the lower the quality. 

total_sulfur_dioxide vs. quality: not much of a trend here; 3 and 4 have lower total sulfur dioxide than the others, but the rest have the same bar level.

density vs. quality: No trend; all have the same density level

ph vs. quality: No trend; all have the same ph level

sulphates vs. quality: No trend; the lowest and highest quality are the same

alcohol vs. quality: No trend; the alcohol levels are almost the same






lmplot Notes:

fixed_acidity vs. quality: the regression line appears to decline instead of going upwards. 

volatile_acidity vs. quality: the regression line looks worse than the one above, and the line looks in decline. 

citric_acid vs. quality: the LoR is going in the upper direction

residual_sugar vs. quality: the LoR is in decline

chlorides vs. quality: the LoR is in decline

free_sulfur_dioxide vs. quality: the LoR is heading upwards

total_sulfur_dioxide vs. quality: the LoR is in decline, almost flat

density vs. quality: the LoR is in significant decline

ph vs. quality: the LoR is slightly heading upwards

sulphates vs. quality: the LoR is in heading upwards

alcohol vs. quality: the LoR has a solid upper line



Modeling:

ran the OLS+RFE, OLS, LARS, Polynomial and GLM models with all 12 features without the target variable "quality" and the best model was the polynomial with 2 degrees. The poly w/2D had the lowest RMSE in train and validate and the highest R2.


Test the model without 'density' and 'residual_sugar' and poly_2D was still the best model

Test the model with only the three Kbest features ('volatile_acidity', 'chlorides', and 'alcohol') and poly_2D was still the best model

![image.png](attachment:0c335567-e306-47c3-bec9-344e3cdab21d.png)