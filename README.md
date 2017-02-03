### Predicting churn from user attributes

Data: Telecom customer data. Each row contains customer attributes such as call minutes during different times of the day, charges incurred for services, duration of account, and whether or not the customer left or not.  

I use the customer features to train a random forest classifier and calculate probability of churn and expected loss. 


### Documents  
churn_helper.py contains functions to calculate churn probability, expected loss, 
calibration, and discrimination.  
model_helper.py contains functions for model and cross validation.  
Churn.ipynb goes through the workflow.  

---

To see a full description, go to the blog post [here](https://joomik.github.io/churn/).