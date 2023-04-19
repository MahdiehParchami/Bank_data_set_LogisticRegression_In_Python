# Bank_data_set_LogisticRegression_In_Python
Implementing logistic Regression technique on Bank Dataset

In this study I want to perform the logistic Regression technique in a bank data set to predict the Loan Approval of new loan requests.

In this study, I cleaned the data and performed descriptive analysis to get a better understanding of the data, and then I examined the Logistic Regression to predict the Loan Approval 
of new loan requests. At first, I performed Feature Scaling to have a standard scale then performed the logistic Regression using all variables. The accuracy of the model was 96.6% with a Precision 
rate of 85%. Also, I discovered that Income, Education, CD account, Family, and CCAvg were the variables most significantly impacted with P-values of less than 0.05, while Age, Mortgage and Zip 
Code did not show a significant impact with a P-value greater than 0.05. Moreover, The CCavg, CD account, and Income coefficients are positive, which means a one-unit increase in any of these 
variables will increase a person's chances of borrowing a loan. On the other hand, a person's chances of borrowing will decrease if they increase their credit card, online, or securities accounts. 
In order to optimize our model, I dropped variables that had no significant impact on predicting loan approval, and I used only the most significant variables in our model. Results 
showed that precision increased from 85% to 87%, but accuracy did not change significantly.
