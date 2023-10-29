# Dataset: [Credit-Risk-Analysis](https://colab.research.google.com/drive/1dgcLjI79c4C0rAKCQckL9YnTcOxEaPVi#scrollTo=1O_QUKxkTFq3)

## About Dataset
The dataset contains complete loan data for all loans issued through the year 2007 - 2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information.

This project consists of 6 parts:
1. Introduction
2. Data Cleaning
3. Data visualization
4. Preprocessing
5. Modeling
6. Inference
7. Conclusion

# 1. Introduction
This dataset contains total of **8,55,969 records** with **73 features** including target variable.

## Attribute description:
1. **id:** A unique identifier or reference number for each entry in the dataset.
2. **member_id:** A unique identifier or reference number for each member or individual associated with the loan.
3. **loan_amnt:** The total amount of the loan applied for or approved.
4. **funded_amnt:** The actual amount of the loan that has been funded or disbursed to the borrower.
5. **funded_amnt_inv:** The amount of the loan that was funded by investors. In some cases, loans can be funded by multiple investors.
6. **term:** The term or duration of the loan, typically in months (e.g., 36 months or 60 months).
7. **int_rate:** The annual interest rate (in percentage) charged on the loan.
8. **installment:** The monthly payment amount that the borrower needs to make to repay the loan.
9. **grade:** A rating or grade assigned to the loan, often reflecting the borrower's creditworthiness.
10. **sub_grade:** A more specific rating or grade within the grade category, providing additional information about the loan's risk.
11. **emp_title:** The job title or occupation of the borrower.
12. **emp_length:** The length of time the borrower has been employed.
13. **home_ownership:** The type of home ownership (e.g., own, rent, mortgage) of the borrower.
14. **annual_inc:** The annual income of the borrower.
15. **verification_status:** The status of income verification for the borrower (e.g., verified, not verified).
16. **issue_d:** The date when the loan was issued or originated.
17. **pymnt_plan:** An indicator of whether the loan has a payment plan.
18. **desc:** A description or additional information related to the loan.
19. **purpose:** The purpose for which the loan is being used (e.g., debt consolidation, home improvement).
20. **title:** The title or description of the loan.
21. **zip_code:** The borrower's ZIP code.
22. **addr_state:** The state in which the borrower resides.
23. **dti:** Debt-to-Income ratio, representing the borrower's total debt payments as a percentage of their income.
24. **delinq_2yrs:** The number of times the borrower has been 30+ days past due on a payment in the last two years.
25. **earliest_cr_line:** The date when the borrower's earliest credit line was opened.
26. **inq_last_6mths:** The number of credit inquiries in the last six months.
27. **mths_since_last_delinq:** The number of months since the borrower's last delinquency(failed to pay the scheduled payment).
28. **mths_since_last_record:** The number of months since the last public record (e.g., bankruptcy, tax lien).
29. **open_acc:** The number of open credit lines or accounts.
30. **pub_rec:** The number of derogatory public records on the borrower's credit report.
31. **revol_bal:** The total revolving balance (credit card balance) of the borrower.
32. **revol_util:** The revolving utilization rate, representing the percentage of available credit being used.
33. **total_acc:** The total number of credit lines or accounts on the borrower's credit report.
34. **initial_list_status:** The initial listing status of the loan (e.g., whole or fractional).
35. **out_prncp:** The outstanding principal balance of the loan.
36. **out_prncp_inv:** The outstanding principal balance of the loan funded by investors.
37. **total_pymnt:** The total amount paid by the borrower.
38. **total_pymnt_inv:** The total amount paid by investors.
39. **total_rec_prncp:** The total amount of principal repaid.
40. **total_rec_int:** The total amount of interest paid.
41. **total_rec_late_fee:** The total amount of late fees paid.
42. **recoveries:** The amount recovered in case of loan default.
43. **collection_recovery_fee:** A fee associated with the collection of a defaulted loan.
44. **last_pymnt_d:** The date of the last payment received.
45. **last_pymnt_amnt:** The amount of the last payment received.
46. **next_pymnt_d:** The date of the next scheduled payment.
47. **last_credit_pull_d:** The date when the borrower's credit report was last pulled.
48. **collections_12_mths_ex_med:** The number of collections in the last 12 months excluding medical collections.
49. **mths_since_last_major_derog:** The number of months since the last major derogatory event.
50. **policy_code:** A code indicating the lending policy in effect at the time of the loan.
51. **application_type:** The type of application (e.g., individual or joint).
52. **annual_inc_joint:** The annual income of joint applicants (if applicable).
53. **dti_joint:** The Debt-to-Income ratio for joint applicants (if applicable).
54. **verification_status_joint:** The status of income verification for joint applicants (if applicable).
55. **acc_now_delinq:** The number of accounts on which the borrower is currently delinquent.
56. **tot_coll_amt:** The total collection amounts ever owed.
57. **tot_cur_bal:** The total current balance of all current credit lines.
58. **open_acc_6m:** The number of open credit lines in the last 6 months.
59. **open_il_6m:** The number of currently open installment accounts in the last 6 months.
60. **open_il_12m:** The number of installment accounts opened in the last 12 months.
61. **open_il_24m:** The number of installment accounts opened in the last 24 months.
62. **mths_since_rcnt_il:** The months since the most recent installment account opened.
63. **total_bal_il:** The total balance of all installment accounts.
64. **il_util:** The ratio of total current balance to the high credit/credit limit on all installments.
65. **open_rv_12m:** The number of revolving accounts opened in the last 12 months.
66. **open_rv_24m:** The number of revolving accounts opened in the last 24 months.
67. **max_bal_bc:** The maximum balance on a revolving account in the last 12 months.
68. **all_util:** The balance-to-limit ratio on all trades.
69. **total_rev_hi_lim:** The total revolving high credit/credit limit.
70. **inq_fi:** The number of personal finance-related inquiries.
71. **total_cu_tl:** The number of finance trades.
72. **inq_last_12m:** The number of credit inquiries in the last 12 months.
73. **default_ind:** A binary indicator (0 or 1) to indicate loan default (1) or non-default (0).

# 2. Data Cleaning and Preprocessing

There are some columns having around **75 - 100% of the value filled with NaN**. 
We can impute them with remaining other values if and only if the remaining data represents majority of the datasets imputing them with remaining 25% of the dataset might give false representation.

We are filling the null values(NaN) in the next_pymnt_d using the knowledge of the filled cells.

# 3. Data Visualization

![This is Screenshot of the total counts of Borrowers.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_Count_Borrowers.png)

Most borrowers have taken loan for debt consolidation, credit card and home
improvement.Debt consolidation can simplify repayments and potentially offer lower interest rates.Some borrowers are managing their short-term financial needs through credit cards.

Majority of borrowers have home_ownership status as rent and mortgage. Few of them own their house.

Most borrowers have loan rating as B and C. Grades B and C typically indicate a range where borrowers may have a satisfactory but not excellent creditworthiness.

![This is Screenshot of Piechats of Borrowers.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_PieCharts.png)

One-third of borrowers income is not verified. This could be due to various reasons such as self-employment, irregular income, or difficulties in obtaining employment verification documents.

Most borrowers have opted for short term loans, i.e, 36 months. Lenders may encourage or prefer shorter-term loans to mitigate their risk exposure. Opting for a shorter-term loan could mean higher monthly payments but also a quicker payoff, allowing borrowers to free up their budgets sooner.

Only 5% borrowers having default means stable economic conditions and reflects positively on the lender's ability to evaluate credit risk and make sound lending decisions.

![This is Screenshot of Borrowers from US State.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_UsState.png)

Majority Borrowers are from California, New York, Texas and Florida. The concentration of borrowers in these states could be influenced by higher population density and increased economic activity. Borrowers from economically stable regions may have greater confidence in their ability to repay loans.

![This is Screenshot of the all the Ranges of Borrowers.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_Range_Borrowers.png)

The median income of borrower is $65k which suggests a moderate income level within the borrower population. As the income increases, the count of borrowers decreases. Less borrowers in income range 0-25k suggests lender avoid giving loan due to high credit risk.

Amount of loan taken is mostly between 5k-15k. This may indicate addressing immediate expenses or making moderate-sized investments. From a lender's perspective, smaller loan amounts may be perceived as lower risk.

Borrowers with a debt-to-income ratio in the range of 15-20% could indicate a balanced approach to managing debt obligations. They are likely mindful of not overcommitting their income to debt repayment, leaving room for other financial priorities. From a lender's perspective, borrowers with a debt-to-income ratio in the 15-20% range may represent a lower risk profile.

![This is Screenshot of the all the Distributions.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_Distribution.png)

Interest rate is mostly between 10-15%. Lenders often use risk-based pricing, where the interest rate is influenced by the borrower's creditworthiness. Borrowers with rates in this range may have credit profiles that place them in a moderate risk category.
Monthly installment amount mostly ranges between $200-500.
Interest amount paid peaks around $2000.
Most of the borrowers have employment length greater than 10 years. Borrowers with employment lengths exceeding 10 years may be perceived as having stable and secure employment. Lenders may view such borrowers favorably, considering their long-term commitment to their current job.

# 4. Data preprocessing
* We can impute such columns where we have majority of known data.
* Imputing the missing values in 'tot_coll_bal', 'tot_cur_bal', 'revol_util' and
'total_rev_hi_lim' with the median of the filled cells, of its respective columns.

![This is Screenshot of the Home Owners.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_Homeowners.png)

we can see that home ownership is categorical column with values as:
* Rent
* Own
* Any
* Mortgage
* other
* None
We can remove instance of home ownership with type "Any" as it can be any of the type shown above.
---
* we need to find the time for which credit was issued.
* From the attributes defination 25 and 47 which is first date when borrower's credit line was issued and last date when borrower's credit line was reviewed by the lender for further decision.
* we can find the length by finding the difference between those two dates and further making new column of difference.

### Feature Selection
* we can see that there is high correlation between { total_payment_inv , total_pymnt , total_rec_prncp , installment}
* similarly between {loan_amnt , funded_amnt , funded_amnt_inv}
* same for {sub_grade , int_rate}
Thus we can keep one of them and drop rest to reduce the redundancy in feature which ultimately reduces overfitting of dataset.

We are now left with **30 columns** and **797436 instances** to work with.

# 5. Modeling
### 1. Random Forest Classifier

![This is Screenshot of the Randome Forest Classifier.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_RandomForest.png)

**Precision:** 1.000
**Recall:** 0.853
**Accuracy:** 0.975
**F1 Score:** 0.921

### 2. Logistic Regression

![This is Screenshot of the Logistic Regression.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_LogisticRegresion.png)

**Precision:** 0.973
**Recall:** 0.746
**Accuracy:** 0.953
**F1 Score:** 0.845

### 3. KNN Classifier

![This is Screenshot of the KNN Classifier.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_KNNClassifier.png)

**Precision:** 1.000
**Recall:** 0.000
**Accuracy:** 0.830
**F1 Score:** 0.000

This is the result of all the Regression Models:

![This is Screenshot of Results of all the Regression Models.](https://github.com/adityasinghania3802/Credit-Risk-Analysis/blob/main/Images/Screenshot_FinalResults.png)

# Contributors
* Krupesh Parmar (202101160) - Data Cleaning and Modeling
* Vrishin Shah (202101094) - Modeling
* Srikar Padaliya (202101095) - Data Preprocessing 
* Aditya Singhania (202101086) - Data Visualization 
* Kirtan Mevada (202101012) - Documentation
