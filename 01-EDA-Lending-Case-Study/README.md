# Lending Club Case Study
> Lending Club is a company that offers different kinds of loans to people in cities. The company has to decide whether to approve or reject a loan application based on the applicant's profile.
>The company faces two kinds of risks when making this decision:
> * If the applicant can pay back the loan, then rejecting the loan means losing a potential customer for the company
> * If the applicant cannot pay back the loan, i.e. he/she is likely to default, then approving the loan may cause a financial loss for the company


## Table of Contents
* [Problem Statement](#problem-statement)
* [General Info](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## Problem Statement
### Business Understanding
The company can make two kinds of decisions when someone applies for a loan:
Loan approved: The company gives the loan, and there are 3 possible outcomes as follows:
> * Fully paid: The applicant pays back the whole loan (the principal and the interest rate)
> * Current: The applicant is still paying the instalments, i.e. the loan period is not over yet. These applicants are not considered as 'defaulted'.
> * Charged-off: The applicant fails to pay the instalments on time for a long time, i.e. he/she defaults on the loan

Loan rejected: The company does not give the loan (because the applicant does not qualify for it etc.). Since the loan was not given, the company does not have any transactional history of those applicants and neither does this dataset

### Business Objectives
This company is the big online platform for lending money, offering personal loans, business loans, etc. Borrowers can get lower interest rate loans through a quick online interface.

Like most other lenders, the biggest source of financial loss (called credit loss) for this company is lending money to ‘risky’ applicants who don't pay back or run away with the money they owe. In other words, the borrowers who default are the ones who cause the most loss to the lenders. In this case, the customers marked as 'charged-off' are the 'defaulters'.

If the company can identify these risky loan applicants, then it can avoid giving them loans and reduce the amount of credit loss. The goal of this case study is to identify these applicants using EDA.

In other words, the company wants to find out the factors (or driver variables) that influence loan default, i.e. the variables that are strong indicators of default. The company can use this knowledge for its portfolio and risk assessment.

To improve your understanding of the domain, you are suggested to do some independent research on risk analytics (you should understand the types of variables and their importance).

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## General Information
- Steps for EDA :
<ol>
    <li>Data Understanding</li>
    <li>Data Cleaning</li>
    <li>Univariate Analysis</li>
    <li>Bivariate Analysis</li>
    <li>Multivariate Analysis</li>
    <li>Conclusion</li>
</ol>
- Data Set : Loan Lending Club loans 

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Important Data Factors
<table width="889">
<tbody>
<tr>
<td width="237">
<p><strong>Loan related</strong></p>
</td>
<td width="326">
<p><strong>Customer related</strong></p>
</td>
<td width="326">
<p><strong>Customer behaviour related</strong></p>
</td>
</tr>
<tr>
<td width="237">
<p>loan_amnt</p>
</td>
<td width="326">
<p>home_ownership</p>
</td>
<td width="326">
<p>delinq_2yrs</p>
</td>
</tr>
<tr>
<td width="237">
<p>funded_amnt</p>
</td>
<td width="326">
<p>term</p>
</td>
<td width="326">
<p>pub_rec_bankruptcies</p>
</td>
</tr>
<tr>
<td width="237">
<p>term</p>
</td>
<td width="326">
<p>verification_status</p>
</td>
<td width="326">
<p>revol_bal</p>
</td>
</tr>
<tr>
<td width="237">
<p>int_rate</p>
</td>
<td width="326">
<p>purpose</p>
</td>
<td width="326">
<p>recoveries</p>
</td>
</tr>
<tr>
<td width="237">
<p>installment</p>
</td>
<td width="326">
<p>grade</p>
</td>
<td width="326">
<p>purpose</p>
</td>
</tr>
<tr>
<td width="237">
<p>annual_inc</p>
</td>
<td width="326">
<p>pub_rec_bankruptcies</p>
</td>
<td width="326">
<p>earliest credit line</p>
</td>
</tr>
<tr>
<td width="237">
<p>dti</p>
</td>
<td width="326">
<p>issue_d_month</p>
</td>
<td width="326">
<p>loan_status</p>
</td>
</tr>
<tr>
<td width="237">
<p>open_acc</p>
</td>
<td width="326">
<p>Annual salary</p>
</td>
<td width="326">&nbsp;</td>
</tr>
<tr>
<td width="237">
<p>pub_rec</p>
</td>
<td width="326">
<p>Grade</p>
</td>
<td width="326">&nbsp;</td>
</tr>
<tr>
<td width="237">
<p>total_acc</p>
</td>
<td width="326">&nbsp;</td>
<td width="326">&nbsp;</td>
</tr>
<tr>
<td width="237">
<p>total_pymnt</p>
</td>
<td width="326">&nbsp;</td>
<td width="326">&nbsp;</td>
</tr>
<tr>
<td width="237">
<p>total_pymnt_inv</p>
</td>
<td width="326">&nbsp;</td>
<td width="326">&nbsp;</td>
</tr>
<tr>
<td width="237">
<p>total_rec_prncp</p>
</td>
<td width="326">&nbsp;</td>
<td width="326">&nbsp;</td>
</tr>
<tr>
<td width="237">
<p>total_rec_int</p>
</td>
<td width="326">&nbsp;</td>
<td width="326">&nbsp;</td>
</tr>
</tbody>
</table>

## Conclusions
<div>
<ul>
<li>14% of the borrowers defaulted on their loans.</li>
<li>Only 57% of the loans were recovered.</li>
<li>Out of the fully recovered loans, 17% is profit.</li>
<li>The loan business is vulnerable to losses from defaults and low recoveries, and it needs to monitor its default rate closely and take preventive measures to keep it below 16.49%.</li>
<li>Borrowers who default on their loans tend to have higher interest rates, a moderate number of open accounts, and lower total payments and investments. They often have a delinquency in the past two to three years.</li>
<li>Loan amount vs term: Borrowers with a 60-month term are more likely to default than those with a 36-month term.</li>
<li>Interest rate vs term: Borrowers with a 36-month term default at lower interest rates than those with a 60-month term.</li>
<li>Annual income vs verification status: Borrowers who default have similar incomes regardless of verification status, but verified borrowers have slightly higher incomes.</li>
<li>Loan amount vs verification status: Borrowers who default have different loan amounts depending on verification status, and verified borrowers have the highest loan amounts.</li>
<li>Annual income vs purpose: Borrowers who default have different incomes depending on the purpose, and those who borrow for home improvement or house purchase have the highest incomes.</li>
<li>Loan amount vs purpose: Borrowers who default have different loan amounts depending on the purpose, and those who borrow for debt consolidation, credit cards, or small businesses have the highest loan amounts.</li>
<li>Grade vs loan amount: Borrowers who default have different loan amounts depending on grade, and those who have lower grades have higher loan amounts.</li>
<li>Grade vs int_rate: Borrowers who default have different interest rates depending on grade, and those who have lower grades have higher interest rates.</li>
<li>Grade vs total_pymnt: Borrowers who default have different total payments depending on grade, and those who have lower grades have higher total payments.</li>
<li>Int_rate vs issue_d_year: Borrowers who default have different interest rates depending on the year they issued their loans.</li>
</ul>
</div>

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- numpy - 1.24.4
- pandas - 2.0.3
- matplotlib - 3.7.1
- seaborn - 0.12.2


<!-- As the library versions keep on changing, it is recommended to mention the version of the library used in this project -->

## Acknowledgements
Give credit here.
- This project was a group case study for an online advanced course.
- https://www.geeksforgeeks.org/
- https://seaborn.pydata.org/
- https://pandas.pydata.org/
- https://learn.upgrad.com/


## Contact
Created by [@saurabhbhargava6 and @badarihp] - feel free to contact us!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
