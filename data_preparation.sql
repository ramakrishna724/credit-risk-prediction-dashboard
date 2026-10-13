-- Credit Risk Analysis - Data Preparation & Feature Engineering
-- Impute missing values for employment length (median) and interest rate (median)
UPDATE credit_risk
SET person_emp_length = (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY person_emp_length) FROM credit_risk)
WHERE person_emp_length IS NULL;

UPDATE credit_risk
SET loan_int_rate = (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY loan_int_rate) FROM credit_risk)
WHERE loan_int_rate IS NULL;

-- Feature Engineering: Loan-to-Income Ratio
ALTER TABLE credit_risk ADD COLUMN loan_to_income_ratio NUMERIC;
UPDATE credit_risk SET loan_to_income_ratio = loan_amnt / NULLIF(person_income, 0);

-- Feature Engineering: Age Bands
ALTER TABLE credit_risk ADD COLUMN age_band VARCHAR(20);
UPDATE credit_risk SET age_band = 
  CASE 
    WHEN person_age < 25 THEN 'Under 25'
    WHEN person_age BETWEEN 25 AND 35 THEN '25-35'
    WHEN person_age BETWEEN 36 AND 50 THEN '36-50'
    ELSE 'Over 50'
  END;

-- Feature Engineering: Employment Length Category
ALTER TABLE credit_risk ADD COLUMN emp_length_cat VARCHAR(20);
UPDATE credit_risk SET emp_length_cat = 
  CASE 
    WHEN person_emp_length BETWEEN 0 AND 2 THEN '0-2 years'
    WHEN person_emp_length BETWEEN 3 AND 5 THEN '3-5 years'
    WHEN person_emp_length BETWEEN 6 AND 10 THEN '6-10 years'
    ELSE '10+ years'
  END;
