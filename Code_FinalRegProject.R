# Code for Final Regression Project
# Name: Aryan Jalli
# Import all of the required libraries 

library(tidyverse)
library(forcats)
library(gbm)
library(ggplot2)

set.seed(563) # SEED USED THROUGHOUT THE PROJECT

# we have to load in the data 

train <- read.csv("C:/Users/pawan/aryan/train.csv")

# Remove ID column if present
if ("Id" %in% names(train)) train <- train %>% select(-Id)
if ("ID" %in% names(train)) train <- train %>% select(-ID)

# Convert character variables to factors
train <- train %>% mutate(across(where(is.character), as.factor))

# Conduct EDA

# Create a SalePrice distribution -> This is a simple histogram
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(bins = 40, fill = "steelblue", color = "black") +
  labs(title = "Distribution of SalePrice")

# Since our response variable is in Y=log(Saleprice) -> create Log(SalePrice) distribution
ggplot(train, aes(x = log(SalePrice))) +
  geom_histogram(bins = 40, fill = "darkgreen", color = "black") +
  labs(title = "Distribution of log(SalePrice)")

# calculate missing data summary
missing_summary <- train %>%
  summarise(across(everything(), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(),
               names_to = "Variable",
               values_to = "PctMissing") %>%
  arrange(desc(PctMissing))

head(missing_summary, 15)

# Find the highest correlation features with SalePrice (conducting on numeric only)
numeric_vars <- train %>% select(where(is.numeric))

cor_summary <- numeric_vars %>%
  summarise(across(-SalePrice, ~ cor(., SalePrice, use = "complete.obs"))) %>%
  pivot_longer(everything(),
               names_to = "Variable",
               values_to = "Correlation") %>%
  arrange(desc(abs(Correlation)))

head(cor_summary, 10)

# Scatterplot relationship between SalePrice vs Year Built. We see that year built has a slight 
# non-linear relationship with the sale price. curvature goes up, doesnt follow a stright line
ggplot(train, aes(x = Year.Built, y = SalePrice)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "SalePrice vs Year Built")

# Scatterplot relationship between SalePrice vs Living Area. There is a non-linear relationship between these 
# variables since additional square footage provides dminishing marginal value as home size increases. This leads
# a curvature in the price size relationship. 
ggplot(train, aes(x = Gr.Liv.Area, y = SalePrice)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "SalePrice vs Above-Ground Living Area")

# Boxplot of SalePrice by Overall Quality
ggplot(train, aes(x = factor(Overall.Qual), y = SalePrice)) +
  geom_boxplot(fill = "red") +
  labs(title = "SalePrice by Overall Quality")

# Clean our dataset

# Median imputation for numeric variables. Median values of each feature will replace values not present in that feature.  
impute_median <- function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  x
}

# Mode imputation for categorical variables. Most common mode value will take on the missing values for that feature.
impute_mode <- function(x) {
  tab <- table(x)
  x[is.na(x)] <- names(tab)[which.max(tab)]
  x
}

# apply these changes onto our dataset
train_clean <- train %>%
  mutate(
    across(where(is.numeric), impute_median),
    across(where(is.factor), impute_mode)
  ) %>%
  mutate(across(where(is.factor), fct_drop)) %>%
  select(where(~ !(is.factor(.) && nlevels(.) < 2)))

# compute feature engineering for special non-linear columns 

train_clean <- train_clean %>%
  mutate(
    TotalSF = X1st.Flr.SF + X2nd.Flr.SF + Total.Bsmt.SF,
    Age = 2025 - Year.Built,
    TotalBaths = Full.Bath + Half.Bath + Bsmt.Full.Bath + Bsmt.Half.Bath,
    PorchSF = Open.Porch.SF + Enclosed.Porch + Screen.Porch + X3Ssn.Porch,
    log_SalePrice = log(SalePrice)
  )

# Implement our model
# MODEL SELECTED: Gradient Boosting Machine (GBM)

# Set our response variable
predictors <- setdiff(names(train_clean), c("SalePrice", "log_SalePrice"))

formula_gbm <- as.formula(
  paste("log_SalePrice ~", paste(predictors, collapse = " + "))
)

# more information about this described in the report
gbm_model <- gbm(
  formula = formula_gbm,
  data = train_clean,
  distribution = "gaussian",
  n.trees = 10000,
  shrinkage = 0.005,            # This is our learning rate
  interaction.depth = 6,
  bag.fraction = 0.7,
  n.minobsinnode = 10,
  train.fraction = 1.0,
  verbose = FALSE
)

best_iter <- gbm.perf(gbm_model, method = "OOB", plot.it = FALSE)

# Load in our test data from kaggle
# After loading we have to clean the data again

test <- read.csv("C:/Users/pawan/aryan/test.csv")
test_ids <- test$ID

test <- test %>% mutate(across(where(is.character), as.factor))

# Similar cleaning process that we used on the training set
test_clean <- test %>%
  mutate(
    across(where(is.numeric), impute_median),
    across(where(is.factor), impute_mode)
  ) %>%
  mutate(across(where(is.factor), fct_drop)) %>%
  mutate(
    TotalSF = X1st.Flr.SF + X2nd.Flr.SF + Total.Bsmt.SF,
    Age = 2025 - Year.Built,
    TotalBaths = Full.Bath + Half.Bath + Bsmt.Full.Bath + Bsmt.Half.Bath,
    PorchSF = Open.Porch.SF + Enclosed.Porch + Screen.Porch + X3Ssn.Porch
  )

# Force factor levels to match training
for (col in names(test_clean)) {
  if (col %in% names(train_clean) && is.factor(train_clean[[col]])) {
    lv <- levels(train_clean[[col]])
    test_clean[[col]] <- factor(test_clean[[col]], levels = lv)
    test_clean[[col]][is.na(test_clean[[col]])] <- lv[1]
  }
}

# Predict the house sale prices plus create a submission doc in excel

pred_log <- predict(gbm_model, newdata = test_clean, n.trees = best_iter)
pred <- exp(pred_log)

submission <- data.frame(                  # submission file will have 2 columns which are ID and SalePrice
  ID = test_ids,
  SalePrice = as.numeric(pred)
)

write.csv(submission, "submission.csv", row.names = FALSE)
head(submission) # just a quick overview of the excel



