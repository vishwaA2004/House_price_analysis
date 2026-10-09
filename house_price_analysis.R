# ============================================================
# House Pricing Analysis using R
# Course : 23CAP317 - R Programming (Mini Project)
# Author : Vishwa A
# Data   : Ames Housing (2,930 residential sales, Ames, Iowa)
# Goal   : Predict Sale_Price from property characteristics
#          using Linear Regression, Random Forest and GLM models
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(corrplot)
  library(caret)
  library(ranger)
})

set.seed(2026)
dir.create("outputs", showWarnings = FALSE)

# ------------------------------------------------------------
# STAGE 0 : Helper functions (user-defined functions)
# ------------------------------------------------------------
rmse   <- function(actual, pred) sqrt(mean((actual - pred)^2))
mae    <- function(actual, pred) mean(abs(actual - pred))
r2     <- function(actual, pred) 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
skew   <- function(x) { x <- x[!is.na(x)]; mean((x - mean(x))^3) / sd(x)^3 }

# Variance Inflation Factor computed by hand (no extra package needed)
vif_table <- function(df_num) {
  out <- sapply(names(df_num), function(v) {
    fit <- lm(df_num[[v]] ~ ., data = df_num[, names(df_num) != v, drop = FALSE])
    1 / (1 - summary(fit)$r.squared)
  })
  sort(out, decreasing = TRUE)
}

# Evaluate a model and return a one-row data frame
evaluate <- function(name, actual, pred) {
  data.frame(Model = name,
             RMSE = round(rmse(actual, pred), 0),
             MAE  = round(mae(actual, pred), 0),
             R2   = round(r2(actual, pred), 4))
}

# ------------------------------------------------------------
# STAGE 1 : Loading the dataset
# ------------------------------------------------------------
house <- read.csv("data/ames_housing.csv", stringsAsFactors = TRUE)
cat("Dimensions:", dim(house), "\n")

# Keep a focused, meaningful set of predictors
num_cols <- c("Lot_Frontage", "Lot_Area", "Year_Built", "Year_Remod_Add",
              "Mas_Vnr_Area", "Total_Bsmt_SF", "Gr_Liv_Area", "Full_Bath",
              "Half_Bath", "Bedroom_AbvGr", "TotRms_AbvGrd", "Fireplaces",
              "Garage_Cars", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF",
              "Longitude", "Latitude")
cat_cols <- c("Neighborhood", "Bldg_Type", "House_Style", "Overall_Cond",
              "Central_Air", "Garage_Type")
df <- house[, c(num_cols, cat_cols, "Sale_Price")]

# ------------------------------------------------------------
# Data structures demo (vector, list, matrix, factor, data frame)
# ------------------------------------------------------------
price_vec   <- df$Sale_Price                                  # vector
project_info <- list(title   = "House Pricing Analysis",     # list
                     author  = "Vishwa A",
                     rows    = nrow(df),
                     models  = c("Linear Regression", "Random Forest", "GLM"))
price_by_ac <- tapply(df$Sale_Price, df$Central_Air, median)  # named vector
str(project_info)
print(price_by_ac)

# ------------------------------------------------------------
# STAGE 2 : Structure and summary
# ------------------------------------------------------------
str(df)
print(summary(df[, c("Sale_Price", "Gr_Liv_Area", "Year_Built", "Lot_Area")]))

# ------------------------------------------------------------
# STAGE 3 : Data cleaning
# ------------------------------------------------------------
na_summary <- colSums(is.na(df))
print(na_summary[na_summary > 0])

# Hidden missing values: in this dataset a missing Lot_Frontage is stored as 0
cat("Lot_Frontage recorded as 0:", sum(df$Lot_Frontage == 0), "rows\n")
df$Lot_Frontage[df$Lot_Frontage == 0] <- NA
cat("Missing values after recoding zeros:", sum(is.na(df)), "\n")

# Control structure: loop over numeric columns, impute NA with median
for (col in num_cols) {
  if (any(is.na(df[[col]]))) {
    med <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med
    cat("Imputed", col, "with median", med, "\n")
  }
}
cat("Missing values after cleaning:", sum(is.na(df)), "\n")
cat("Duplicate rows:", sum(duplicated(df)), "\n")

# Feature engineering
df$House_Age  <- 2010 - df$Year_Built          # data ends in 2010
df$Remod_Age  <- 2010 - df$Year_Remod_Add
df$Total_Bath <- df$Full_Bath + 0.5 * df$Half_Bath
df$Log_Price  <- log(df$Sale_Price)

# ------------------------------------------------------------
# STAGE 4 : Data visualisation
# ------------------------------------------------------------
cat("Skewness of Sale_Price:", round(skew(df$Sale_Price), 3), "\n")
cat("Skewness of log(Sale_Price):", round(skew(df$Log_Price), 3), "\n")

p1 <- ggplot(df, aes(Sale_Price)) +
  geom_histogram(bins = 50, fill = "#2E6F8E", colour = "white") +
  scale_x_continuous(labels = function(x) paste0("$", x / 1000, "k")) +
  ggtitle("Distribution of Sale Price") + labs(x = "Sale price", y = "Count")
ggsave("outputs/01_price_hist.png", p1, width = 7, height = 4.5, dpi = 150)

p2 <- ggplot(df, aes(Log_Price)) +
  geom_histogram(bins = 50, fill = "#C9742B", colour = "white") +
  ggtitle("Distribution of log(Sale Price)") + labs(x = "log(Sale price)", y = "Count")
ggsave("outputs/02_logprice_hist.png", p2, width = 7, height = 4.5, dpi = 150)

num_vars <- df[, c("Sale_Price", "Gr_Liv_Area", "Total_Bsmt_SF", "Garage_Area",
                   "Garage_Cars", "Total_Bath", "TotRms_AbvGrd", "House_Age",
                   "Remod_Age", "Fireplaces", "Lot_Area", "Mas_Vnr_Area",
                   "Bedroom_AbvGr", "Open_Porch_SF", "Wood_Deck_SF")]
cor_mat <- cor(num_vars, use = "pairwise.complete.obs")
png("outputs/03_corrplot.png", width = 1000, height = 900, res = 130)
corrplot(cor_mat, method = "color", type = "upper", tl.cex = 0.7,
         tl.col = "black", addCoef.col = "black", number.cex = 0.45)
dev.off()
print(round(sort(cor_mat[, "Sale_Price"], decreasing = TRUE), 3))

p4 <- ggplot(df, aes(Gr_Liv_Area, Sale_Price)) +
  geom_point(alpha = 0.35, colour = "#2E6F8E") +
  geom_smooth(method = "lm", colour = "#B03A2E", se = FALSE) +
  scale_y_continuous(labels = function(x) paste0("$", x / 1000, "k")) +
  ggtitle("Living area vs Sale Price") + labs(x = "Above-ground living area (sq ft)", y = "Sale price")
ggsave("outputs/04_area_vs_price.png", p4, width = 7, height = 4.5, dpi = 150)

# Boxplots for outlier checking
box_long <- stack(df[, c("Gr_Liv_Area", "Total_Bsmt_SF", "Garage_Area", "Lot_Area")])
box_long$values <- log1p(box_long$values)      # log(1 + x) keeps houses with no basement/garage
p5 <- ggplot(box_long, aes(ind, values)) + geom_boxplot(fill = "#A9CCE3") +
  ggtitle("Boxplots for outlier checking (log(1 + x) scale)") +
  labs(x = NULL, y = "log(1 + value)")
ggsave("outputs/05_boxplots.png", p5, width = 7, height = 4.5, dpi = 150)

top_nb <- names(sort(table(df$Neighborhood), decreasing = TRUE))[1:10]
p6 <- ggplot(df[df$Neighborhood %in% top_nb, ], aes(reorder(Neighborhood, Sale_Price, median), Sale_Price)) +
  geom_boxplot(fill = "#F5CBA7") + coord_flip() +
  scale_y_continuous(labels = function(x) paste0("$", x / 1000, "k")) +
  ggtitle("Sale Price by Neighborhood (10 most common)") + labs(x = NULL, y = "Sale price")
ggsave("outputs/06_neighborhood.png", p6, width = 7, height = 4.8, dpi = 150)

# Outlier count by the 1.5 x IQR rule (function + sapply)
count_outliers <- function(x) { q <- quantile(x, c(.25, .75)); i <- diff(q)
  sum(x < q[1] - 1.5 * i | x > q[2] + 1.5 * i) }
print(sapply(df[, c("Sale_Price", "Gr_Liv_Area", "Lot_Area", "Garage_Area")], count_outliers))

# ------------------------------------------------------------
# STAGE 5 : Multicollinearity check
# ------------------------------------------------------------
vif_in <- df[, c("Gr_Liv_Area", "Total_Bsmt_SF", "Garage_Area", "Garage_Cars",
                 "Total_Bath", "TotRms_AbvGrd", "Bedroom_AbvGr", "House_Age",
                 "Remod_Age", "Fireplaces", "Lot_Area")]
vif_vals <- vif_table(vif_in)
print(round(vif_vals, 2))

# ------------------------------------------------------------
# STAGE 6 : Train / test split (80 / 20)
# ------------------------------------------------------------
model_df <- df[, !(names(df) %in% c("Year_Built", "Year_Remod_Add", "Full_Bath", "Half_Bath"))]
trainIndex <- createDataPartition(model_df$Sale_Price, p = 0.8, list = FALSE)
train <- model_df[trainIndex, ]
test  <- model_df[-trainIndex, ]
cat("Train rows:", nrow(train), "| Test rows:", nrow(test), "\n")

# Drop test rows whose factor level never appears in training
for (f in cat_cols) {
  unseen <- !(test[[f]] %in% unique(train[[f]]))
  if (any(unseen)) { cat("Dropping", sum(unseen), "test rows with unseen level in", f, "\n")
    test <- test[!unseen, ] }
}
for (f in cat_cols) { train[[f]] <- droplevels(train[[f]]); test[[f]] <- factor(test[[f]], levels = levels(train[[f]])) }

results <- list()      # list to collect model results

# ------------------------------------------------------------
# STAGE 7a : Linear regression (on log price)
# ------------------------------------------------------------
lm_model <- lm(Log_Price ~ . - Sale_Price, data = train)
print(summary(lm_model))
lm_pred <- exp(predict(lm_model, newdata = test))          # back-transform to dollars
results$lm <- evaluate("Linear Regression (log price)", test$Sale_Price, lm_pred)

# Residual diagnostics
res <- residuals(lm_model)
cat("Shapiro-Wilk on residuals (sample of 2000): p =",
    signif(shapiro.test(sample(res, 2000))$p.value, 3), "\n")
p7 <- ggplot(data.frame(fit = fitted(lm_model), res = res), aes(fit, res)) +
  geom_point(alpha = 0.3, colour = "#2E6F8E") + geom_hline(yintercept = 0, colour = "#B03A2E") +
  ggtitle("Linear model: residuals vs fitted") + labs(x = "Fitted log price", y = "Residual")
ggsave("outputs/07_lm_residuals.png", p7, width = 7, height = 4.5, dpi = 150)

# ------------------------------------------------------------
# STAGE 7b : Random forest regression
# ------------------------------------------------------------
rf_model <- ranger(Sale_Price ~ . - Log_Price, data = train, num.trees = 300,
                   importance = "impurity", seed = 2026)
rf_pred  <- predict(rf_model, data = test)$predictions
results$rf <- evaluate("Random Forest", test$Sale_Price, rf_pred)

imp <- sort(rf_model$variable.importance, decreasing = TRUE)
print(head(imp, 10))
imp_df <- data.frame(Variable = names(imp)[1:12], Importance = imp[1:12])
p8 <- ggplot(imp_df, aes(reorder(Variable, Importance), Importance)) +
  geom_col(fill = "#2E6F8E") + coord_flip() +
  ggtitle("Random forest: top 12 variable importances") + labs(x = NULL)
ggsave("outputs/08_rf_importance.png", p8, width = 7, height = 4.5, dpi = 150)

# ------------------------------------------------------------
# STAGE 7c : Generalized linear model (Gamma family, log link)
# Sale price is positive and right-skewed, so a Gamma GLM is natural.
# ------------------------------------------------------------
glm_model <- glm(Sale_Price ~ . - Log_Price, data = train,
                 family = Gamma(link = "log"))
print(summary(glm_model))
glm_pred <- predict(glm_model, newdata = test, type = "response")
results$glm <- evaluate("GLM (Gamma, log link)", test$Sale_Price, glm_pred)

# ------------------------------------------------------------
# STAGE 7d : Logistic regression - classify "High price" (above median)
# ------------------------------------------------------------
cut_price <- median(train$Sale_Price)
train_cl <- train; test_cl <- test
train_cl$HighPrice <- as.integer(train_cl$Sale_Price > cut_price)
test_cl$HighPrice  <- as.integer(test_cl$Sale_Price  > cut_price)
logit_model <- glm(HighPrice ~ . - Sale_Price - Log_Price, data = train_cl,
                   family = binomial(link = "logit"))
logit_prob  <- predict(logit_model, newdata = test_cl, type = "response")
logit_class <- factor(ifelse(logit_prob > 0.5, 1, 0), levels = c(0, 1))
conf <- confusionMatrix(logit_class, factor(test_cl$HighPrice, levels = c(0, 1)), positive = "1")
print(conf)

# ------------------------------------------------------------
# STAGE 8 : Model comparison and actual-vs-predicted plots
# ------------------------------------------------------------
comparison <- do.call(rbind, results)
rownames(comparison) <- NULL
print(comparison)
write.csv(comparison, "outputs/model_comparison.csv", row.names = FALSE)

plot_avp <- function(actual, pred, title, file) {
  p <- ggplot(data.frame(a = actual, p = pred), aes(a, p)) +
    geom_point(alpha = 0.35, colour = "#2E6F8E") +
    geom_abline(slope = 1, intercept = 0, colour = "#B03A2E") +
    scale_x_continuous(labels = function(x) paste0("$", x / 1000, "k")) +
    scale_y_continuous(labels = function(x) paste0("$", x / 1000, "k")) +
    ggtitle(title) + labs(x = "Observed price", y = "Predicted price")
  ggsave(file, p, width = 6.5, height = 5, dpi = 150)
}
plot_avp(test$Sale_Price, lm_pred,  "Linear Regression: Predicted vs Actual", "outputs/09_lm_avp.png")
plot_avp(test$Sale_Price, rf_pred,  "Random Forest: Predicted vs Actual",     "outputs/10_rf_avp.png")
plot_avp(test$Sale_Price, glm_pred, "GLM (Gamma): Predicted vs Actual",       "outputs/11_glm_avp.png")

# ------------------------------------------------------------
# STAGE 9 : Predict the price of a new house (function + prediction)
# ------------------------------------------------------------
predict_new_house <- function(new_house) {
  list(linear = exp(predict(lm_model, new_house)),
       forest = predict(rf_model, data = new_house)$predictions,
       glm    = predict(glm_model, new_house, type = "response"))
}
new_house <- test[1, ]
cat("Actual price of sample house:", new_house$Sale_Price, "\n")
print(predict_new_house(new_house))

# ------------------------------------------------------------
# Save everything printed values need for the report
# ------------------------------------------------------------
saveRDS(list(comparison = comparison, vif = vif_vals, importance = imp,
             cor_price = sort(cor_mat[, "Sale_Price"], decreasing = TRUE),
             conf = conf$overall, conf_byclass = conf$byClass, conf_table = conf$table,
             cutoff = cut_price, n_train = nrow(train), n_test = nrow(test)),
        "outputs/results.rds")
cat("Done.\n")
