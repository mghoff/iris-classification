## Logistic function by gradient descent with iris
# Clear environment
rm(list=ls())

# Import iris data
data(iris)

# Limit to two predictor variables for simplicity
df <- iris[, c("Sepal.Length", "Petal.Length", "Species")]

# Limit to where Species is in c('versicolor','verginica') only
df <- subset(df, df$Species %in% c("versicolor","virginica")); rm(iris)

# Separate data into predictors(X) and labels(Y)
X <- subset(df, select = -Species)
X <- scale(X) # Scale X to be within 0 and 1
Y <- ifelse(df$Species == "virginica", 1, 0)

# Get coefficients using standard glm() function
summary(mod <- glm(Y ~ X, family = binomial (link = "logit")))
coef(mod)
incpt <- coef(mod)[1] / -coef(mod)[3]
slope <- coef(mod)[2] / -coef(mod)[3]

plot(x = X[, 1], y = X[, 2], pch = 19, col = df$Species)
legend(x = 'topleft', legend = unique(df$Species), col = unique(df$Species), pch = 19)
abline(a = incpt, b = slope, lty = 2, col = 'blue')


## Begin gradient descent for Logistic regression
# Define sigmoid activation function
sigmoid <- function(z) {
    1 / (1 + exp(-z))
}

# Define the log-likelihood loss function
logLikelihood <- function(X.mat, y, beta_hat) {
    scores <- X.mat %*% beta_hat
    ll <- y * scores - log(1 +exp(scores))
    sum(ll)
}

logisticReg <- function(x, y, epochs, lr) {
    X.mat <- as.matrix(cbind(1, x))
    Y <- as.matrix(y)
    LR <- lr
    beta_hat <- matrix(0, nrow = ncol(X.mat))
    for (i in 1:epochs) {
        # Calculate residual
        residual <- (sigmoid(X.mat%*%beta_hat) - Y)
        # Update weights with GD
        delta <- (t(X.mat)%*%residual) * (1 / length(Y))
        beta_hat <- beta_hat - (LR*delta)
        # print(beta_hat)
    }
    print(logLikelihood(X.mat, y, beta_hat))
    print(beta_hat)
}

logisticReg(X, Y, 10000, 5)


