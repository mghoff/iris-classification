# Clear Environment
rm(list=ls())

## Start Exploratory Data Analysis
# Load iris dataset
data(iris); df <- iris; rm(iris)

# Check out head of iris
head(df)

# summarise iris
summary(df)

# Check for values missing
sum(is.na(df))

## Limit data for simplicity
# Limit iris data to Petal.Length & Petal.Width & keep Species label
# df <- df[, c("Petal.Length", "Petal.Width", "Species")]

# Limit data to where Species is in c('versicolor','verginica') only
df <- subset(df, df$Species %in% c("versicolor","virginica"))


## Scatter Plot Section
# Define colors for plotting group
colors <- rep(NA, length(df$Species))
colors[which(df$Species == "versicolor")] <- "steelblue"
colors[which(df$Species == "virginica")] <- "red"

# Create scatter plot
plot(Petal.Length ~ Petal.Width, col = colors, data = df,
     main = "IRIS DATA | Petal Length by Width",
     xlab = "Petal Length", ylab = "Petal Width", 
     type = 'p', pch = 19)
legend('topleft', pch = 19,
       legend = c("versicolor", "virginica"), 
       col = c("steelblue", "red"))


## Start NeuralNet work
# Create binary classification of Species 
df$Label <- ifelse(df$Species == "versicolor", 1, 0)

# Drop Species from data
df <- subset(df, select = -Species)

# Define inputs(X) & outputs(Y) as matrices
X <- as.matrix(subset(df, select = -Label))
Y <- as.matrix(subset(df, select = Label))

#####################################################
################## MATH FUNCTIONS ###################
#####################################################
# define sigmoid function
sigmoid <- function(x){1 / (1 + exp(-x))}
# define derivative of sigmoid function
sigmoid_prime <- function(x){x * (1 - x)}
# define cost function
compute_cost <- function(pred, Y) {
    m <- length(Y)
    logProbs <- log(pred)*Y + (1 - Y)*log(1 - pred)
    cost <- round( - sum(logProbs) / m, digits = 6)
}

#####################################################
################# NEURAL NETWORK ####################
#####################################################
# Define NN function
nn_model <- function(x, y, n_hidden_layer_neurons = 6, lr = 0.01, epochs = 10000, output_neurons = 1, print_cost = FALSE) {
    # Layer dimensions
    inputlayer_neurons= ncol(x)
    hiddenlayer_neurons= n_hidden_layer_neurons

    ## Weight(W) and Bias(B) initialization
    # hidden (input) layer weights and biases
    W1 <- matrix(rnorm(inputlayer_neurons*hiddenlayer_neurons, mean=0, sd=1), 
                 inputlayer_neurons, hiddenlayer_neurons)
    b1 <- matrix(0, nrow(x), hiddenlayer_neurons) # trivial initialization
    # output layer weights and biases
    W2 <- matrix(rnorm(hiddenlayer_neurons*output_neurons, mean=0, sd=1), 
                 hiddenlayer_neurons, output_neurons)
    b2 <- matrix(0, nrow(x), output_neurons) # trivial initialization
    
    for(i in 1:epochs){
        ## forward propagation
        Z1 <- (x %*% W1) + b1       # hidden layer
        A1 <- sigmoid(Z1)           # hidden layer activation
        Z2 <- (A1 %*% W2) + b2      # output layer
        output <- A2 <- sigmoid(Z2) # output later activation
        
        ## Back Propagation
        outputError <- output - y                           # Calculate Output Error
        slope_output_layer <- sigmoid_prime(A2)    
        d_output <- outputError * slope_output_layer
        hiddenLayerError <- d_output %*% t(W2)              # Calculate hidden layer error
        slope_hidden_layer <- sigmoid_prime(A1)
        d_hiddenLayer <- hiddenLayerError * slope_hidden_layer
        
        # Update weights and biases
        W2 <- W2 - lr*(t(A1) %*% d_output)
        b2 <- b2 - lr*rowSums(d_output)
        W1 <- W1 - lr*(t(x) %*% d_hiddenLayer)
        b1 <- b1 - lr*rowSums(d_hiddenLayer)
            
        ## compute and print cost
        if (print_cost & i %% 1000 == 0) {
            print(paste0("Cost after iteration ", i, ": ", compute_cost(pred=output, Y=y)))
        }
    }
    print(paste("Final cost:", compute_cost(pred=output, Y=y)))
    weights <- list("W1" = W1, "W2" = W2)
    biases <- list("b1" = b1, "b2" = b2)
    predictions <- output
    list("Predictions" = predictions, 
         "Weights" = weights, 
         "Biases" = biases)
}

# Run NN model
trained_net_preds <- nn_model(x = X, y = Y, 
                              n_hidden_layer_neurons = 20, 
                              print_cost = TRUE)
