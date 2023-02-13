# Various_ML_Classification_Techniques_on_HouseBuyersClassification

Logistic Regression

The goal of logistic regression is to model the relationship between a set of independent variables and a binary dependent variable.In logistic regression, the independent variables are combined linearly to predict the log-odds of the dependent variable taking on a specific value. The log-odds are then transformed into a probability using the logistic function, which outputs a value between 0 and 1. The predicted probability is used to classify the observation into one of the two categories.

 K-Nearest Neighbors
 
The basic idea behind KNN is to define the similarity between two points based on their distances in the feature space. Given a new data point, KNN finds its k nearest neighbors in the training data and assigns the new point to the class that is most frequent among these neighbors. The number of nearest neighbors, k, is a user-defined hyperparameter and must be set prior to fitting the model.

Support Vector Machine

The intuition behind SVM is to find the hyperplane that has the maximum margin, which is the distance between the hyperplane and the closest data points from both classes. The margin represents the region of certainty around the decision boundary, and the goal is to maximize this margin. By maximizing the margin, SVM aims to minimize the misclassification rate and make the classifier more robust to noise in the data.SVM can handle non-linearly separable data by transforming the input space into a higher dimensional space where a linear boundary can be drawn. This transformation is performed using a technique called kernel trick. The choice of the kernel function used will determine the shape of the boundary in the transformed space, and therefore the boundary shape in the original space.

Multi Layer Perceptron

A multi-layer perceptron (MLP) is a type of artificial neural network that is composed of multiple layers of interconnected nodes, or neurons. The basic building block of an MLP is a single neuron, which receives input from other neurons, processes it, and then produces an output. In an MLP, the input layer receives the input features, the hidden layers process the input data, and the output layer produces the final predictions.
The processing of the input data in each neuron is performed by applying an activation function to the weighted sum of the inputs. Common activation functions used in MLPs include the sigmoid, tanh, and rectified linear unit (ReLU) functions. The weights and biases of each neuron are optimized during the training phase to minimize the difference between the predicted output and the actual output.
The training process starts with a random initialization of the weights and biases, and then the model updates these parameters iteratively based on the training data. This process is performed using an optimization algorithm, such as gradient descent, which adjusts the weights and biases in the direction that minimizes the error. The training continues until the error reaches a minimum or a maximum number of iterations is reached.

Gaussian Process Classifier 

A Gaussian Process Classifier (GPC) is a probabilistic machine learning method used for classification tasks. The basic idea behind a Gaussian Process Classifier is to model the distribution of the inputs and outputs using a Gaussian process. A Gaussian process is a collection of random variables with a certain mean and covariance function. The mean function is used to model the average behavior of the data, while the covariance function models the dependencies between the inputs and outputs.The GPC models the probability distribution of the output given the input data. Given a new input, the GPC can make predictions about the class label of the output based on the probabilities it has learned from the training data. The predictions made by the GPC are not point estimates but rather a distribution over the possible outputs.

Decision tree classifier

A decision tree classifier is a tree-based model used for supervised learning. It works by breaking down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. Each internal node of the tree corresponds to a test on an attribute, each branch corresponds to the outcome of a test and each leaf node assigns a class label to a particular instance. The goal of this process is to construct a decision tree that accurately predicts the class label of an instance based on its attributes. The intuition behind decision tree classifiers is to divide the feature space into smaller regions and assign the class labels to these regions in such a way that the instances within a region have similar class labels.

Random Forest Classifier

Random Forest Classifier is an ensemble machine learning algorithm that uses decision trees as the building blocks. An ensemble is a combination of multiple models to improve the performance of a single model. Random Forest Classifier creates multiple decision trees during the training phase and then combines their predictions to produce the final output. The algorithm randomly selects a subset of the features for each tree, so each tree will have a different structure. During the prediction phase, each tree will make a prediction, and the final output will be the average or majority vote of all trees. This makes the algorithm more robust and less prone to overfitting compared to a single decision tree. The intuition behind the Random Forest Classifier is that by combining the predictions of multiple trees, it can better capture the non-linear relationships between the features and the target variable and produce more accurate predictions.



