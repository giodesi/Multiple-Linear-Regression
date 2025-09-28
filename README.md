# **Multiple Linear Regression Analysis Project**

## **Project Overview**

This project provides a comprehensive implementation of multiple linear regression analysis through both a Jupyter notebook for educational exploration and a production-ready web application. The project demonstrates the complete workflow of building, training, evaluating, and deploying multiple linear regression models, which predict an outcome based on two or more independent variables.

The implementation focuses on accessibility and generalization, enabling users to apply advanced regression techniques to their own data without requiring extensive programming knowledge or statistical expertise. The project maintains scientific rigor while presenting complex concepts through intuitive visualizations and clear metrics.

## **Quick Links**

### **Live Application**

Access the fully functional web application deployed on Hugging Face Spaces: [**Multiple Linear Regression App**](https://huggingface.co/spaces/giodesi/Multiple_Linear_Regression)

### **Project Resources**

The complete implementation includes both educational materials and production-ready code. The Jupyter notebook provides a comprehensive walkthrough of multiple linear regression concepts with detailed explanations and visualizations, available at [Mulitple-Linear-Regression.ipynb](https://github.com/giodesi/Multiple-Linear-Regression/blob/main/Mulitple-Linear-Regression.ipynb). The source code for the Streamlit web application can be found in [app.py](https://github.com/giodesi/Multiple-Linear-Regression/blob/main/app.py).

## **Project Structure**

The project repository is organized as follows:

* **Mulitple-Linear-Regression.ipynb**: An educational Jupyter notebook that provides a step-by-step guide to multiple linear regression, from data exploration to model evaluation.  
* **app.py**: A production-ready Streamlit web application that allows users to upload their own datasets, train models, and generate predictions through an interactive interface.  
* **FuelConsumptionCo2.csv**: The sample dataset used in the notebook and as a default in the web application. It contains data on vehicle fuel consumption and CO2 emissions.  
* **requirements.txt**: A list of Python dependencies required to run the project.  
* **LICENSE**: The MIT License file governing the use of the project's source code.  
* **README.md**: This file, providing a comprehensive overview of the project.

## **Key Features**

The web application (app.py) offers a range of advanced features designed for both novice and experienced users:

* **Interactive Data Upload**: Users can upload datasets in CSV format or use the default fuel consumption dataset.  
* **Dynamic Feature Selection**: Allows for the selection of multiple independent variables (features) and a single dependent variable (target).  
* **Automated Data Preprocessing**: Includes optional data standardization (scaling) to improve model performance.  
* **Correlation Analysis**: Generates an interactive correlation heatmap to help users identify relationships between variables and select relevant features.  
* **Advanced Model Training**: Implements multiple linear regression using scikit-learn, splitting the data into training and testing sets for robust evaluation.  
* **Comprehensive Evaluation**: Reports key performance metrics, including **R-squared (**R2**)**, **Mean Squared Error (MSE)**, and **Mean Absolute Error (MAE)**.  
* **Interactive Visualizations**: Includes scatter plots and residual plots to help visualize model fit and diagnose potential issues.  
* **Single and Batch Predictions**: Users can make real-time predictions for single data points or upload a CSV file for batch predictions.

## **Installation and Usage**

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/giodesi/Multiple_Linear_Regression.git  
   cd Multiple_Linear_Regression
   ```

2. Install dependencies:  
   It is recommended to create a virtual environment first.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:  
   ```bash
   streamlit run app.py
   ```

4. Explore the Jupyter Notebook:  
   To delve into the theoretical and practical details, run the Jupyter notebook:  
   ```bash
   jupyter notebook Mulitple-Linear-Regression.ipynb
   ```

## **Methodology**

The project follows a structured machine learning workflow:

1. **Data Loading and Exploration**: The dataset is loaded using Pandas, and an initial analysis is performed to understand its structure, identify data types, and check for missing values.  
2. **Feature Selection**: Users can select multiple features that are believed to influence the target variable. A correlation matrix is used to guide this selection process.  
3. **Data Splitting**: The dataset is divided into training and testing sets (typically an 80/20 split) to ensure the model is evaluated on unseen data.  
4. **Model Training**: A multiple linear regression model is instantiated from `sklearn.linear_model.LinearRegression` and trained on the training dataset. The model learns the optimal coefficients for each selected feature.  
5. **Prediction and Evaluation**: The trained model is used to make predictions on the test set. The predictions are then compared to the actual values to evaluate the model's performance using standard regression metrics.

## **Evaluation Metrics**

The model's performance is assessed using the following standard metrics:

* **R-squared (**R2**)**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher value indicates a better fit.  
* **Mean Squared Error (MSE)**: Calculates the average of the squares of the errors. It is sensitive to large errors.  
* **Mean Absolute Error (MAE)**: Computes the average of the absolute differences between predicted and actual values. It provides a more direct interpretation of the average error magnitude.

## **Future Work**

Future development will focus on enhancing the application's analytical power and user experience. Key areas for improvement include automated feature selection algorithms, implementation of regularization techniques (e.g., Ridge and Lasso) to prevent overfitting, and advanced diagnostics for multicollinearity.

Additional statistical enhancements could incorporate confidence intervals for predictions, hypothesis testing for coefficient significance, influence diagnostics for outlier detection, and automated assumption checking with corrective recommendations. The interface could expand to include model comparison capabilities and export functionality for models and visualizations.

## **License**

This project is licensed under the MIT License - see the [LICENSE](https://github.com/giodesi/Multiple-Linear-Regression/blob/main/LICENSE) file for details. This permissive license allows for commercial use, modification, distribution, and private use while requiring only attribution and limiting liability.

## **Attribution**

The fuel consumption dataset used in the notebook originates from the Government of Canada's Open Data portal, demonstrating the application of linear regression to real-world environmental data. When using this project or its derivatives, please maintain appropriate attribution to both this project and the original data source.

## **Support and Documentation**

For optimal results, users should ensure their data contains meaningful linear relationships between variables and check for multicollinearity. The application provides informative error messages for common issues and includes contextual guidance throughout the interface to support users at all expertise levels.