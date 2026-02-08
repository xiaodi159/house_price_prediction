# House Price Prediction

## Project Overview
This project aims to develop a model that predicts house prices based on various features such as location, size, number of bedrooms, and other relevant attributes. The model is trained using a dataset containing historical housing prices and will utilize machine learning techniques to generate accurate predictions for new listings.

## Installation
To get started with the project, you need to install the required dependencies. Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/xiaodi159/house_price_prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd house_price_prediction
    ```
3. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
You can use the model to predict house prices by running the following command:
```bash
python predict.py <input_data>
```
Replace `<input_data>` with the path to your input file containing the necessary features.

## Model Architecture
The model uses a combination of linear regression and decision tree algorithms to improve prediction accuracy. The architecture consists of:
- Data Preprocessing: Cleaning and transforming the feature set.
- Feature Selection: Identifying the most important features that impact house prices.
- Model Training: Utilizing training datasets to train different algorithms.
- Model Evaluation: Testing the model's performance on a validation dataset.

## Results
The model achieved an accuracy of 85% on the test dataset with a mean absolute error of $5000 in price prediction. Further improvements can be made by incorporating more features and utilizing different algorithms.

## Additional Information
For any questions or contributions to the project, please contact the repository owner or create an issue on GitHub.