# AI-SE-week2-ML-_model
Week2 ML model that analyses Crop yield

# Crop Yield Prediction Model

## Project Overview

This project develops a machine learning model to predict crop yield based on historical agricultural data, weather information, and key environmental factors. The goal is to provide valuable insights for farmers, agricultural organizations, and policymakers to improve planning, optimize resource allocation, and enhance resilience in the face of climate variability.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Sustainable Development Goals (SDGs) Addressed](#sustainable-development-goals-sdgs-addressed)
3.  [Machine Learning Approach](#machine-learning-approach)
4.  [Data](#data)
5.  [Model Performance](#model-performance)
6.  [Ethical Considerations](#ethical-considerations)
7.  [Setup and Installation](#setup-and-installation)
8.  [Usage](#usage)
9.  [Future Enhancements](#future-enhancements)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)

## Sustainable Development Goals (SDGs) Addressed

This project directly contributes to several United Nations Sustainable Development Goals (SDGs), particularly:

*   **SDG 2: Zero Hunger** - By providing more accurate yield forecasts, the model empowers farmers to increase productivity, improve planning, and strengthen resilience to climate change, thus contributing to food security.
*   **SDG 8: Decent Work and Economic Growth** - Optimized resource allocation based on predictions can lead to greater efficiency and economic stability for farmers.
*   **SDG 13: Climate Action** - The model's integration of weather data supports adaptation to climate change impacts and can aid in early warning systems for adverse conditions.

## Machine Learning Approach

The core of this project utilizes a **supervised machine learning** approach.

*   **Algorithm:** We employ an **XGBoost Regressor** model, a powerful gradient boosting algorithm known for its accuracy and efficiency in handling structured data.
*   **Feature Engineering:** We incorporated domain-specific features such as **Growing Degree Days (GDD)** and simulated **Soil Moisture** alongside raw weather and historical data.
*   **Training and Optimization:** The model is trained on historical data split into training and testing sets. **Grid Search with cross-validation** was used to systematically find the optimal hyperparameters for the XGBoost model, aiming to improve generalization performance.

## Data

*   **Source:** The model is trained on historical agricultural data provided via a CSV file upload and integrates real-time weather data from the OpenWeatherMap API.
*   **Key Features Used:** The model considers features such as Year, average rainfall, pesticides use, average temperature, Growing Degree Days (GDD), Soil Moisture, and crop type.
*   **Upload Requirement:** To run the notebook, users need to upload their specific historical agricultural dataset in CSV format. The notebook includes steps to read and process this uploaded data.

## Model Performance

The model's performance was evaluated using key regression metrics on both the training and testing datasets.

*   **Training Performance:**
    *   MAE: [1469.90] kg/ha
    *   RMSE: [2317.64] kg/ha
    *   R²: [ 1.00]

*   **Testing Performance:**
    *   MAE: [ 4807.38] kg/ha
    *   RMSE: [10762.22] kg/ha
    *   R²: [ 0.98]

**Discussion:**
While the training performance shows a perfect fit (R² = 1.00), the testing performance (R² = 0.98) indicates some degree of overfitting. However, the testing R² of 0.98 is still very high, demonstrating that the model explains a large proportion of the variance in crop yield on unseen data, and the absolute errors (MAE/RMSE) provide a practical measure of prediction accuracy.

## Ethical Considerations

We recognize the importance of ethical considerations in developing and deploying AI models in agriculture. Key points considered include:

*   **Data Bias:** Acknowledging that the model's performance is dependent on the representativeness of the training data and could exhibit bias if the data is not diverse.
*   **Access and Equity:** Thinking about how to ensure equitable access to the technology and its benefits for all farmers, regardless of scale or location.
*   **Transparency:** Exploring ways to provide some level of interpretability for the model's predictions to build trust and understanding among users.
*   **Data Privacy:** Recognizing the need for secure handling of agricultural data.

## Setup and Installation

This project is designed to be run in Google Colab.

1.  **Open the Notebook:** Upload or open the `.ipynb` file in Google Colab.
2.  **Install Dependencies:** Run the first code cell to install the required libraries:
3.  **Upload Data:** Run the data upload cell and select your historical agricultural data CSV file.
4.  **Create `.env` File:** Run the cell to create the `.env` file and replace the placeholder API key with your actual OpenWeatherMap API key.
5.  **Obtain OpenWeatherMap API Key:** You will need a free API key from OpenWeatherMap ([https://openweathermap.org/api](https://openweathermap.openweathermap.org/api)).
6.  **Run Remaining Cells:** Execute the remaining code cells sequentially to load data, define functions, train the model, evaluate performance, and save the model.

## Usage

1.  Follow the [Setup and Installation](#setup-and-installation) steps to train the model.
2.  Once the model is trained and saved (`crop_model_from_csv.pkl`), you can use the `predict_with_realtime_weather` function (defined in the notebook) to get yield predictions for a specific location and crop type using current weather data.
    *   Ensure the `model` variable is loaded or the saved model is loaded into a variable named `model`.
    *   Ensure you have the `api_key` and the `trained_feature_columns` available (these are derived during the training process).
    *   Example usage (within a Python script or another notebook after loading the model and required variables):
    *

     # Load API key
    load_dotenv()
    api_key = os.getenv("OPENWEATHER_API_KEY")

    # Load the trained model
    try:
        with open('crop_model_from_csv.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        print("Model loaded successfully.")
        except FileNotFoundError:
        print("Error: Model file not found. Please run the training cells first.")
        loaded_model = None

    # --- Need to get the list of feature columns used during training ---
    # This list is created in the main training cell.
    # You would typically save this list alongside the model or ensure
    # the prediction function can reconstruct it based on crop types
    # and the standard features.
    # For demonstration, assuming you have 'X_train' available or
    # a saved list of column names. If you don't have X_train, you'd
    # need to load a small part of the training data or the feature columns list.
    # Let's assume you have a way to get trained_feature_columns
    # Example (if you have X_train):
    # trained_feature_columns = X_train.columns.tolist()
    # Or, you might need to manually list them or save them in the training cell.
    # As a placeholder:
    trained_feature_columns = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'gdd', 'soil_moisture'] # Add dummy crop columns here based on your data
    # Example with dummy columns (replace with your actual crop names):
    # trained_feature_columns = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'gdd', 'soil_moisture', 'crop_Maize', 'crop_Potatoes', 'crop_Rice', 'crop_Wheat']
    # --- End getting feature columns ---


    if loaded_model is not None and trained_feature_columns is not None:
        location_to_predict = "Pretoria" # Replace with desired location
        crop_to_predict = "Maize"     # Replace with desired crop type (must be one the model was trained on)

        # Call the prediction function with the loaded model and feature columns
        predicted_yield = predict_with_realtime_weather(
            location_to_predict,
            crop_to_predict,
            api_key,
            loaded_model,         # Pass the loaded model
            trained_feature_columns # Pass the feature columns
        )

        if isinstance(predicted_yield, (int, float)):
             print(f"\nPredicted yield for {crop_to_predict} in {location_to_predict}: {predicted_yield:.2f} kg/ha")
        else:
             print(f"\nPrediction failed: {predicted_yield}")
    else:
         print("\nCould not make a prediction. Model or feature columns not available.").

    ## Future Enhancements

*   Investigate and implement techniques to reduce overfitting (e.g., more aggressive hyperparameter tuning, regularization, early stopping based on a validation set).
*   Incorporate additional data sources such as actual soil data, satellite imagery (e.g., NDVI), or more granular weather data.
*   Explore the use of other machine learning algorithms and deep learning approaches.
*   Develop a more robust method for handling missing data (beyond simply dropping rows).
*   Create a user-friendly interface or API for easier interaction with the model.
*   Validate the model on more diverse geographical regions and crop types.
*   Improve the simulation or integration of real-time soil moisture data.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure the code is well-commented.
4.  Test your changes thoroughly.
5.  Submit a pull request with a clear description of your changes.


## Contact

If you have any questions or feedback, please feel free to contact [KlaasMatlou [+27720724950]] or [tshupianematlou@gmail.com].
