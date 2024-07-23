## Detecting Parkinson’s Disease Using Python and Machine Learning

### Project Description

This project aims to build a machine learning model to accurately detect Parkinson’s Disease in individuals using an XGBoost classifier. Parkinson’s Disease is a progressive neurodegenerative disorder that affects movement, often causing tremors and stiffness. With over million individuals affected annually in many countries including India, early and accurate detection is crucial.

### Objective

To develop a model using the XGBoost algorithm that can identify the presence of Parkinson’s Disease with high accuracy.

### Tools and Libraries

- **Python**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **XGBoost**

### Dataset

We use the UCI ML Parkinsons dataset, which contains 195 records and 24 features relevant to the diagnosis of Parkinson’s Disease.

### Methodology

1. **Data Loading and Preparation**:
   - Import necessary libraries.
   - Load the dataset and extract features and labels.
   - Normalize features using MinMaxScaler.
   
2. **Data Splitting**:
   - Split the data into training and testing sets with an 80-20 ratio.

3. **Model Building**:
   - Initialize and train the XGBClassifier on the training set.

4. **Evaluation**:
   - Predict the test set outcomes and calculate the model’s accuracy.

### Steps to Run the Project

1. **Install required libraries**:
   ```sh
   pip install numpy pandas scikit-learn xgboost
   ```

2. **Run Jupyter Lab**:
   ```sh
   jupyter lab
   ```

3. **Execute the Python script** to load data, preprocess, build, train, and evaluate the model.

### Results

The XGBClassifier model achieves an accuracy of 94.87% in detecting Parkinson’s Disease, showcasing the effectiveness of the model given the relatively small dataset.

### Conclusion

This project demonstrates the potential of using machine learning, specifically the XGBoost algorithm, in medical diagnosis. The high accuracy achieved highlights the importance and potential impact of such models in healthcare.

