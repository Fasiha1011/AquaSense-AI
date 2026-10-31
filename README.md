# AquaSense AI 💧
### Water Potability Prediction Using Artificial Neural Network (ANN)

## About the Project
AquaSense AI is a machine learning project that predicts whether water is potentially potable (suitable for drinking) or not potable based on its water quality measurements.

The project uses an Artificial Neural Network (ANN) to analyze water quality parameters and classify water samples into two categories.

## Objectives
- Analyze water quality measurements.
- Handle missing values in the dataset.
- Train an Artificial Neural Network model.
- Predict water potability.
- Display predictions through a Flask web application.

## Dataset
The project uses the Water Potability dataset.

It contains nine input features:

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

**Target variable:** Potability  
- `0` = Not Potable
- `1` = Potable

## Technologies Used
- Python
- Jupyter Notebook
- Pandas and NumPy
- Scikit-learn
- Artificial Neural Network (MLPClassifier)
- Flask
- HTML and CSS

## Project Structure
```text
AquaSense-AI/
├── AquaSense.ipynb
├── app.py
├── water_potability.csv
└── templates/
    └── index.html
```

## How It Works
1. Load and explore the water potability dataset.
2. Handle missing values using median imputation.
3. Scale the input features using StandardScaler.
4. Train an ANN model.
5. Evaluate the model using classification metrics.
6. Use Flask to provide a web interface for entering water quality measurements and viewing predictions.

## How to Run
1. Install the required Python libraries.
2. Open the Jupyter Notebook to explore and train the model.
3. Run the Flask application using Python.
4. Open `http://127.0.0.1:5000` in your browser.

## Note
AquaSense AI is an educational machine learning project. Its predictions are estimates and should not replace certified laboratory water-quality testing.

## Author
**Fasiha Faraz**

## Project Type
Academic Project — Artificial Intelligence 

