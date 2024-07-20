# Maternal Health Risk Predictor

## Overview
The Maternal Health Risk Predictor is a machine learning-based system designed to predict the likelihood of health complications in pregnant women. The system uses various health parameters to provide early warnings and recommendations, ensuring timely medical intervention.

## Features
- Accurate prediction of maternal health risks
- User-friendly web interface using Gradio
- Supports real-time data input and analysis
- Visualizations of prediction results

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Gradio
- Jupyter Notebook

## Project Structure
```
maternal_health_risk_predictor/
├── data/
│ ├── raw_data.csv
├── notebooks/
│ ├── data_preprocessing.ipynb
│ └── model_training.ipynb
├── src/
│ ├── app.py
│ ├── model.py
├── .gitignore
├── README.md
├── requirements.txt
├── LICENSE
└── setup.py
```

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Vikhram-S/Maternal-Health-Risk-Predictor.git
    ```
2. Navigate to the project directory:
    ```
    cd Maternal-Health-Risk-Predictor.git
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
1. Run the Gradio application:
    ```
    app.py
    ```
2. Open the provided local URL in your browser to access the application interface.

## Usage
1. Input the required health parameters in the Gradio interface.
2. Click the "Predict" button to get the risk prediction.
3. View the results and take necessary actions based on the prediction.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the (LICENSE) file for details.

## Contact
For any inquiries or support, please contact (mailto:vikhrams15gmail.com).

