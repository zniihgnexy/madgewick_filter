import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score



results_dict = {
    'Class': ['0', '1', '2', '3', '4', '5', '6', 'macro avg', 'weighted avg', 'Accuracy'],
    'Precision': [0.98, 0.83, 0.00, 0.89, 0.88, 0.60, 0.00, 0.60, 0.74, 0.8201634877384196],
    'Recall': [1.00, 0.94, 0.00, 0.90, 0.82, 0.98, 0.00, 0.66, 0.82, None],  # Accuracy doesn't need Recall
    'F1-score': [0.99, 0.88, 0.00, 0.89, 0.85, 0.75, 0.00, 0.62, 0.77, None],  # Accuracy doesn't need F1-score
    'Support': [96, 53, 28, 61, 44, 65, 20, 367, 367, None],  # Accuracy doesn't need Support
}



results_df = pd.DataFrame(results_dict)


results_df.to_excel('model_evaluation_results.xlsx', index=False)

print("Excel 文件已保存。")
