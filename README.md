# Predictive-Analytics-for-Bone-Health-Project

## Project Scope 🔎

The aim of this project is to develop 3 deep learning models and compare which one performs the best to classify whether or not the bone health is healthy based on the features available in the dataset.

## Project Objectives 🎯

The project objectives is to:

1. Get the dataset
2. Perform EDA (Exploratory Data Analysis)
3. Preprocess the data
4. Implement 3 Deep Learning Models on our preprocessed data
5. Evaluate each of the models and get the F1-Score, Precision, Recall, ROC Curve, AUC, etc.
6. Fine-tune and improve the models performance (if neccessary).

## Project Resources 📚

1. https://www.google.com/url?q=https%3A%2F%2Fsyslog.ravelin.com%2Fclassification-with-tabnet-deep-dive-49a0dcc8f7e8
2. https://www.google.com/url?q=https%3A%2F%2Fresearch.google%2Fblog%2Fwide-amp-deep-learning-better-together-with-tensorflow%2F
3. https://www.google.com/url?q=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fapi%2Fsklearn.metrics.html%23classification-metrics
4. How to build ANN using tensorflow - https://www.turing.com/kb/building-neural-network-in-tensorflow
5. Tensorflow ANN documentation - https://www.tensorflow.org/tutorials/customization/custom_layers
6. TF Model Documentation - https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
7. TensorFlow - https://www.tensorflow.org/tutorials/quickstart/beginner
8. Wide and Deep Network Model - https://thegrigorian.medium.com/exploring-wide-and-deep-networks-b6af7f0a5d3a
9. Tabnet Model - https://www.kaggle.com/code/marcusgawronsky/tabnet-in-tensorflow-2-0
10. Data Preprocessing - https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/
11. Nomalization - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#standardscaler
12. Splitting data - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#train-test-split
13. Matplotlib pie charts - https://matplotliborg/stable/gallery/pie_and_polar_charts/index.html
14. Figures, subplots - https://matplotlib.org/stable/gallery/subplots_axes_and_figures/index.html
15. Scatter plots - https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter
16. Correlation Heatmap - https://www.google.com/url?q=https%3A%2F%2Fwww.geeksforgeeks.org%2Fseaborn-heatmap-a-comprehensive-guide%2F

Charts of the 3 deep learning models: TabNet, ANN, Wide and Deep Model showing the accuracy and loss curves for train and test

![image](https://github.com/user-attachments/assets/30a0ad06-a23a-4c0e-9c20-2eca4a13f9b6)

## Comclusion and final thoughts

Overall Observations

Model Performance:
TabNet outperforms both ANN and Deep models in terms of both accuracy and loss. ANN shows respectable performance but could potentially be optimized further. Deep model requires significant improvement, as its accuracy is consistently lower.

Conclusion

The graphs accurately represent the model performance over epochs, highlighting the superior effectiveness of TabNet for this dataset and task. Further analysis, such as examination of learning rates or model architectures, may provide insights for improving the ANN and Deep models.

In this study, we explored multiple deep learning models to classify solar panel faults, including a standard Artificial Neural Network (ANN), a deeper neural network, and the TabNet model. After extensive training and evaluation, TabNet emerged as the best-performing model, achieving the highest accuracy and lowest loss.

Key Takeaways: TabNet’s Superiority:

It achieved near 100% accuracy on both training and test data, outperforming traditional deep learning models. The model’s unique feature selection mechanism contributed to improved generalization. ANN & Deep Learning Models:

Both models showed consistent improvement over epochs but lagged slightly behind TabNet. Fine-tuning hyperparameters such as dropout, batch normalization, and learning rate could further enhance their performance. Model Comparison:

The accuracy and loss plots demonstrated TabNet’s stability, while other models showed signs of gradual improvement over time.

Areas for Improvement & Future Work:

🔹 Avoid Overfitting in TabNet: The near-perfect performance raises concerns of overfitting. Adding regularization techniques like dropout or increasing data diversity could help.

🔹 Feature Importance Analysis: Extracting feature importance from TabNet can provide deeper insights into which parameters influence predictions the most.

🔹 Hyperparameter Optimization: Exploring techniques like Bayesian Optimization or Grid Search could fine-tune architectures for even better performance.

🔹 Deploying the Model: The next step could involve deploying the best-performing model using FastAPI, TensorFlow Serving, or integrating it into a web-based interface for real-time predictions.

Overall, this project demonstrated the potential of Health Status of bones, highlighting how deep learning models—especially TabNet—can be leveraged to enhance predictive accuracy and automate diagnostics. Further refinement and deployment of this system could significantly improve the healthcare sector, maintenance, and efficiency in the real world. 🌞🔍🚀
