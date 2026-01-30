📧 Spam Mail Detector
📌 Project Description

This project detects whether an SMS message is Spam or Ham (Not Spam) using Machine Learning and NLP techniques.

🎯 Objective

To build a classifier that automatically identifies spam messages based on text content.

📂 Dataset

SMS Spam Collection Dataset (UCI Repository)

Contains 5,574 SMS messages labeled as spam or ham

🛠️ Technologies Used

Python

Pandas

NLTK

Scikit-learn

⚙️ Methodology

Load SMS dataset

Clean text (lowercase, remove punctuation & stopwords)

Convert text to numbers using TF-IDF

Train Naive Bayes classifier

Evaluate using accuracy and F1-score

Test with new messages

📊 Result

Achieved ~97% accuracy

Successfully classifies spam and ham messages

▶️ How to Run
python spam_day1.py

✅ Conclusion

The model effectively detects spam messages and can be used for SMS or email filtering applications.
