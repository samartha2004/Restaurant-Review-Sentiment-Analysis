import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text).lower()
    words = text.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text

def main():
    try:
        if not os.path.exists('sentiment_train.csv'):
            print("Error: sentiment_train.csv not found in the current directory.")
            print("Please ensure the dataset file is present.")
            return
        
        print("Loading dataset...")
        df = pd.read_csv('sentiment_train.csv')
        
        print("Preprocessing text...")
        df['Processed_Text'] = df['Sentence'].apply(preprocess_text)
        
        print("Creating Bag of Words model...")
        cv = CountVectorizer(max_features=1500, ngram_range=(1, 2))
        X = cv.fit_transform(df['Processed_Text']).toarray()
        y = df.iloc[:, 1].values
        
        print("Saving CountVectorizer...")
        joblib.dump(cv, "cv.pkl")
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        
        print("\n" + "="*50)
        print("Training Naive Bayes Model...")
        print("="*50)
        classifier_nb = MultinomialNB(alpha=0.2)
        classifier_nb.fit(X_train, y_train)
        y_pred_nb = classifier_nb.predict(X_test)
        accuracy_nb = accuracy_score(y_test, y_pred_nb)
        f1_nb = f1_score(y_test, y_pred_nb)
        print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
        print(f"Naive Bayes F1 Score: {f1_nb:.4f}")
        joblib.dump(classifier_nb, "model_nb.pkl")
        print("Naive Bayes model saved as model_nb.pkl")
        
        print("\n" + "="*50)
        print("Training Random Forest Model...")
        print("="*50)
        classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)
        classifier_rf.fit(X_train, y_train)
        y_pred_rf = classifier_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)
        print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
        print(f"Random Forest F1 Score: {f1_rf:.4f}")
        joblib.dump(classifier_rf, "model_rf.pkl")
        print("Random Forest model saved as model_rf.pkl")
        
        print("\n" + "="*50)
        print("Training SVM Model...")
        print("="*50)
        classifier_svm = SVC(kernel='linear', random_state=0)
        classifier_svm.fit(X_train, y_train)
        y_pred_svm = classifier_svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        f1_svm = f1_score(y_test, y_pred_svm)
        print(f"SVM Accuracy: {accuracy_svm:.4f}")
        print(f"SVM F1 Score: {f1_svm:.4f}")
        joblib.dump(classifier_svm, "model_svm.pkl")
        print("SVM model saved as model_svm.pkl")
        
        print("\n" + "="*50)
        print("Training Logistic Regression Model...")
        print("="*50)
        classifier_lr = LogisticRegression(random_state=0, max_iter=1000)
        classifier_lr.fit(X_train, y_train)
        y_pred_lr = classifier_lr.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
        print(f"Logistic Regression F1 Score: {f1_lr:.4f}")
        
        print("\n" + "="*50)
        print("Hyperparameter Tuning with GridSearchCV...")
        print("="*50)
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']}
        grid_search = GridSearchCV(LogisticRegression(random_state=0, max_iter=1000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        best_classifier_lr = LogisticRegression(random_state=0, max_iter=1000, **best_params)
        best_classifier_lr.fit(X_train, y_train)
        y_pred_best_lr = best_classifier_lr.predict(X_test)
        accuracy_best_lr = accuracy_score(y_test, y_pred_best_lr)
        f1_best_lr = f1_score(y_test, y_pred_best_lr)
        print(f"Tuned Logistic Regression Accuracy: {accuracy_best_lr:.4f}")
        print(f"Tuned Logistic Regression F1 Score: {f1_best_lr:.4f}")
        print(f"Best Parameters: {best_params}")
        joblib.dump(best_classifier_lr, "model_lr_tuned.pkl")
        print("Tuned Logistic Regression model saved as model_lr_tuned.pkl")
        
        print("\n" + "="*50)
        print("Model Comparison Summary")
        print("="*50)
        results = {
            'Model': ['Naive Bayes', 'Random Forest', 'SVM', 'Logistic Regression', 'Tuned LR'],
            'Accuracy': [accuracy_nb, accuracy_rf, accuracy_svm, accuracy_lr, accuracy_best_lr],
            'F1 Score': [f1_nb, f1_rf, f1_svm, f1_lr, f1_best_lr]
        }
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        best_model_idx = results_df['Accuracy'].idxmax()
        print(f"\nBest Model: {results_df.loc[best_model_idx, 'Model']}")
        print(f"Best Accuracy: {results_df.loc[best_model_idx, 'Accuracy']:.4f}")
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print("\nGenerated Files:")
        print("  - cv.pkl (CountVectorizer)")
        print("  - model_nb.pkl (Naive Bayes)")
        print("  - model_rf.pkl (Random Forest)")
        print("  - model_svm.pkl (SVM)")
        print("  - model_lr_tuned.pkl (Tuned Logistic Regression)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'sentiment_train.csv' exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
