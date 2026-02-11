import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from .feature_extractor import FeatureExtractor  # Import our custom feature extractor

class ImageClassifier:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
            'svm': SVC(kernel='rbf', probability=True, class_weight='balanced'),
            'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
        }
        self.current_model = 'random_forest'
        self.features = []
        self.labels = []
        self.class_names = ['Original', 'Forged']
        self.extractor = FeatureExtractor(block_size=32)  # Using our feature extractor
        self.accuracy = 0
        self.confusion_mat = None
        
    def extract_features(self, image_path, label):
        """Extract features using our advanced feature extractor"""
        try:
            features = self.extractor.extract(image_path)
            self.features.append(features)
            self.labels.append(label)
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def train_model(self, model_type='random_forest', test_size=0.2):
        """Enhanced training with cross-validation and model selection"""
        if len(self.features) == 0:
            raise ValueError("No features extracted. Add images first.")
            
        X = np.array(self.features)
        y = np.array(self.labels)
        
        # Verify class balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        # Shuffle and split with stratification
        X, y = shuffle(X, y, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)
        
        # Train selected model
        self.current_model = model_type
        model = self.models[model_type]
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        
        # Evaluation
        y_pred = model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.confusion_mat = confusion_matrix(y_test, y_pred)
        
        # Save model
        joblib.dump(model, f'copy_move_{model_type}.joblib')
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return self.accuracy, self.confusion_mat
    
    def predict_image(self, image_path):
        """Predict with confidence and feature debugging"""
        model_path = f'copy_move_{self.current_model}.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {self.current_model} not trained. Please train first.")
            
        model = joblib.load(model_path)
        
        # Extract features
        features = self.extractor.extract(image_path)
        
        # Predict
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
        confidence = max(proba)
        
        # Debug output
        print(f"\nPrediction features (top 5 most important):")
        if hasattr(model, 'feature_importances_'):
            top_indices = np.argsort(model.feature_importances_)[-5:][::-1]
            for idx in top_indices:
                print(f"Feature {idx}: {features[idx]:.2f}")
        
        return self.class_names[prediction], confidence
    
    def plot_confusion_matrix(self, cm=None, save_path=None):
        """Enhanced confusion matrix visualization"""
        cm = cm or self.confusion_mat
        if cm is None:
            raise ValueError("No confusion matrix available. Train model first.")
            
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar=False)
        plt.title(f'Confusion Matrix\nAccuracy: {self.accuracy:.2%}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def evaluate_on_folder(self, folder_path, expected_label):
        """Batch evaluation on a folder of images"""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        model_path = f'copy_move_{self.current_model}.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not trained. Please train first.")
            
        model = joblib.load(model_path)
        correct = 0
        total = 0
        
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    features = self.extractor.extract(os.path.join(folder_path, file))
                    prediction = model.predict([features])[0]
                    if prediction == expected_label:
                        correct += 1
                    total += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\nEvaluation on {folder_path}:")
        print(f"Correct: {correct}/{total} ({accuracy:.2%})")
        return accuracy

# Example usage
if __name__ == "__main__":
    classifier = ImageClassifier()
    
    # Load dataset (replace with your paths)
    print("Loading dataset...")
    for original_img in os.listdir("dataset/original"):
        classifier.extract_features(f"dataset/original/{original_img}", 0)
    
    for forged_img in os.listdir("dataset/forged"):
        classifier.extract_features(f"dataset/forged/{forged_img}", 1)
    
    # Train and evaluate
    print("\nTraining model...")
    accuracy, cm = classifier.train_model(model_type='random_forest')
    classifier.plot_confusion_matrix()
    
    # Test prediction
    print("\nTesting prediction...")
    pred, conf = classifier.predict_image("dataset/forged/example_forged.jpg")
    print(f"Prediction: {pred} (Confidence: {conf:.2%})")