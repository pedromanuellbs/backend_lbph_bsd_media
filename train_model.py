# train_model.py

# Import the correct function name: 'train_and_evaluate_full_dataset'
from face_data import train_and_evaluate_full_dataset

if __name__ == "__main__":
    # Call the correct function
    metrics = train_and_evaluate_full_dataset()
    print("\n================ FINAL METRICS ================")
    print(f"Cross-Validation Metrics: {metrics}")
    print("===========================================")