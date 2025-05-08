import download
import preprocess
import understand
import visualize

def execute_pipeline():
    print("Step 1: Downloading Dataset")
    download.download_fake_news_dataset()

    print("Step 2: Understanding Dataset")
    understand.analyze_dataset()  # Replace with the actual function name from understand.py

    print("Step 3: Visualizing Dataset")
    # Call functions from visualize.py to visualize the dataset

    print("Step 4: Preprocessing Dataset")
    preprocess.preprocess_fake_news_data()

if __name__ == "__main__":
    execute_pipeline()
