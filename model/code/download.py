import os
import subprocess

def download_fake_news_dataset():
    # Ensure the Kaggle directory exists
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

    # Determine the absolute path to kaggle.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kaggle_json_path = os.path.join(script_dir, "../requirements/kaggle.json")

    if os.path.exists(kaggle_json_path):
        subprocess.run(["mv", kaggle_json_path, os.path.expanduser("~/.kaggle/")])
        subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")])
    else:
        print("kaggle.json not found in the requirements folder. Please place it there.")
        return

    # Download the dataset using Kaggle API
    print("Downloading Fake and Real News Dataset...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "clmentbisaillon/fake-and-real-news-dataset", "--unzip", "-p", os.path.join(script_dir, "../dataset")])

    print("Dataset downloaded and extracted to ../dataset")

if __name__ == "__main__":
    download_fake_news_dataset()