import os
import zipfile
import gdown

MODEL_DIR = "film_model/saved_model"
ZIP_PATH = "film_model.zip"
DRIVE_ID = "1HfnQ8mZWIbxJlyxUrWSrcvEfqn-dqBxQ"

def download_and_extract():
    if os.path.exists(MODEL_DIR):
        print("[INFO] Model is al aanwezig. Geen download nodig.")
        return

    if not os.path.exists(ZIP_PATH):
        print("[INFO] Model zip niet gevonden. Downloaden vanaf Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)
    
    print("[INFO] Uitpakken van model zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("film_model")
    
    print("[INFO] Model succesvol gedownload en uitgepakt.")

if __name__ == "__main__":
    download_and_extract()