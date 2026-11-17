# SmartDoc Extractor

## Run locally
1. Install Python 3.11.
2. Open a terminal in this folder.
3. Create a virtual environment:
   `python -m venv .venv`
4. Activate it on Windows:
   `.venv\\Scripts\\activate`
5. Install dependencies:
   `pip install -r requirements.txt`
6. Start the app:
   `streamlit run app.py`
7. The app opens in a browser, usually at `http://localhost:8501`.

## Share through Streamlit Community Cloud
Push `app.py` and `requirements.txt` to a GitHub repository, create an app in Streamlit Community Cloud, and select `app.py` as the entrypoint.

## Notes
- The MAERSK header logo is embedded in `app.py`; no separate image file is required.
- TrOCR handwriting OCR can require a model download unless a local `models/trocr-small-handwritten` folder is included. If the target network blocks model downloads, keep handwriting OCR disabled or package the model separately.
