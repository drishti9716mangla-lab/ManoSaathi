# ManoSaathi updated files

## Files
- `index.html`: updated UI, bilingual site toggle, wellbeing quick prompts and mobile bottom navigation.
- `app.py`: Flask API. Calls the OpenAI-compatible Chat Completions API, includes recent conversation history, and optionally retrieves matching snippets from a Kaggle CSV/JSON file.
- `requirements.txt`: Python packages.
- `.env.example`: template only; copy it to `.env` and insert your own API key.

## Important dataset note
The uploaded files did not include the Kaggle dataset itself. The backend therefore loads it if you place it at `data/kaggle_dataset.csv` (or set `KAGGLE_DATASET_PATH` in `.env`). It also supports a JSON dataset. If your dataset has a different filename or column structure, set the path accordingly. The retrieval is basic lexical matching, not model training.

## Run
1. Put `index.html`, `app.py`, and `requirements.txt` in the same project folder.
2. Create a `data` folder and put your Kaggle CSV/JSON file there if you want it used.
3. Create `.env` in the project folder with your API key. Never put the API key in `index.html` or share it.
4. In VS Code terminal:
   `py -m pip install -r requirements.txt`
   `py app.py`
5. Open `http://127.0.0.1:5000` in your browser.
6. Test `http://127.0.0.1:5000/api/health` to check whether the API key and dataset were detected.

## Notes
- If the API key is missing or the API request fails, the backend falls back to the intent JSON dataset if present, then a supportive generic response.
- The symptom screener is educational and is not a diagnosis.
- The crisis keyword check is a basic safeguard, not a guaranteed crisis detector. Do not deploy this as a clinical or emergency service without expert review and safety testing.
- The Hindi UI translation covers the visible site labels and common static content. User-generated chatbot replies are requested in the selected language; dataset fallback content may not always be perfectly translated.
