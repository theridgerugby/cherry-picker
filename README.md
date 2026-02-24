# Cherry Picker

Cherry Picker is a web tool for research. It finds papers on arXiv and uses AI to summarize them.

## Features
- It finds the newest papers on arXiv.
- It makes your search topic better for Finding papers.
- It reads the papers and gets the main info.
- It shows what topics are new and popular.
- It gives papers a score to show if they are good.
- It lists the skills and tools mentioned in the papers.
- It writes a simple report for you.

## How to install
You need Python 3.8 or newer.

1. Download the code to your computer.
2. Open your terminal in the code folder.
3. Install the needed parts:
   ```bash
   pip install -r requirements.txt
   ```
4. Put your Google API Key in a file named `.env`:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## How to use
Run this command to start the app:
```bash
streamlit run app.py
```
The app will open in your web browser.

## The files in this project
- `app.py`: This is the main file for the web app.
- `input_validator.py`: It checks if your search topic is okay.
- `query_generator.py`: It creates a good search for arXiv.
- `paper_fetcher.py`: It gets papers from arXiv.
- `paper_extractor.py`: It reads the text from the papers.
- `trend_analyzer.py`: It finds new trends in the research.
- `credibility_scorer.py`: It checks if the papers are high quality.
- `skills_analyzer.py`: It finds technical skills in the papers.
- `report_generator.py`: It creates the final report file.
- `agent.py`: It uses AI to analyze the papers.
- `config.py`: It has the settings for the app.

## Using Docker
You can also use Docker to run the app:
1. Build the app:
   ```bash
   docker build -t cherry-picker .
   ```
2. Run the app:
   ```bash
   docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key cherry-picker
   ```
3. Go to `http://localhost:8501` on your computer.
