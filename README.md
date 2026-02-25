# Cherry Picker

Cherry Picker is an AI-powered Streamlit web application designed to help researchers seamlessly find, summarize, and analyze the newest academic papers from arXiv. 

**Note: This tool is only designed to help you find papers you are interested in; you still have to read the papers yourself.**

## Features
- **Smart arXiv Search**: Finds the most recent and relevant papers based on your search topic.
- **AI Summaries**: Uses Google Gemini to read the text of the papers and extract the main contributions, methodology, and key findings.
- **Trend & Skill Analysis**: Identifies popular trends in your research domain and generates a learning roadmap of technical skills mentioned in the papers.
- **Quality Scoring**: Gives papers a credibility score so you can focus on the most impactful research.
- **Markdown Reports**: Automatically writes a clean, structured summary report of all findings.
- **Vector Search**: Saves extracted paper data to a local ChromaDB database so you can query insights.

## How to Install
You will need Python 3.8 or newer.

1. Download or clone the code to your computer.
2. Open your terminal in the code folder.
3. Install the needed parts:
   ```bash
   pip install -r requirements.txt
   ```
4. Put your Google API Key in a file named `.env`:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## How to Use
Run this command to start the web app:
```bash
streamlit run app.py
```
The application interface will open automatically in your web browser. Type in your topic of interest and let the agent gather your research!

## Project Files
- `app.py`: The main file for the Streamlit web interface.
- `agent.py`: LangChain ReAct agent that orchestrates the overall AI workflow.
- `paper_fetcher.py` and `query_generator.py`: Connects to arXiv to search and download papers.
- `paper_extractor.py`: Extracts structured text and insights from paper PDFs.
- `trend_analyzer.py`: Spots new shifts and directions in the academic landscape.
- `credibility_scorer.py`: Evaluates and scores paper quality.
- `skills_analyzer.py`: Discovers technical skills and tools needed to understand the papers.
- `report_generator.py`: Creates the final markdown report file.
- `config.py`: Controls the application settings and AI model selections.

## Using Docker
You can also run the app using Docker:
1. Build the app image:
   ```bash
   docker build -t cherry-picker .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 -e GOOGLE_API_KEY=your_api_key cherry-picker
   ```
3. Visit `http://localhost:8501` on your computer.
