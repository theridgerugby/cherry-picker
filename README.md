# 🍒 Cherry Picker

**Cherry Picker** is an Apple-inspired Streamlit web application designed for **arXiv Research Intelligence**. It empowers researchers to input a topic and automatically receive a comprehensive, AI-driven markdown report summarizing the latest relevant papers directly from arXiv.

Unlike tools that rely on stale training data, Cherry Picker fetches the most recent papers in real-time, extracts their core content, and performs deep analysis to uncover trends, assess credibility, and identify essential skills within the domain.

## ✨ Features

- **Real-time arXiv Fetching:** Always up-to-date. Papers are queried directly from arXiv based on your research topic.
- **Complex Query Generation:** Transforms simple user inputs into highly optimized arXiv search queries.
- **Intelligent Content Extraction:** Parses and extracts the core textual content from the downloaded papers.
- **Trend Analysis:** Identifies emerging trends and declining approaches within the fetched corpus of papers.
- **Credibility Scoring:** Evaluates research papers based on their methodology, content, and author/institution metrics.
- **Skills Analysis:** Automatically detects tools, techniques, and skills required or introduced in the domain.
- **Polished Markdown Reports:** Compiles all findings into a structured, easily readable, and beautifully formatted Markdown report.
- **Premium UI:** Features an Apple-inspired, sleek, and responsive user interface built with Streamlit.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A Google API Key (for Gemini models used in analysis)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cherry-picker
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory (or use Streamlit secrets) and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Running the App

Start the Streamlit application by running:

```bash
streamlit run app.py
```

The application will launch in your default web browser.

## 🏗️ Architecture overview

Cherry Picker is composed of several specialized modules:

- `app.py`: The main Streamlit application and UI orchestrator.
- `input_validator.py` & `query_generator.py`: Validate user input and generate optimized arXiv queries.
- `paper_fetcher.py` & `paper_extractor.py`: Interface with arXiv to download and extract data from papers.
- `trend_analyzer.py`: Analyzes the historical and emerging trends in the research domain.
- `credibility_scorer.py`: Scores the credibility and impact potential of the papers.
- `skills_analyzer.py`: Extracts specific tools and technical skills mentioned in the research.
- `report_generator.py`: Assembles the final markdown report from the analytical modules.
- `agent.py`: Orchestrates the Language Model (e.g., Gemini) invocations for summaries and deep-dive analyses.
- `config.py`: Contains configuration constants for the application.

## 🐳 Docker Deployment

You can also run Cherry Picker using Docker:

```bash
docker build -t cherry-picker .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_google_api_key_here cherry-picker
```

Navigate to `http://localhost:8501` to access the app.
