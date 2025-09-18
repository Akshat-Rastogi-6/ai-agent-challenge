# ai-agent-challenge
Coding agent that generates custom parsers for bank statement PDFs and evaluates them automatically.

![Overview](images/overview.png)


## Quickstart (5 steps)

1. # Clone the specific branch
git clone --branch Final_Chnages https://github.com/Akshat-Rastogi-6/ai-agent-challenge.git

# Change directory to the cloned repo
cd ai-agent-challenge

2. # Create a virtual environment using Python 3.11
 py -m venv .venv

# Activate the virtual environment
source .venv/Scripts/activate

# Install requirements
pip install -r requirements.txt


3) Configure API keys
- Create a `.env` file in the project root with:
  - `GOOGLE_API_KEY=your_google_api_key`
  - (optional) `LANGCHAIN_API_KEY=your_langchain_key`

4) Add a sample statement PDF
- Create a folder for your bank and place a PDF:
  - Example: `data\sbi\sbi.pdf`
- Optional: if you already have a reference CSV, place it as `data\sbi\result.csv`. If not, the agent will auto-create a baseline CSV from the PDF.

5) Run the agent
- New bank (auto-baseline + built-in evaluation): `python agent.py --target sbi`
- ICICI reference (runs pytest if available): `python agent.py --target icici`
- Outputs:
  - Generated parser: `custom_parsers\<bank>_parser.py`
  - Expected CSV (auto-created if missing): `data\<bank>\result.csv`
  - Logs show evaluation status; re-run with more refinement attempts if needed:
    - `python agent.py --target sbi --max-attempts 5`

![Agent Flow](images/agent_flow.png)
