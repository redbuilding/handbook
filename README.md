# 📚🤖✍️ Handbook Generator

## Description
The Handbook Generator is a Python application that automatically creates comprehensive handbooks on user-specified topics. It utilizes multiple Language Models (LLMs), web search capabilities, and user feedback to generate high-quality, structured content.

## Features
- Automated handbook generation on any given topic
- Web search integration for up-to-date context
- Multi-threaded chapter generation using multiple LLMs
- Interactive feedback system for content refinement
- Markdown output for easy readability and further editing

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/handbook-generator.git
cd handbook-generator
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root and add the following API keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
SERPER_API_KEY=your_serper_api_key
```

## Usage

1. Run the main script:
```
python main.py
```

2. Enter the topic for your handbook when prompted.

3. The application will generate the handbook content and provide options for feedback and refinement.

4. Once satisfied, choose to finish and save the handbook. The output will be saved as a Markdown file in the current directory.

## Models Used

This application uses the following language models:
- gpt-4o-mini (via OpenAI)
- claude-3-5-sonnet-20240620 (via Anthropic)
- google/gemini-flash-1.5 (via OpenRouter)

The estimated cost to generate one handbook is approximately $0.14 USD. The process takes a couple minutes to complete.

## Note on API Usage and Costs

Please be aware that this application makes use of paid API services. Ensure you have sufficient credits or funds in your accounts for OpenAI, Anthropic, OpenRouter, and Serper.

## Logging

The application logs its operations in `handbook_generator.log` for debugging and monitoring purposes.

## Learn more

This application was built for a hackathon organized by echohive, who developed unified.py. Learn more at:  https://echohive.live/
