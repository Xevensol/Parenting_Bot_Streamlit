# Parenting Bot Application

## Overview

The Parenting Bot Application is a simple yet effective chatbot designed to assist users with parenting queries. This bot utilizes advanced AI technologies to provide responses and is powered by Professor Dr. Javed Iqbal's expertise.

## Features

- **User-Friendly Interface**: An intuitive design that allows users to easily input their queries.
- **AI-Powered Responses**: Utilizes OpenAI's embeddings for generating relevant answers to parenting questions.
- **Environment Configuration**: Easily configurable with a `.env` file to set your OpenAI API key.

## Requirements

- Python 3.10.15
- Streamlit

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
   pip install -r requirements.txt

4. Create a .env file in the root directory and add your OpenAI API key:
   OPENAI_API_KEY=your_openai_api_key_here


To run the application, use the following command:
streamlit run app.py



