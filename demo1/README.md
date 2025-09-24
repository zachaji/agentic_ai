
Make sure to install uv using pip install uv
Create API keys for google gemini and pinecone and add to .env file

uv sync 

streamlit run .\demo1.py 

----------

To run from vs code, enter CTRL+SHIFT+P , select interpreter from this project
and then select environment from the same folder (drop down appears for the project)
Then re-open terminate and run streamlit command
Browser opens with localhost and then upload files
----------

For creating this project from scratch run:
uv init  

uv add streamlit pandas python-dotenv pinecone==7.0.0 sentence-transformers phidata pyreadline3 google-generativeai PyPDF2>=3.0.0

Copy the demo1.py 

and then run
streamlit run .\demo1.py 
