

Make sure to install uv using pip install uv

uv init  

uv add streamlit pandas python-dotenv pinecone==7.0.0 sentence-transformers phidata pyreadline3 google-generativeai PyPDF2>=3.0.0

uv sync 

uv run .\demo1.py  

streamlit run .\demo1.py 