# University Chatbot - Machine Learning Project

## Steps to setup:
### 1. Clone the Repository
#### Clone this repository to your local machine:
```bash
git clone https://github.com/vansh1225/university-chatbot-ml
cd university-chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
```

Activate the virtual environment:





- On macOS/Linux:

        
        source venv/bin/activate
        


- On Windows:

        venv\Scripts\activate

### 3. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 4. Download NLTK and SpaCy Resources
```
python3 -m nltk.downloader punkt wordnet
python3 -m spacy download en_core_web_sm
```

### 5.Running the Chatbot
#### To run the chatbot, execute the main Flask application:
``` 
python3 app.py
```