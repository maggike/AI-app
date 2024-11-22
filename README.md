**Installation**
**Follow these steps to set up the project locally.**

**1. Clone the Repository**

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

**2. Set Up the Backend**
**a. Create a Virtual Environment**
It's recommended to use a virtual environment to manage dependencies.
python3 -m venv venv

Activate the virtual environment:
On macOS and Linux:
source venv/bin/activate

**b. Install Backend Dependencies**
pip install -r requirements.txt

**3. Set Up the Frontend**
Navigate to the frontend directory.
cd frontend

Install frontend dependencies:
npm install
Configuration
1. Obtain OpenAI API Key
Sign up or log in to your OpenAI account.
Navigate to the API Keys section.
Generate a new API key.
2. Create Environment Variables
a. Backend Configuration
Create a .env file in the root directory of the backend (where app.py is located).
touch .env
Add the following to the .env file:

OPENAI_API_KEY=your-openai-api-key-here
FLASK_APP=app.py
FLASK_ENV=development
PORT=5000

Replace your-openai-api-key-here with your actual OpenAI API key.


**Running the Application**
**1. Start the Backend Server**
Ensure you're in the 'backend' directory and the virtual environment is activated.
python3 app.py
The backend server should now be running at http://localhost:5000.

**2. Start the Frontend Server**
Open a new terminal window/tab, navigate to the frontend directory, and start the frontend.
cd frontend
npm start
This should open the frontend application in your default browser at http://localhost:3000.
