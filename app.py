from flask import Flask, request, jsonify, render_template
import rag_1  # Import your RAG system module

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.json.get('message')
    session_id = request.json.get('session_id', 'abc123')
    if user_input.lower() in ['end', 'bye']:
        response = "Thank you! Hope I answered your queries. Bye!"
        reset_session(session_id)
        return jsonify({'response': response, 'follow_up': ''})
    else:
        response = rag_1.process_rag_system(user_input, session_id)
        follow_up = "Do you want to continue or end the conversation? (Type 'END' or 'BYE' to end)"
        return jsonify({'response': response, 'follow_up': follow_up})

def reset_session(session_id):
    # Reset the session logic here
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5002)  # Change the port number if needed
