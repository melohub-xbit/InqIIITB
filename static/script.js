document.addEventListener('DOMContentLoaded', function() {
    displayMessage("Welcome to InqIIITB! What do you want to know about?", 'bot-message');
});

document.getElementById('submit-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    if (message) {
        displayMessage(message, 'user-message');
        userInput.value = '';
        displayLoading();
        fetch('/send_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message, session_id: 'abc123' })
        })
        .then(response => response.json())
        .then(data => {
            removeLoading();
            displayMessage(data.response, 'bot-message');
            if (data.follow_up) {
                displayMessage(data.follow_up, 'bot-message');
            }
            if (data.response.toLowerCase().includes('thank you')) {
                setTimeout(() => {
                    resetChat();
                }, 3000);
            }
        });
    }
}

function displayMessage(message, className) {
    const chatBox = document.getElementById('chat-box');
    const messageBubble = document.createElement('div');
    messageBubble.className = `message ${className}`;
    messageBubble.textContent = message;
    chatBox.appendChild(messageBubble);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function displayLoading() {
    const chatBox = document.getElementById('chat-box');
    const loadingBubble = document.createElement('div');
    loadingBubble.className = 'message bot-message loading';
    loadingBubble.id = 'loading';
    loadingBubble.innerHTML = '<div class="loading-icon"></div>';
    chatBox.appendChild(loadingBubble);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeLoading() {
    const loadingBubble = document.getElementById('loading');
    if (loadingBubble) {
        loadingBubble.remove();
    }
}

function resetChat() {
    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML = '';
    displayMessage("Welcome to InqIIITB! What do you want to know about?", 'bot-message');
}
