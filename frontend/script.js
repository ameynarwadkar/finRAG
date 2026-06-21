document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');

    // Auto-scroll to bottom of chat
    const scrollToBottom = () => {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };

    const addMessage = (content, sender = 'user', isHtml = false) => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);

        const avatarDiv = document.createElement('div');
        avatarDiv.classList.add('avatar');
        avatarDiv.textContent = sender === 'user' ? 'U' : 'AI';

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        if (isHtml) {
            contentDiv.innerHTML = content;
        } else {
            contentDiv.textContent = content;
        }

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        chatHistory.appendChild(messageDiv);
        scrollToBottom();
        return messageDiv;
    };

    const addTypingIndicator = () => {
        const contentHtml = `
            <div class="typing-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        `;
        const msg = addMessage(contentHtml, 'ai', true);
        msg.id = 'typing-indicator-msg';
    };

    const removeTypingIndicator = () => {
        const indicator = document.getElementById('typing-indicator-msg');
        if (indicator) {
            indicator.remove();
        }
    };

    const formatAIResponse = (data) => {
        const CELEX_MAP = {
            "mifid2": "32014L0065",
            "psd2": "32015L2366",
            "gdpr": "32016R0679",
            "dora": "32022R2554"
        };

        // Parse inline citations e.g. [[PSD2 Art. 64]]
        let answerText = data.generation.answer;
        answerText = answerText.replace(/\[\[([a-zA-Z0-9]+)\s+Art\.\s+([0-9a-zA-Z\.]+)\]\]/g, (match, docId, artNum) => {
            const docKey = docId.toLowerCase();
            const celex = CELEX_MAP[docKey] || "";
            let textFragment = "";

            // Try to find the exact source to get the title for the text fragment
            if (data.sources) {
                const src = data.sources.find(s => s.doc_id.toLowerCase() === docKey && s.article_number == artNum);
                if (src && src.article_title) {
                    textFragment = `#:~:text=${encodeURIComponent(src.article_title)}`;
                }
            }

            const url = celex ? `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:${celex}${textFragment}` : "#";
            return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="source-link">[${docId.toUpperCase()} Art. ${artNum}]</a>`;
        });

        // Format the answer text nicely (supporting basic markdown-like paragraphs)
        let html = `<p>${answerText.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;

        // Add retrieved sources if available
        if (data.sources && data.sources.length > 0) {

            html += `<div class="sources-box">
                <div class="sources-title">Sources Retrieved:</div>`;

            data.sources.forEach(src => {
                const docName = src.doc_id ? src.doc_id.toLowerCase() : "";
                const celex = CELEX_MAP[docName] || "";

                // By searching for the exact contiguous string "Article X Title", we prevent matching partial ranges
                const encodedTitle = src.article_title ? encodeURIComponent(src.article_title) : "";
                const textFragment = src.article_number ? `#:~:text=${encodedTitle}` : "";
                const url = celex ? `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:${celex}${textFragment}` : "#";

                html += `<div class="source-item">
                    <a href="${url}" target="_blank" rel="noopener noreferrer" class="source-link">
                        [${src.doc_id.toUpperCase()} Art. ${src.article_number}] ${src.article_title}
                    </a>
                </div>`;
            });
            html += `</div>`;
        }

        return html;
    };

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // Add user message to UI
        addMessage(text, 'user');
        userInput.value = '';
        userInput.disabled = true;

        // Add loading indicator
        addTypingIndicator();

        try {
            // Adjust the URL if the backend is hosted elsewhere. 
            // Relative path works because we serve frontend from the same FastAPI app.
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: text })
            });

            removeTypingIndicator();

            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }

            const data = await response.json();
            addMessage(formatAIResponse(data), 'ai', true);

        } catch (error) {
            removeTypingIndicator();
            addMessage(`<p style="color: #ff7b72;">Error: Could not connect to the assistant. Please try again later.</p>`, 'ai', true);
            console.error(error);
        } finally {
            userInput.disabled = false;
            userInput.focus();
        }
    });
});
