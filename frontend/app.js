document.addEventListener('DOMContentLoaded', () => {
    // === UI Elements ===
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    
    const statsBody = document.getElementById('stats-body');
    const rebuildBtn = document.getElementById('rebuild-btn');
    const rebuildStatus = document.getElementById('rebuild-status');

    const uploadForm = document.getElementById("upload-form");
    const uploadStatus = document.getElementById("upload-status");
    const uploadSubmit = document.getElementById("upload-submit");

    // === Theme Management ===
    const themeToggleBtn = document.getElementById("theme-toggle");
    const currentTheme = localStorage.getItem("anyrag_theme") || "light";
    
    if (currentTheme === "dark") {
        document.body.classList.add("dark-mode");
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            document.body.classList.toggle("dark-mode");
            if (document.body.classList.contains("dark-mode")) {
                localStorage.setItem("anyrag_theme", "dark");
            } else {
                localStorage.setItem("anyrag_theme", "light");
            }
        });
    }

    // === Session Management ===
    let sessionId = localStorage.getItem("anyrag_session_id");
    if (!sessionId) {
        sessionId = "session_" + Math.random().toString(36).substring(2, 9);
        localStorage.setItem("anyrag_session_id", sessionId);
    }
    
    const sessionDisplay = document.getElementById("current-session-id");
    if (sessionDisplay) sessionDisplay.textContent = sessionId;

    const newSessionBtn = document.getElementById("new-session-btn");
    if (newSessionBtn) {
        newSessionBtn.addEventListener("click", () => {
            sessionId = "session_" + Math.random().toString(36).substring(2, 9);
            localStorage.setItem("anyrag_session_id", sessionId);
            sessionDisplay.textContent = sessionId;
            chatHistory.innerHTML = '';
            addMessage("Started a new fresh session! Upload some data on the Manage panel to begin.", "ai");
            loadStats();
        });
    }

    // === Chat Functions ===
    const scrollToBottom = () => {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };
    
    // Hide hero section on first message
    const hideHero = () => {
        const hero = document.getElementById('hero-section');
        if (hero && !hero.classList.contains('hidden')) {
            hero.style.opacity = '0';
            setTimeout(() => { hero.classList.add('hidden'); }, 300);
        }
    };

    const addMessage = (content, sender = 'user', isHtml = false) => {
        hideHero();
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);

        const avatarDiv = document.createElement('div');
        avatarDiv.classList.add('avatar');
        avatarDiv.innerHTML = sender === 'user' ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>' : 'A';

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
        let answerText = "";
        if (typeof data.generation === "string") {
            answerText = data.generation;
        } else if (data.generation && data.generation.answer) {
            answerText = data.generation.answer;
        } else {
            answerText = "Could not generate an answer.";
        }

        answerText = answerText.replace(/\[\s*(\d+)\s*\]/g, (match, num) => {
            return `<span class="source-link">[${num}]</span>`;
        });

        let html = `<p>${answerText.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;

        if (data.sources && data.sources.length > 0) {
            html += `<div class="sources-box">
                <div class="sources-title">Sources Retrieved:</div>`;

            data.sources.forEach((src, idx) => {
                html += `<div class="source-item">
                    <span class="source-link">
                        [${idx+1}] ${(src.source_file||"").toUpperCase()} ${src.chunk_id} - ${src.section_heading}
                    </span>
                </div>`;
            });
            html += `</div>`;
        }

        if (data.generation && data.generation.confidence_metrics) {
            const metrics = data.generation.confidence_metrics;
            html += `<div class="confidence-box" style="margin-top:10px; font-size: 0.85em; color: var(--text-secondary);">
                <strong>Confidence Metrics:</strong>
                Retrieval: ${(metrics.retrieval_confidence * 100).toFixed(1)}% |
                Citation: ${(metrics.citation_coverage * 100).toFixed(1)}% |
                Completeness: ${(metrics.completeness * 100).toFixed(1)}%
            </div>`;
        }

        return html;
    };

    if (chatForm) {
        // Handle Enter key for textarea
        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });

        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;

            addMessage(text, 'user');
            userInput.value = '';
            userInput.disabled = true;
            addTypingIndicator();

            try {
                const methodSelect = document.getElementById('retrieval-method');
                const retrievalMethod = methodSelect ? methodSelect.value : "hybrid_rrf";

                const response = await fetch('/v1/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: text, session_id: sessionId, retrieval_method: retrievalMethod })
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
    }

    // === Manage Functions ===
    const loadStats = async () => {
        if (!statsBody) return;
        try {
            const response = await fetch('/v1/documents?session_id=' + encodeURIComponent(sessionId));
            const data = await response.json();
            
            statsBody.innerHTML = '';
            if (data.length === 0) {
                statsBody.innerHTML = '<tr><td colspan="2" style="text-align:center;">No documents found.</td></tr>';
                return;
            }

            data.forEach(stat => {
                const tr = document.createElement('tr');
                const tdId = document.createElement('td');
                tdId.textContent = stat.source_file;
                const tdCount = document.createElement('td');
                tdCount.textContent = stat.count;
                tr.appendChild(tdId);
                tr.appendChild(tdCount);
                statsBody.appendChild(tr);
            });
        } catch (err) {
            console.error(err);
            statsBody.innerHTML = '<tr><td colspan="2" style="text-align:center;color:#ff7b72;">Failed to load statistics.</td></tr>';
        }
    };

    loadStats();

    if (rebuildBtn) {
        rebuildBtn.addEventListener('click', async () => {
            rebuildBtn.disabled = true;
            rebuildBtn.textContent = 'Rebuilding...';
            rebuildStatus.classList.add('hidden');
            rebuildStatus.className = 'status-msg hidden';

            try {
                const response = await fetch('/v1/build_index', { 
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                });
                if (!response.ok) throw new Error('Failed to rebuild indices');
                
                rebuildStatus.textContent = 'Success! Indices are up to date.';
                rebuildStatus.className = 'status-msg success';
            } catch (err) {
                console.error(err);
                rebuildStatus.textContent = 'Error: Could not rebuild indices.';
                rebuildStatus.className = 'status-msg error';
            } finally {
                rebuildBtn.disabled = false;
                rebuildBtn.textContent = 'Rebuild Vector Indices';
            }
        });
    }


    if (uploadForm) {
        uploadForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById("file-upload");
            if (!fileInput.files.length) return;

            const formData = new FormData();
            formData.append("session_id", sessionId);
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append("files", fileInput.files[i]);
            }

            uploadSubmit.disabled = true;
            uploadSubmit.textContent = "Uploading...";
            if(uploadStatus) {
                uploadStatus.classList.remove("hidden");
                uploadStatus.className = "status-msg";
                uploadStatus.textContent = "Uploading and incrementally indexing... Please wait.";
            }
            
            try {
                const res = await fetch("/v1/upload", {
                    method: "POST",
                    body: formData
                });
                
                const data = await res.json();
                if(uploadStatus) {
                    if (data.status === "success") {
                        uploadStatus.className = "status-msg success";
                        uploadStatus.textContent = data.message;
                        fileInput.value = "";
                        loadStats();
                    } else {
                        uploadStatus.className = "status-msg error";
                        uploadStatus.textContent = data.message;
                    }
                }
            } catch (err) {
                if(uploadStatus) {
                    uploadStatus.className = "status-msg error";
                    uploadStatus.textContent = "Error: " + err.message;
                }
            } finally {
                uploadSubmit.disabled = false;
                uploadSubmit.textContent = "Upload and Ingest";
            }
        });
    }
});
