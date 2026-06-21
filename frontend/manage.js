document.addEventListener('DOMContentLoaded', () => {
    const statsBody = document.getElementById('stats-body');
    const rebuildBtn = document.getElementById('rebuild-btn');
    const rebuildStatus = document.getElementById('rebuild-status');
    const ingestForm = document.getElementById('ingest-form');
    const ingestSubmitBtn = document.getElementById('ingest-submit');
    const ingestStatus = document.getElementById('ingest-status');

    // Fetch and display stats
    const loadStats = async () => {
        try {
            const response = await fetch('/stats');
            const data = await response.json();
            
            statsBody.innerHTML = '';
            if (data.length === 0) {
                statsBody.innerHTML = '<tr><td colspan="2" style="text-align:center;">No documents found.</td></tr>';
                return;
            }

            data.forEach(stat => {
                const tr = document.createElement('tr');
                const tdId = document.createElement('td');
                tdId.textContent = stat.doc_id;
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

    // Rebuild Index
    rebuildBtn.addEventListener('click', async () => {
        rebuildBtn.disabled = true;
        rebuildBtn.textContent = 'Rebuilding... (This takes a moment)';
        rebuildStatus.classList.add('hidden');
        rebuildStatus.className = 'status-msg hidden';

        try {
            const response = await fetch('/build_index', { method: 'POST' });
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

    // Ingest new document
    ingestForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const docId = document.getElementById('ingest-doc-id').value.trim();
        const artNum = document.getElementById('ingest-art-num').value.trim();
        const title = document.getElementById('ingest-title').value.trim();
        const text = document.getElementById('ingest-text').value.trim();
        
        if (!docId || !artNum || !title || !text) return;

        ingestSubmitBtn.disabled = true;
        ingestSubmitBtn.textContent = 'Adding to Corpus...';
        ingestStatus.classList.add('hidden');

        try {
            const response = await fetch('/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    doc_id: docId,
                    article_number: artNum,
                    article_title: title,
                    text: text
                })
            });

            if (!response.ok) throw new Error('Failed to add document');

            ingestStatus.textContent = 'Success! Document added. Remember to Rebuild Indices to make it searchable.';
            ingestStatus.className = 'status-msg success';
            
            ingestForm.reset();
            loadStats(); // Refresh table
        } catch (error) {
            ingestStatus.textContent = 'Error: Could not add document.';
            ingestStatus.className = 'status-msg error';
            console.error(error);
        } finally {
            ingestSubmitBtn.disabled = false;
            ingestSubmitBtn.textContent = 'Add to Corpus';
        }
    });
});
