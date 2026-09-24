document.addEventListener('DOMContentLoaded', () => {
    initClock();
    initNavigation();
    loadAllData();
    initVoiceRecorder();
    initCandidateFilters();
    initModal();
    initTaxonomy();

    document.getElementById('trigger-btn').addEventListener('click', triggerBrief);
    document.getElementById('copy-brief-btn').addEventListener('click', copyBriefToClipboard);
});

let currentBriefData = null;
let currentCandidates = [];
let taxonomyList = [];

// Clock Init
function initClock() {
    const clock = document.getElementById('clock-display');
    const update = () => {
        const now = new Date();
        clock.innerText = now.toLocaleTimeString([], { hour12: false });
    };
    update();
    setInterval(update, 1000);
}

// Navigation Init
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const pageTitle = document.getElementById('page-title');

    const titles = {
        overview: "Executive Overview",
        candidates: "Research Candidates & Intelligence",
        arc: "Portfolio Project Arc Tracker",
        history: "Historical Brief Archive",
        settings: "Taxonomy & Processing Settings"
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            navItems.forEach(n => n.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.add('active');
            pageTitle.innerText = titles[targetTab] || "Signal Dashboard";
        });
    });

    document.querySelectorAll('.switch-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-target');
            const navBtn = document.querySelector(`.nav-item[data-tab="${target}"]`);
            if (navBtn) navBtn.click();
        });
    });
}

// Load All Initial Data
async function loadAllData() {
    await Promise.all([
        loadLatestBrief(),
        loadArcStatus(),
        loadHistory(),
        loadTaxonomy()
    ]);
}

// 1. Brief Loader
async function loadLatestBrief() {
    try {
        const res = await fetch('/brief/latest');
        if (res.ok) {
            currentBriefData = await res.json();
            renderBrief(currentBriefData);
        } else {
            document.getElementById('latest-brief-formatted').innerHTML = `
                <div class="empty-state">
                    <p style="color: var(--text-muted);">No brief generated yet. Click "Run Signal Brief" to kick off ingestion & analysis.</p>
                </div>
            `;
        }
    } catch (e) {
        showToast('Error loading latest brief', 'error');
    }
}

function renderBrief(data) {
    const container = document.getElementById('latest-brief-formatted');
    container.innerText = data.brief_text || "Empty brief text";

    // Metrics
    document.getElementById('metric-events-count').innerText = data.calendar_events ? data.calendar_events.length : 0;
    document.getElementById('metric-candidates-count').innerText = data.top_items ? data.top_items.length : 0;
    
    if (data.generated_at) {
        const dateStr = new Date(data.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        document.getElementById('metric-last-time').innerText = dateStr;
    }

    // Candidates
    if (data.top_items) {
        currentCandidates = data.top_items;
        renderCandidates(currentCandidates);
    }
}

// 2. Arc Status Loader
async function loadArcStatus() {
    try {
        const res = await fetch('/arc-status');
        if (res.ok) {
            const statuses = await res.json();
            document.getElementById('metric-projects-count').innerText = statuses.length;
            renderArcSummary(statuses);
            renderArcFull(statuses);
        }
    } catch (e) {
        showToast('Error loading arc status', 'error');
    }
}

function renderArcSummary(statuses) {
    const list = document.getElementById('arc-summary-list');
    list.innerHTML = '';
    statuses.slice(0, 5).forEach(s => {
        const item = document.createElement('div');
        item.className = 'arc-mini-card';
        item.innerHTML = `
            <div class="arc-mini-title">
                <span>${escapeHtml(s.project)}</span>
            </div>
            <div class="arc-mini-status"><strong>Status:</strong> ${escapeHtml(s.status)}</div>
            <div class="arc-mini-next"><strong>Next:</strong> ${escapeHtml(s.next_action)}</div>
        `;
        list.appendChild(item);
    });
}

function renderArcFull(statuses) {
    const grid = document.getElementById('arc-full-grid');
    grid.innerHTML = '';
    statuses.forEach(s => {
        const card = document.createElement('div');
        card.className = 'card glass arc-card';
        card.innerHTML = `
            <div class="arc-card-header">
                <span class="arc-card-title">${escapeHtml(s.project)}</span>
                <button class="delete-btn" onclick="deleteProject('${escapeHtml(s.project)}')">&times;</button>
            </div>
            <div style="font-size:0.9rem; margin-bottom:0.75rem;">
                <span style="color:var(--text-muted)">Status:</span><br>
                <strong style="color:var(--text-primary);">${escapeHtml(s.status)}</strong>
            </div>
            <div style="font-size:0.85rem; color:var(--text-secondary);">
                <span style="color:var(--text-muted)">Next Action:</span><br>
                ${escapeHtml(s.next_action)}
            </div>
        `;
        grid.appendChild(card);
    });
}

async function deleteProject(name) {
    if (!confirm(`Remove "${name}" from tracked projects?`)) return;
    try {
        const res = await fetch(`/arc-status/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (res.ok) {
            showToast(`Deleted ${name}`);
            loadArcStatus();
        }
    } catch (e) {
        showToast('Error deleting project', 'error');
    }
}

// 3. Candidates Rendering & Filtering
function initCandidateFilters() {
    const searchInput = document.getElementById('candidate-search');
    const sourceSelect = document.getElementById('source-filter');
    const scoreRange = document.getElementById('score-filter');
    const scoreVal = document.getElementById('score-filter-val');

    const applyFilters = () => {
        const q = searchInput.value.toLowerCase();
        const src = sourceSelect.value;
        const minScore = parseFloat(scoreRange.value);
        scoreVal.innerText = minScore.toFixed(1);

        const filtered = currentCandidates.filter(sc => {
            const item = sc.candidate;
            const matchesQuery = item.title.toLowerCase().includes(q) || (item.summary && item.summary.toLowerCase().includes(q));
            const matchesSource = src === 'all' || item.source === src;
            const matchesScore = sc.relevance_score >= minScore;
            return matchesQuery && matchesSource && matchesScore;
        });

        renderCandidates(filtered);
    };

    searchInput.addEventListener('input', applyFilters);
    sourceSelect.addEventListener('change', applyFilters);
    scoreRange.addEventListener('input', applyFilters);
}

function renderCandidates(list) {
    const grid = document.getElementById('candidates-grid');
    grid.innerHTML = '';

    if (!list || list.length === 0) {
        grid.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 2rem; color: var(--text-muted);">No candidates matching criteria.</div>`;
        return;
    }

    list.forEach(sc => {
        const c = sc.candidate;
        const card = document.createElement('div');
        card.className = 'card glass candidate-card';
        card.innerHTML = `
            <div>
                <div class="candidate-header">
                    <span class="badge ${c.source}">${c.source}</span>
                    <span class="score-tag">★ ${sc.relevance_score.toFixed(1)}</span>
                </div>
                <div class="candidate-title">
                    <a href="${c.url}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>
                </div>
                <div class="candidate-summary">
                    ${escapeHtml(sc.digest_summary || sc.rationale || c.summary)}
                </div>
            </div>
            <div style="font-size:0.75rem; color:var(--text-muted); display:flex; justify-content:space-between;">
                <span>ID: ${escapeHtml(c.external_id)}</span>
                <a href="${c.url}" target="_blank" style="color:var(--accent-cyan); text-decoration:none;">View Link →</a>
            </div>
        `;
        grid.appendChild(card);
    });
}

// 4. Trigger Brief Action
async function triggerBrief() {
    const btn = document.getElementById('trigger-btn');
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width:16px;height:16px;margin:0;"></span> Running Brief...`;

    try {
        const res = await fetch('/brief/auto', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        if (res.ok) {
            showToast('Signal Brief Run Complete! ✨');
            await loadLatestBrief();
            await loadHistory();
        } else {
            showToast('Brief execution failed', 'error');
        }
    } catch (e) {
        showToast('Error triggering brief', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="btn-icon">🚀</span><span class="btn-label">Run Signal Brief</span>`;
    }
}

// 5. Voice Recorder & Visualizer
function initVoiceRecorder() {
    const micBtn = document.getElementById('mic-btn');
    const recBar = document.getElementById('recording-bar');
    const stopBtn = document.getElementById('stop-rec-btn');
    const timerDisplay = document.getElementById('recording-timer');
    const canvas = document.getElementById('audio-waveform');
    const ctx = canvas.getContext('2d');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval;
    let startTime;
    let audioCtx, analyser, dataArray, animId;

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            // Audio visualizer setup
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser();
            const source = audioCtx.createMediaStreamSource(stream);
            source.connect(analyser);
            analyser.fftSize = 64;
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            const drawWaveform = () => {
                animId = requestAnimationFrame(drawWaveform);
                analyser.getByteFrequencyData(dataArray);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const barWidth = (canvas.width / dataArray.length) * 1.5;
                let x = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    const barHeight = (dataArray[i] / 255) * canvas.height;
                    ctx.fillStyle = '#f43f5e';
                    ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
                    x += barWidth;
                }
            };
            drawWaveform();

            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
                cancelAnimationFrame(animId);
                if (audioCtx) audioCtx.close();

                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioChunks = [];
                stream.getTracks().forEach(t => t.stop());

                recBar.classList.add('hidden');
                micBtn.classList.remove('recording');

                showToast('Transcribing & Extracting Arc Updates...');
                await sendVoiceNote(audioBlob);
            };

            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add('recording');
            recBar.classList.remove('hidden');

            startTime = Date.now();
            timerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
                const s = String(elapsed % 60).padStart(2, '0');
                timerDisplay.innerText = `${m}:${s}`;
            }, 1000);

        } catch (err) {
            showToast('Microphone access denied or unavailabile', 'error');
        }
    };

    const stopRecording = () => {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            clearInterval(timerInterval);
            isRecording = false;
        }
    };

    micBtn.addEventListener('click', () => {
        if (!isRecording) startRecording();
        else stopRecording();
    });

    stopBtn.addEventListener('click', stopRecording);
}

async function sendVoiceNote(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'update.webm');

    try {
        const res = await fetch('/arc-status/voice', { method: 'POST', body: formData });
        if (res.ok) {
            const data = await res.json();
            showToast(data.confirmation_text || 'Arc update processed');
            loadArcStatus();
        } else {
            showToast('Voice processing error', 'error');
        }
    } catch (e) {
        showToast('Error uploading voice note', 'error');
    }
}

// 6. Project Modal Form
function initModal() {
    const modal = document.getElementById('add-project-modal');
    const openBtn = document.getElementById('add-project-btn');
    const closeBtn = document.getElementById('close-modal-btn');
    const form = document.getElementById('project-form');

    openBtn.addEventListener('click', () => modal.classList.remove('hidden'));
    closeBtn.addEventListener('click', () => modal.classList.add('hidden'));

    form.addEventListener('submit', async e => {
        e.preventDefault();
        const project = document.getElementById('form-project-name').value;
        const status = document.getElementById('form-project-status').value;
        const next_action = document.getElementById('form-project-next').value;

        try {
            const res = await fetch('/arc-status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ project, status, next_action })
            });
            if (res.ok) {
                showToast(`Project '${project}' updated`);
                modal.classList.add('hidden');
                form.reset();
                loadArcStatus();
            }
        } catch (err) {
            showToast('Failed to save project', 'error');
        }
    });
}

// 7. Brief History
async function loadHistory() {
    try {
        const res = await fetch('/brief/history');
        if (res.ok) {
            const history = await res.json();
            renderHistoryList(history);
        }
    } catch (e) {
        console.error('Error loading history:', e);
    }
}

function renderHistoryList(history) {
    const list = document.getElementById('history-list');
    list.innerHTML = '';

    if (history.length === 0) {
        list.innerHTML = `<div style="color:var(--text-muted); font-size:0.85rem;">No history yet.</div>`;
        return;
    }

    history.forEach((brief, idx) => {
        const d = new Date(brief.generated_at);
        const item = document.createElement('div');
        item.className = `history-item ${idx === 0 ? 'active' : ''}`;
        item.innerText = `Run ${d.toLocaleDateString()} ${d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}`;
        item.addEventListener('click', () => {
            document.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            document.getElementById('history-detail-content').innerText = brief.brief_text;
        });
        list.appendChild(item);
    });

    if (history[0]) {
        document.getElementById('history-detail-content').innerText = history[0].brief_text;
    }
}

// 8. Taxonomy Settings
async function loadTaxonomy() {
    try {
        const res = await fetch('/taxonomy');
        if (res.ok) {
            const data = await res.json();
            taxonomyList = data.taxonomy || [];
            renderTaxonomyChips();
        }
    } catch (e) {
        console.error('Error loading taxonomy', e);
    }
}

function initTaxonomy() {
    const addBtn = document.getElementById('add-tag-btn');
    const input = document.getElementById('new-tag-input');
    const saveBtn = document.getElementById('save-taxonomy-btn');

    addBtn.addEventListener('click', () => {
        const val = input.value.trim();
        if (val && !taxonomyList.includes(val)) {
            taxonomyList.push(val);
            input.value = '';
            renderTaxonomyChips();
        }
    });

    saveBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/taxonomy', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ taxonomy: taxonomyList })
            });
            if (res.ok) showToast('Taxonomy settings saved!');
        } catch (e) {
            showToast('Error saving taxonomy', 'error');
        }
    });
}

function renderTaxonomyChips() {
    const container = document.getElementById('taxonomy-chips');
    container.innerHTML = '';
    taxonomyList.forEach((tag, idx) => {
        const chip = document.createElement('div');
        chip.className = 'chip';
        chip.innerHTML = `
            <span>${escapeHtml(tag)}</span>
            <span class="chip-remove" onclick="removeTag(${idx})">&times;</span>
        `;
        container.appendChild(chip);
    });
}

window.removeTag = function(idx) {
    taxonomyList.splice(idx, 1);
    renderTaxonomyChips();
};

// Utilities
function copyBriefToClipboard() {
    if (!currentBriefData || !currentBriefData.brief_text) return;
    navigator.clipboard.writeText(currentBriefData.brief_text);
    showToast('Brief copied to clipboard! 📋');
}

function showToast(msg, type = '') {
    const toast = document.getElementById('toast');
    toast.innerText = msg;
    toast.className = `toast ${type}`;
    setTimeout(() => toast.classList.add('hidden'), 4000);
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    })[m]);
}
