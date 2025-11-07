<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Clinical Q&A Engine</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {
      --bg-main: #06080d;
      --bg-panel: rgba(14, 18, 26, 0.96);
      --accent: #3a7bd5;
      --accent-soft: #00d2ff;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --text-main: #f5f5f5;
      --text-muted: #9ca3af;
      --radius-xl: 22px;
      --shadow-soft: 0 18px 40px rgba(0, 0, 0, 0.55);
      --transition-fast: 0.18s ease;
      --font-main: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
        -system-ui, sans-serif;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--font-main);
      background:
        radial-gradient(circle at top left, #151821, #06080d 55%),
        radial-gradient(circle at top right, #10131a, transparent 65%);
      color: var(--text-main);
      min-height: 100vh;
    }

    .page {
      padding: 32px 40px;
    }

    .qa-wrapper {
      max-width: 1280px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.3fr);
      gap: 28px;
      align-items: stretch;
    }

    /* LEFT SIDE */
    .left-panel {
      background: var(--bg-panel);
      border-radius: var(--radius-xl);
      padding: 26px;
      border: 1px solid var(--border-subtle);
      box-shadow: var(--shadow-soft);
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .engine-label {
      font-size: 0.75rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent-soft);
    }

    .title {
      font-size: 2.4rem;
      font-weight: 800;
      color: #f9fafb;
    }

    .subtitle {
      font-size: 0.9rem;
      color: var(--text-muted);
      margin-top: 4px;
    }

    .field-label {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--text-muted);
      margin-top: 10px;
    }

    textarea {
      width: 100%;
      min-height: 120px;
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--border-subtle);
      background: rgba(5, 7, 12, 0.98);
      color: var(--text-main);
      font-size: 0.95rem;
      resize: vertical;
      outline: none;
    }

    textarea:focus {
      border-color: rgba(58, 123, 213, 0.9);
      box-shadow: 0 0 16px rgba(37, 99, 235, 0.26);
    }

    .button-row {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .btn {
      border: none;
      padding: 10px 18px;
      border-radius: 999px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all var(--transition-fast);
    }

    .btn-main {
      background: linear-gradient(135deg, var(--accent), var(--accent-soft));
      color: #020817;
      font-weight: 600;
    }

    .btn-main:hover {
      transform: scale(1.05);
    }

    .btn-ghost {
      background: transparent;
      border: 1px solid rgba(148, 163, 253, 0.25);
      color: #ccc;
    }

    .btn-ghost:hover {
      border-color: rgba(148, 163, 253, 0.5);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 8px rgba(34, 197, 94, 0.9);
    }

    /* RIGHT SIDE */
    .right-panel {
      background: var(--bg-panel);
      border-radius: var(--radius-xl);
      padding: 24px;
      border: 1px solid var(--border-subtle);
      box-shadow: var(--shadow-soft);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .answer-box {
      flex: 1;
      padding: 14px;
      border-radius: 16px;
      background: rgba(4, 7, 12, 0.98);
      border: 1px solid rgba(148, 163, 253, 0.14);
      font-size: 0.9rem;
      line-height: 1.6;
      color: #e5e7eb;
      overflow-y: auto;
      max-height: 500px;
      white-space: pre-wrap;
    }

    .answer-placeholder {
      color: #6b7280;
    }

    @media (max-width: 960px) {
      .qa-wrapper {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <main class="qa-wrapper">
      <!-- LEFT: INPUT -->
      <section class="left-panel">
        <div class="engine-label">Clinical Q&A Engine</div>
        <h1 class="title">Name your clinical question</h1>
        <p class="subtitle">Enter a focused question and get a structured, high-yield summary.</p>

        <div class="field-label">Clinical question</div>
        <textarea id="clinicalQuestion" placeholder="e.g. Management of new-onset atrial fibrillation in ED"></textarea>

        <div class="button-row">
          <button id="generateBtn" class="btn btn-main" type="button">⚡ Generate answer</button>
          <button id="copyBtn" class="btn btn-ghost" type="button">⧉ Copy</button>
          <button id="exportBtn" class="btn btn-ghost" type="button">⤓ Export</button>
          <div class="status-dot"></div>
        </div>
      </section>

      <!-- RIGHT: OUTPUT -->
      <section class="right-panel">
        <div id="answerBox" class="answer-box">
          <div class="answer-placeholder">
            Your structured answer will appear here (Overview, Assessment, Management, etc.).
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const questionInput = document.getElementById("clinicalQuestion");
    const answerBox = document.getElementById("answerBox");
    const generateBtn = document.getElementById("generateBtn");
    const copyBtn = document.getElementById("copyBtn");
    const exportBtn = document.getElementById("exportBtn");

    async function generateAnswer() {
      const question = questionInput.value.trim();
      if (!question) {
        answerBox.innerHTML = '<div style="color:#f97316;">Please enter a question first.</div>';
        return;
      }
      answerBox.innerHTML = "Generating...";
      try {
        const res = await fetch("/api/clinical-qa", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });
        const data = await res.json();
        if (data.answer) {
          answerBox.textContent = data.answer;
        } else {
          answerBox.textContent = data.error || "No answer returned.";
        }
      } catch (err) {
        answerBox.textContent = "Error connecting to backend.";
      }
    }

    generateBtn.addEventListener("click", generateAnswer);
    questionInput.addEventListener("keydown", e => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generateAnswer();
    });

    copyBtn.addEventListener("click", async () => {
      const text = answerBox.textContent.trim();
      if (!text) return;
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "⧉ Copy"), 1000);
    });

    exportBtn.addEventListener("click", () => {
      const text = answerBox.textContent.trim();
      if (!text) return;
      const blob = new Blob([text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "clinical-answer.txt";
      a.click();
      URL.revokeObjectURL(url);
    });
  </script>
</body>
</html>
