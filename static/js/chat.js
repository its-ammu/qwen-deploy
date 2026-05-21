(function () {
  const STORAGE_KEY = "qwen_chat_history_v1";
  const form = document.getElementById("chat-form");
  const messagesEl = document.getElementById("messages");
  const sendBtn = document.getElementById("send-btn");
  const clearBtn = document.getElementById("clear-chat");
  const attachmentLabel = document.getElementById("attachment-label");
  const fileInputs = ["image", "audio", "video"].map((id) =>
    document.getElementById(id)
  );

  /** @type {{role: string, content: string}[]} */
  let history = loadHistory();

  function loadHistory() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) return parsed;
      }
    } catch (_) {
      /* ignore */
    }
    return [];
  }

  function saveHistory() {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  }

  function renderHistory() {
    messagesEl.innerHTML = "";
    history.forEach((msg) => {
      appendMessage(msg.role, msg.content, "", false);
    });
  }

  function appendMessage(role, text, extraHtml, persist) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    const p = document.createElement("div");
    p.textContent = text;
    div.appendChild(p);
    if (extraHtml) {
      div.insertAdjacentHTML("beforeend", extraHtml);
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    if (persist !== false && role !== "system") {
      /* history updated by caller */
    }
  }

  function updateAttachmentLabel() {
    const names = fileInputs
      .filter((input) => input.files && input.files[0])
      .map((input) => `${input.id}: ${input.files[0].name}`);
    attachmentLabel.textContent = names.length ? names.join(" · ") : "";
  }

  fileInputs.forEach((input) => {
    input.addEventListener("change", updateAttachmentLabel);
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      history = [];
      saveHistory();
      renderHistory();
    });
  }

  renderHistory();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = document.getElementById("message").value.trim();
    const hasFile = fileInputs.some((input) => input.files && input.files[0]);
    if (!message && !hasFile) return;

    const systemPrompt = document.getElementById("system_prompt").value.trim();
    const maxTokens = document.getElementById("max_tokens").value;

    let displayText = message;
    if (hasFile) {
      const tags = fileInputs
        .filter((i) => i.files && i.files[0])
        .map((i) => `[${i.id}: ${i.files[0].name}]`);
      displayText = [message, ...tags].filter(Boolean).join(" ");
    }

    appendMessage("user", displayText || "(media)", "", false);
    history.push({ role: "user", content: displayText || message || "(audio/media)" });

    const formData = new FormData();
    formData.append("message", message || "Describe the attached media.");
    formData.append("history", JSON.stringify(history.slice(0, -1)));
    if (systemPrompt) formData.append("system_prompt", systemPrompt);
    if (maxTokens) formData.append("max_tokens", maxTokens);

    fileInputs.forEach((input) => {
      if (input.files && input.files[0]) {
        formData.append(input.id, input.files[0]);
      }
    });

    sendBtn.disabled = true;
    sendBtn.textContent = "Generating...";

    try {
      const response = await fetch(window.QWEN_UI.chatUrl, {
        method: "POST",
        body: formData,
        credentials: "same-origin",
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error?.message || response.statusText);
      }

      let extra = "";
      if (data.audio_url) {
        extra = `<audio controls src="${data.audio_url}" style="margin-top:8px;width:100%"></audio>`;
      }
      const reply = data.text || "";
      appendMessage("assistant", reply, extra, false);
      history.push({ role: "assistant", content: reply });
      saveHistory();

      form.reset();
      updateAttachmentLabel();
    } catch (err) {
      history.pop();
      saveHistory();
      appendMessage("system", `Error: ${err.message}`, "", false);
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
    }
  });
})();
