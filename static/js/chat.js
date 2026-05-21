(function () {
  const form = document.getElementById("chat-form");
  const messagesEl = document.getElementById("messages");
  const sendBtn = document.getElementById("send-btn");
  const attachmentLabel = document.getElementById("attachment-label");
  const fileInputs = ["image", "audio", "video"].map((id) =>
    document.getElementById(id)
  );

  const history = [];

  function appendMessage(role, text, extraHtml) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    if (extraHtml) {
      div.innerHTML = "";
      const p = document.createElement("div");
      p.textContent = text;
      div.appendChild(p);
      div.insertAdjacentHTML("beforeend", extraHtml);
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
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

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = document.getElementById("message").value.trim();
    if (!message) return;

    const systemPrompt = document.getElementById("system_prompt").value.trim();
    const maxTokens = document.getElementById("max_tokens").value;

    appendMessage("user", message);
    history.push({ role: "user", content: message });

    const formData = new FormData();
    formData.append("message", message);
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
      appendMessage("assistant", data.text || "", extra);
      history.push({ role: "assistant", content: data.text || "" });

      form.reset();
      updateAttachmentLabel();
    } catch (err) {
      appendMessage("system", `Error: ${err.message}`);
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
    }
  });
})();
