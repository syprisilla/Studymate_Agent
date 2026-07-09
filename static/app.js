let currentSessionId = localStorage.getItem("session_id") || "default";

async function createNewChat() {
  const res = await fetch("/api/chats", {
    method: "POST",
  });

  const chat = await res.json();

  currentSessionId = chat.session_id;
  localStorage.setItem("session_id", currentSessionId);

  document.querySelector("#messages").innerHTML = "";

  await loadChatList();
}

async function loadChatList() {
  const res = await fetch("/api/chats");
  const chats = await res.json();

  const list = document.querySelector("#chatList");
  list.innerHTML = "";

  chats.forEach((chat) => {
    const item = document.createElement("button");
    item.className = "chat-history-item";

    item.innerHTML = `
      <strong>${chat.title || "새 대화"}</strong>
      <span>${chat.pdf_name || "PDF 없음"}</span>
    `;

    item.addEventListener("click", () => {
      openChat(chat.session_id);
    });

    list.appendChild(item);
  });
}

async function openChat(sessionId) {
  currentSessionId = sessionId;
  localStorage.setItem("session_id", currentSessionId);

  const res = await fetch(`/api/chats/${sessionId}`);
  const data = await res.json();

  const messages = document.querySelector("#messages");
  messages.innerHTML = "";

  data.messages.forEach((msg) => {
    appendMessage(msg.role, msg.content);
  });
}

document.querySelector("#newChatBtn").addEventListener("click", createNewChat);

loadChatList();
