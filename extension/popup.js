// The work runs in the background, so closing this popup does not stop a send.
const button = document.getElementById("send");
const status = document.getElementById("status");

function show(value) {
  if (!value) return;
  status.textContent = value.text;
  status.className = value.state;
  button.disabled = value.state === "busy" && Date.now() - value.at < 120000;
}

chrome.storage.session.get("status").then(({ status: value }) => show(value));
chrome.storage.session.onChanged.addListener((changes) => show(changes.status?.newValue));

button.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const autoScroll = document.getElementById("scroll").checked;
  chrome.runtime.sendMessage({ type: "grab", tabId: tab.id, autoScroll });
});
