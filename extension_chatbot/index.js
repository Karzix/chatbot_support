const api = "http://103.209.34.217:5000/";
document.getElementById("submit").addEventListener("click", sendMessage);
async function sendMessage() {
    const userInput = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");
    let message = userInput.value.trim();
    if (!message) return;

    // Hiển thị tin nhắn người dùng
    let userMsg = document.createElement("div");
    userMsg.className = "message user-message";
    userMsg.textContent = message;
    chatBox.appendChild(userMsg);
    userInput.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    // Hiển thị trạng thái "Đang trả lời..."
    let botMsg = document.createElement("div");
    botMsg.className = "message bot-message";
    botMsg.innerHTML = "<i>Đang trả lời...</i>";
    chatBox.appendChild(botMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Gọi API chatbot
    try {
        let response = await fetch( api + "search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: message })
        });
        let data = await response.json();

        // Cập nhật nội dung phản hồi chatbot với HTML đúng format
        botMsg.innerHTML = data.refined_answer ? data.refined_answer : "<i>Xin lỗi, tôi không hiểu câu hỏi của bạn.</i>";
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (error) {
        console.error("Lỗi gọi API:", error);
        botMsg.innerHTML = "<i>Không thể kết nối đến chatbot.</i>";
    }
}