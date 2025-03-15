const api = "http://127.0.0.1:5000/";
async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files.length) {
    alert("Vui lòng chọn một tệp để tải lên.");
    return;
  }
  let formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    let response = await fetch(api + "upload", {
      method: "POST",
      body: formData,
    });
    let data = await response.json();
    if (data.message === "File processed successfully") {
      alert("Tải lên thành công!");
      fetchFiles();
    } else {
      alert("Lỗi khi tải lên!");
    }
  } catch (error) {
    console.error("Lỗi khi tải file:", error);
  }
}

async function fetchFiles() {
  try {
    let response = await fetch(api + "files");
    let files = await response.json();
    let tableBody = document.getElementById("fileTableBody");
    tableBody.innerHTML = "";
    files.forEach((file) => {
      let row = `<tr>
                        <td>${file.file_name}</td>
                        <td>${new Date(file.uploaded_at).toLocaleString()}</td>
                    </tr>`;
      tableBody.innerHTML += row;
    });
  } catch (error) {
    console.error("Lỗi khi lấy danh sách file:", error);
  }
}

window.onload = fetchFiles;
