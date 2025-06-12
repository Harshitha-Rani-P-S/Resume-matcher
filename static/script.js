// Text-based prediction
async function checkMatch() {
  const job_description = document.getElementById("jd").value;
  const resume_text = document.getElementById("resume").value;

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_description, resume_text })
  });

  const result = await response.json();
  document.getElementById("result").innerText =
    result.match
      ? `✅ Match (Confidence: ${result.confidence})`
      : `❌ Not a Match (Confidence: ${result.confidence})`;
}

// PDF-based prediction
async function checkMatchPDF() {
  const job_description = document.getElementById("jd").value;
  const fileInput = document.getElementById("resume_pdf");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please upload a PDF file.");
    return;
  }

  const formData = new FormData();
  formData.append("resume_file", file);
  formData.append("job_description", job_description);

  const response = await fetch("/predict-pdf", {
    method: "POST",
    body: formData
  });

  const result = await response.json();
  document.getElementById("result").innerText =
    result.match
      ? `✅ Match (Confidence: ${result.confidence})`
      : `❌ Not a Match (Confidence: ${result.confidence})`;
}
