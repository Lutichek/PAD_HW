const fileInput = document.querySelector("#file");
const dropZone = document.querySelector("#drop-zone");
const preview = document.querySelector("#preview");
const hint = document.querySelector("#upload-hint");
const fileName = document.querySelector("#file-name");
const predictButton = document.querySelector("#predict");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const scores = document.querySelector("#scores");
const model = document.querySelector("#model");
const examples = document.querySelector("#examples");
const percent = new Intl.NumberFormat("ru-RU", { style: "percent", maximumFractionDigits: 2 });

let selectedFile;
let previewUrl;
let activeRequest;

function showError(message) {
  status.hidden = true;
  error.textContent = message;
  error.hidden = false;
}

function showFile(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.hidden = true;
  hint.hidden = false;
  preview.src = previewUrl;
  fileName.textContent = file.name;
}

preview.addEventListener("load", () => {
  preview.classList.toggle("pixelated", preview.naturalWidth === 32 && preview.naturalHeight === 32);
  preview.hidden = false;
  hint.hidden = true;
});

function showResult(result, classes) {
  const values = result.probabilities;
  if (!Array.isArray(values) || values.length !== classes.length ||
      values.some((value) => !Number.isFinite(value) || value < 0 || value > 1) ||
      !Number.isInteger(result.class_id) || classes[result.class_id] !== result.class_name ||
      typeof result.model_name !== "string" || typeof result.model_version !== "string") {
    throw new Error("Получен неожиданный ответ модели.");
  }
  const top = values.map((value, index) => ({ value, name: classes[index] }))
    .sort((a, b) => b.value - a.value).slice(0, 5);
  scores.replaceChildren(...top.map(({ name, value }) => {
    const item = document.createElement("li");
    const row = document.createElement("div");
    row.className = "score-value";
    const label = document.createElement("span");
    label.textContent = name;
    const text = document.createElement("span");
    text.textContent = percent.format(value);
    row.append(label, text);
    const bar = document.createElement("meter");
    bar.value = value;
    bar.setAttribute("aria-label", name);
    item.append(row, bar);
    return item;
  }));
  model.textContent = "Модель: " + result.model_name + " · версия " + result.model_version;
  status.hidden = true;
  scores.hidden = false;
  model.hidden = false;
}

const dataset = fetch("/static/dataset.json", { signal: AbortSignal.timeout(5000) }).then((response) => {
  if (!response.ok) throw new Error("Не удалось загрузить список классов.");
  return response.json();
});

async function run(loadFile) {
  activeRequest?.abort();
  const request = new AbortController();
  activeRequest = request;
  const signal = AbortSignal.any([request.signal, AbortSignal.timeout(12000)]);
  predictButton.disabled = true;
  scores.hidden = true;
  model.hidden = true;
  error.hidden = true;
  status.hidden = false;
  status.textContent = "Распознаём изображение…";

  try {
    const file = await loadFile(signal);
    if (request.signal.aborted) return;
    selectedFile = file;
    showFile(file);
    const form = new FormData();
    form.append("file", file, file.name);
    const response = await fetch("/predict", { method: "POST", body: form, signal });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(typeof result.detail === "string" ? result.detail : "Не удалось распознать изображение.");
    }
    const { classes } = await dataset;
    if (!request.signal.aborted) showResult(result, classes);
  } catch (failure) {
    if (request.signal.aborted) return;
    if (signal.aborted) {
      showError("Сервис не ответил вовремя. Попробуйте ещё раз.");
    } else if (failure instanceof TypeError || failure instanceof SyntaxError) {
      showError("Не удалось получить ответ сервиса. Попробуйте ещё раз.");
    } else {
      showError(failure.message);
    }
  } finally {
    if (!request.signal.aborted) predictButton.disabled = !selectedFile;
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (file) run(() => file);
});
predictButton.addEventListener("click", () => run(() => selectedFile));
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", (event) => {
  if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove("dragging");
});
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const file = event.dataTransfer.files[0];
  if (file) run(() => file);
});

dataset.then(({ classes, examples: samples }) => {
  document.querySelector("#classes").textContent = classes.join(" · ");
  for (const sample of samples) {
    const button = document.createElement("button");
    button.className = "example";
    button.type = "button";
    const img = document.createElement("img");
    img.src = "/static/examples/" + sample.file;
    img.alt = "";
    const name = classes[sample.class_id];
    button.append(img, name);
    button.addEventListener("click", () => run(async (signal) => {
      const response = await fetch(img.src, { signal });
      if (!response.ok) throw new Error("Не удалось загрузить пример.");
      return new File([await response.blob()], sample.file, { type: "image/png" });
    }));
    examples.append(button);
  }
}).catch((failure) => showError(failure.message));
