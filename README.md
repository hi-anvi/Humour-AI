# humour.ai 🎭

A Flask web app powered by the open-source **Mistral-7B-Instruct** model.
No API key needed — the AI runs entirely on your own machine.

---

## Project structure

```
humour-ai/
├── app.py                  ← Flask server + routes
├── ai_model.py             ← Loads & runs the open-source AI model
├── requirements.txt        ← Python dependencies
├── README.md
└── templates/
    └── index.html          ← Frontend UI (served by Flask)
```

---

## Setup & run

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

---

## Model details

| Setting        | Value                                  |
|----------------|----------------------------------------|
| Default model  | `mistralai/Mistral-7B-Instruct-v0.2`  |
| Download size  | ~14 GB (downloaded once automatically) |
| RAM needed     | ~16 GB (GPU recommended)               |

### Want a lighter model?
Open `ai_model.py` and change line 17:
```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```
This needs only ~4 GB RAM and runs fine on CPU — jokes will be slightly less sharp.

---

## How it works

1. User types a situation in the browser → clicks send
2. Browser sends a `POST /joke` request to Flask with `{ situation, mode }`
3. `app.py` calls `generate_joke()` in `ai_model.py`
4. `ai_model.py` builds a prompt and runs it through the local Mistral model
5. The joke is returned as JSON → displayed in the chat UI

---

## GPU acceleration
If you have an NVIDIA GPU, the model will automatically use it (via `device_map="auto"`).
Install the CUDA version of PyTorch for best performance:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
