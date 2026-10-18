# Aseda · Group 13 Llama-Based Customer-Service Chatbot (No RAG)

MSBA 610 Advanced Text Analytics

| File | What it is |
|---|---|
| `Group13_Chatbot_Colab.ipynb` | Runs the whole project on Google Colab: data, cleaning, EDA, representation, router models, Llama fine-tuning and evaluation. |
| `app.py` | The Streamlit app. Reads everything from `artifacts/`. |
| `textprep.py` | Cleaning code shared by the notebook and the app. |
| `requirements.txt` | Packages for Streamlit Cloud. **The notebook writes the final version**, pinning scikit-learn. |
| `.streamlit/config.toml` | App colours and theme. |
| `artifacts/` | Results from the notebook. *Not included until you run it.* |

---

## Step 1 · Run the notebook in Colab (about 45–70 min)

1. Go to <https://colab.research.google.com>, then **File → Upload notebook** and choose `Group13_Chatbot_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Llama access: on <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>, accept the licence. Then go to
   **Settings → Access Tokens → Create new token**, choose type **Read**, and copy the token.
   *If the licence isn't approved yet, skip this step. The notebook uses an open copy of the same model instead.*
4. In Colab, click the **key icon** in the left bar. Choose **Add new secret**, set Name `HF_TOKEN`, paste the token as
   the Value, and switch **Notebook access** on.
5. **Runtime → Run all.** Click **Connect to Google Drive** and allow access when asked.
6. At the end, **`group13_deploy.zip`** downloads automatically. A copy is also saved in Google Drive under
   `MyDrive/group13_chatbot/`.

If Colab disconnects, run **Runtime → Run all** again. Finished steps load from Drive instead of running again.

## Step 2 · Put the project on GitHub

1. Make sure `artifacts/` and the Colab `requirements.txt` (from `group13_deploy.zip`) sit next to `app.py`.
2. Create a new repository on GitHub and upload the whole project folder (drag and drop on github.com, or push from PyCharm).

## Step 3 · Deploy on Streamlit Community Cloud

1. On <https://share.streamlit.io>, click **Create app**, then choose **Deploy a public app from GitHub**.
2. Pick the repository, branch `main`, and main file path `app.py`.
3. Optionally choose a custom app URL, then click **Deploy**.
4. If the build log shows a package install error, open **Advanced settings** and pick **Python 3.12**.

---

## What changed from the earlier version (for the report's methodology section)

| Earlier version | Now | Why |
|---|---|---|
| Dataset read from a CSV on one laptop | Loaded from Hugging Face in Colab | Anyone can reproduce it |
| Fine-tuning with Apple MLX + Ollama (Mac only) | QLoRA (4-bit, LoRA r=16) on a free Colab T4 | Runs anywhere, and is a standard method |
| 50 steps × 4 examples (200 examples seen) | 2,000 examples, 1 epoch | Enough data for the model to learn the support style |
| Live Ollama calls in the app (can't work when hosted) | Router runs live, Llama answers recorded from Colab | Works on free Streamlit hosting |
| App sent raw lower-case text to a router trained on stemmed text | Shared `textprep.py`, with cleaning inside the saved model | Live predictions match the reported accuracy |
| Stopword list removed "not", "don't" | Negations kept | "I don't want to cancel" ≠ "I want to cancel" |
| Duplicate questions could fall in both train and test | Repeated questions kept once, one fixed 80/10/10 split | No leakage |
| Router output not used by Llama | Router's intent passed to Llama as a routing note | The architecture actually works as described |
| Hallucination = `{{slots}}` in answer not in question | Invented specifics (numbers, phones, links, prices, time windows) the customer never gave | The old metric penalised correct placeholder use and missed real fabrication |
| Correctness = cosine rescaled to (sim+1)/2 | ROUGE-L, BERTScore and raw cosine similarity | Rescaling made unrelated answers score about 0.7 or more |
| Means only | Paired bootstrap 95% CI and Wilcoxon / McNemar tests | Shows whether differences are real |
| `eda.py` missing, app crashed | EDA in the notebook, charts in the app | — |
