# Day 1 — Build the target, then break it

**Goal for today:** run a deliberately vulnerable RAG chatbot, attack it 20+ times, and record what worked.
By tonight you should be able to explain (out loud, without notes) why each successful attack worked.

---

## 1. Setup (about 30 minutes)

```bash
# 1. Unzip the project and open a terminal inside the llm-firewall folder
cd llm-firewall

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file from the template
#    Windows: copy .env.example .env      Mac/Linux: cp .env.example .env
#    Then open .env and paste your API key and a model name.
```

**Getting a free API key:** sign up at Groq (console.groq.com) or Google AI Studio (aistudio.google.com),
create an API key, and pick a model from their model list. Put all three values in `.env`.

**Run it:**

```bash
streamlit run demo/app.py
```

Your browser opens the chat. Ask "How long does a refund take?" — if you get a sensible answer, setup works.
(Terminal-only alternative: `python demo/rag_app.py`.)

**Then, on GitHub:** create a new repo called `llm-firewall`, and commit today's work. Commit at the end of every day this week.

---

## 2. Project structure so far

```
llm-firewall/
├── demo/
│   ├── docs/                    # the "company knowledge base" the bot searches
│   │   ├── shipping_and_returns.txt   normal policy
│   │   ├── products_faq.txt           normal FAQ
│   │   ├── customer_records.txt       FAKE personal data (PII leak target)
│   │   └── vendor_update.txt          POISONED: hidden instructions (indirect injection)
│   ├── rag_app.py               # the RAG pipeline (retrieve → prompt → LLM)
│   └── app.py                   # Streamlit chat UI
├── attacks/attack_log.md        # your 20 attacks + results
├── requirements.txt
├── .env.example                 # template for your API key (real .env is never committed)
└── .gitignore
```

---

## 3. Code walkthrough: `demo/rag_app.py`

Read the file top to bottom with this beside it.

### The secret and the system prompt

```python
CANARY = "NOVA-STAFF-7Q2X"
SYSTEM_PROMPT = f"""You are NovaBot ... Internal staff discount code (CONFIDENTIAL...): {CANARY}"""
```

- The **system prompt** is the developer's hidden instructions. Users never see it — in theory.
- We hide a secret in it on purpose. Real apps hide business rules, internal notes, sometimes even API keys there (a bad practice).
- The secret doubles as a **canary**: a unique string that should never appear in any answer. If it does, we have proof of a leak. On Day 4 the firewall checks every answer for this string. (Name origin: coal miners carried canaries to detect gas early.)

### Step 1 — `load_chunks()`: documents into chunks

- Reads every `.txt` file in `demo/docs/` and splits it at blank lines into paragraphs.
- Short single lines (like "NovaMart Returns Policy") are treated as headings and glued onto the next paragraph, so each chunk keeps its context.
- Each `Chunk` remembers its `source` file, so the UI can show where an answer came from.
- **Why chunk at all?** LLMs have limited context, and you only want the relevant pieces in the prompt, not every document.

### Step 2 — `Retriever`: finding relevant chunks

- **TF-IDF** turns text into numbers. Each chunk becomes a vector where each dimension is a text piece and its value is "how important is this piece to this chunk": frequent here (TF, term frequency) but rare overall (IDF, inverse document frequency).
- `analyzer="char_wb", ngram_range=(3, 5)` means the pieces are 3-to-5 letter fragments inside words. So "refund" matches "refunds", and slightly different spellings still match.
- `search()` turns the question into the same kind of vector and ranks chunks by **cosine similarity** (how closely two vectors point in the same direction; 1 = identical, 0 = nothing in common).
- Real systems use **embedding models** (meaning-based vectors) and vector databases. TF-IDF is a simple stand-in so nothing large needs downloading. Interview tip: be ready to explain this choice and say what you'd use in production.

### Step 3 — `build_messages()`: the vulnerability lives here

```python
user_content = f"Context documents:\n{context_text}\n\nCustomer question: {question}"
```

This one line is the heart of the whole project. The documents and the user's question are **pasted into one string**. The LLM sees instructions from the developer, text from documents, and text from the user as one stream of words, with no hard boundary between "trusted" and "untrusted". That's why:

- a user can type instructions that compete with the system prompt (**direct injection**), and
- a document can contain instructions the model obeys (**indirect injection** — see `vendor_update.txt`).

### `call_llm()`

- Uses the `openai` Python package, but pointed at whatever `LLM_BASE_URL` you set. Groq, Gemini and many others accept the same request format ("OpenAI-compatible"), so switching providers means changing `.env`, not code.
- `temperature=0.2` keeps answers fairly consistent, so an attack that works once tends to work again — useful when testing.

### `NovaBot.ask()` — the pipeline

```python
retrieved = self.retriever.search(question)        # Day 5: firewall scans these chunks
messages = build_messages(question, retrieved)     # Day 5: firewall scans the question first
answer = call_llm(messages)                        # Day 5: firewall scans this answer
```

Three steps, three comments. Those comments mark exactly where the firewall's three checkpoints will plug in (the architecture diagram in your handbook).

---

## 4. Code walkthrough: `demo/app.py`

- **Streamlit** turns a Python script into a web page. Important quirk: it **re-runs the whole script** every time you interact.
- `@st.cache_resource` builds the bot once and reuses it, instead of rebuilding on every rerun.
- `st.session_state` keeps the chat history between reruns (otherwise it would vanish).
- The **sidebar** shows which chunks were retrieved for the last question. Watch it during attacks: you'll see `vendor_update.txt` get pulled in for refund questions, and `customer_records.txt` for order questions. That's how real RAG apps leak data they were never meant to show.

---

## 5. Your tasks for today

- [ ] Setup done, normal questions answered sensibly
- [ ] Play Lakera's Gandalf (search "Lakera Gandalf") for 1–2 hours; note the tricks that worked
- [ ] Run all 20 attacks in `attacks/attack_log.md`; fill in the results honestly
- [ ] Add at least 5 attacks of your own
- [ ] Write the summary at the bottom of the attack log
- [ ] Push everything to GitHub
- [ ] Explain out loud: what is the difference between direct and indirect injection, and which line of `rag_app.py` makes both possible?

**Note:** results vary by model. A strong model may resist some attacks. That's fine and worth recording: "the model resisted X but fell for Y" is a real finding. Try a smaller model too and compare.

When you're done, come back with your attack log results and we'll start Day 2 (the dataset).
