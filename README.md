# LLM Firewall

A security layer that protects LLM apps (RAG systems and agents) from prompt injection, jailbreaks and data leaks — including attacks written in Hinglish and Hindi.

> **Status: work in progress.** Built in public over one week.

## The problem

LLM apps paste trusted instructions, retrieved documents and user input into one prompt. The model can't reliably tell them apart, so attackers can hijack it — directly through the chat, or indirectly through a poisoned document the app retrieves on its own.

## What I found so far (Day 1)

I built a deliberately vulnerable RAG support bot (`demo/`) and attacked it with 25 prompts (`attacks/`), repeating key attacks several times:

- **Poisoned documents were the most reliable attack.** An innocent question ("How long does a refund take?") made the bot tell users to enter their card number and CVV on a fake site in **7 of 8 runs**.
- **The model's own refusals don't hold up.** A request it refused three times leaked the secret on the fourth try.
- **Hinglish attacks leaked more often than English ones in my tests** (3 of 8 vs 1 of 8). Small sample — the next step is testing this properly.

## Roadmap

- [x] Vulnerable RAG target app + attack suite
- [ ] Labelled dataset of attacks (English, Hinglish, Hindi), including indirect injections
- [ ] Baselines vs. fine-tuned multilingual detector
- [ ] Firewall service: input checks, document scanning, output PII masking
- [ ] Before/after evaluation and latency benchmarks
- [ ] Live demo

## Run the demo

```bash
python -m venv .venv
# activate it, then:
python -m pip install -r requirements.txt
cp .env.example .env        # add your LLM API key
python -m streamlit run demo/app.py
python attacks/run_attacks.py
```
