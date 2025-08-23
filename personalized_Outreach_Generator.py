import os, csv, time, pathlib
from dataclasses import dataclass
from typing import Optional, Tuple
from openai import OpenAI

# ---- Pricing for gpt-4o-mini (USD/token) ----
INPUT_PER_TOKEN  = 0.15 / 1_000_000
OUTPUT_PER_TOKEN = 0.60 / 1_000_000
MODEL = "gpt-4o-mini"

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

def est_cost(u: Usage) -> float:
    return u.prompt_tokens * INPUT_PER_TOKEN + u.completion_tokens * OUTPUT_PER_TOKEN

SYSTEM_PROMPT = (
    "You are a sales assistant for Magnusbane (AI Automation Agency). "
    "Generate highly personalized but concise cold outreach components for SMBs. "
    "Tone: friendly, clear, no fluff. European spelling. Avoid hype words."
)

USER_TEMPLATE = """Company: {company}
Niche: {niche}
Website (optional): {website}
Target output language: {lang}

Tasks:
1) Subject: 5-7 words, no emojis, no ALL CAPS.
2) Opener: 1-2 sentences, personalized to the niche/company (avoid repeating the company name in every line).
3) 3 bullets: one-week automations Magnusbane can deliver using Zapier/Make + OpenAI—specific to the niche.
4) CTA: one short line proposing a 15-min call this week.

Output EXACTLY in this format:
Subject: <text>
Opener: <text>
Bullets:
- <bullet 1>
- <bullet 2>
- <bullet 3>
CTA: <text>
"""

def call_model(client: OpenAI, company: str, niche: str, website: Optional[str], lang: str) -> Tuple[str, Usage]:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                company=company, niche=niche, website=website or "N/A", lang=lang)},
        ],
    )
    content = resp.choices[0].message.content.strip()
    usage = Usage(
        prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
        completion_tokens=getattr(resp.usage, "completion_tokens", 0),
        total_tokens=getattr(resp.usage, "total_tokens", 0),
    )
    return content, usage

def parse_output(text: str) -> dict:
    """Parse fixed-format model output into fields."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out = {"subject":"", "opener":"", "bullet1":"", "bullet2":"", "bullet3":"", "cta":""}
    bullets = []; section = None
    for l in lines:
        low = l.lower()
        if low.startswith("subject:"):
            out["subject"] = l.split(":",1)[1].strip(); section=None; continue
        if low.startswith("opener:"):
            out["opener"] = l.split(":",1)[1].strip(); section=None; continue
        if low.startswith("bullets:"):
            section = "bullets"; continue
        if l.startswith("-") and section == "bullets":
            bullets.append(l[1:].strip()); continue
        if low.startswith("cta:"):
            out["cta"] = l.split(":",1)[1].strip(); section=None; continue
    for i in range(3):
        out[f"bullet{i+1}"] = bullets[i] if i < len(bullets) else ""
    return out

def build_email_body(company: str, niche: str, parsed: dict, lang: str = "English") -> str:
    """
    Produce a clean plain‑text email body using parsed parts.
    Keep it short, no fluff. Edit signature as you like.
    """
    subject = parsed.get("subject","").strip()
    opener  = parsed.get("opener","").strip()
    b1 = parsed.get("bullet1","").strip()
    b2 = parsed.get("bullet2","").strip()
    b3 = parsed.get("bullet3","").strip()
    cta = parsed.get("cta","").strip()

    body_lines = []
    body_lines.append(opener)
    if b1 or b2 or b3:
        body_lines.append("")
        if lang.lower().startswith("ro"):
            body_lines.append("Ce putem livra în 1 săptămână:")
        else:
            body_lines.append("What we can deliver in 1 week:")
        if b1: body_lines.append(f"- {b1}")
        if b2: body_lines.append(f"- {b2}")
        if b3: body_lines.append(f"- {b3}")
    if cta:
        body_lines.append("")
        body_lines.append(cta)
    body_lines.append("")
    body_lines.append("Best,")
    body_lines.append("Magnusbane Team")
    return "\n".join(body_lines)

def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Missing OPENAI_API_KEY. Put it in your environment or a .env file.")

    import argparse
    ap = argparse.ArgumentParser(description="Magnusbane Outreach Generator (CSV → CSV)")
    ap.add_argument("--infile", default="prospects_sample.csv",
                    help="CSV with columns: company_name,niche,website(optional),lang(optional)")
    ap.add_argument("--outfile", default=None, help="Output CSV path (auto if omitted)")
    ap.add_argument("--default-lang", default="English", help="Fallback language if 'lang' column missing")
    ap.add_argument("--delay-ms", type=int, default=600, help="Delay between API calls (ms)")
    args = ap.parse_args()

    inp = pathlib.Path(args.infile)
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    ts = time.strftime("%Y%m%d-%H%M")
    outdir = pathlib.Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)
    outfile = pathlib.Path(args.outfile) if args.outfile else outdir / f"prospects_out-{ts}.csv"

    client = OpenAI()
    total = Usage()

    with inp.open("r", encoding="utf-8") as f_in, outfile.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["company_name","niche","website","lang",
                      "subject","opener","bullet1","bullet2","bullet3","cta",
                      "email_body",
                      "prompt_tokens","completion_tokens","total_tokens","est_cost_usd"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            company = (row.get("company_name") or "").strip()
            niche   = (row.get("niche") or "").strip()
            website = (row.get("website") or "").strip()
            lang    = (row.get("lang") or args.default_lang).strip() or args.default_lang
            if not company or not niche:
                print(f"Skipping row (missing company/niche): {row}")
                continue

            text, usage = call_model(client, company, niche, website, lang)
            parsed = parse_output(text)
            email_body = build_email_body(company, niche, parsed, lang=lang)

            total.prompt_tokens     += usage.prompt_tokens
            total.completion_tokens += usage.completion_tokens
            total.total_tokens      += usage.total_tokens

            writer.writerow({
                "company_name": company, "niche": niche, "website": website, "lang": lang,
                **parsed,
                "email_body": email_body,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "est_cost_usd": f"{est_cost(usage):.6f}",
            })

            time.sleep(max(args.delay_ms, 0) / 1000.0)  # gentle rate-limit guard

    grand = est_cost(total)
    print(f"\nSaved → {outfile}")
    print(f"Totals: prompt={total.prompt_tokens}, completion={total.completion_tokens}, total={total.total_tokens}")
    print(f"Estimated total cost: ${grand:.6f} (model: {MODEL})\n")

if __name__ == "__main__":
    main()
