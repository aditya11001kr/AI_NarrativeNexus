import pandas as pd
import re

def clean_body(raw_text):
    """Deep clean the body text only."""
    if pd.isna(raw_text):
        return ""

    # Split headers vs body
    parts = re.split(r"\n\s*\n", raw_text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    cleaned_lines = []
    for line in body.splitlines():
        if re.match(
            r"^(archive-name|from|subject|path|xref|organization|lines|newsgroups|message-id|keywords|last-modified|version):",
            line,
            re.I,
        ):
            continue
        if line.strip().startswith((">", "|")):  # quoted text
            continue
        if line.strip().startswith("--"):  # signature
            break
        if re.search(r"In article\s*<.*?>", line, re.I):
            continue
        if re.search(r"writes:|wrote:", line, re.I):
            continue
        cleaned_lines.append(line)

    body = "\n".join(cleaned_lines)

    # Remove emails, urls, html tags
    body = re.sub(r"\S+@\S+", " ", body)
    body = re.sub(r"http\S+|www\.\S+", " ", body)
    body = re.sub(r"<[^>]+>", " ", body)

    # Remove non-alphanumeric except punctuation
    body = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", body)

    # Collapse multiple spaces/newlines
    body = re.sub(r"\n{2,}", "\n", body)
    body = re.sub(r"\s{2,}", " ", body)

    
    body = body.lower().strip()

    
    body = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", body)

    return body


df = pd.read_excel("20news-18828_clean_50.xlsx", engine="openpyxl")


df["text"] = df["text"].apply(clean_body)


df.to_excel("20news-18828_final_50.xlsx", index=False, engine="openpyxl")

print(f" Final dataset saved: {len(df)} rows, cleaned text only")
