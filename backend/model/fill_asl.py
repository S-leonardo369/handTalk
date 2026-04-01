"""
fill_asl.py — populate vocab_map.json with SignASL video IDs.

Paste the full embed code from signasl.org when prompted.
The script extracts the data-vidref ID automatically.
Press Enter to skip a sign. Ctrl+C to stop and save anytime.

Usage:
    cd C:\\Users\\Bazuka\\documents\\handTalk\\backend\\model
    python fill_asl.py
"""

import json
import re
import os

MAP_PATH = "vocab_map.json"

def extract_vidref(raw: str) -> str | None:
    """Pull data-vidref="..." from any paste containing it."""
    raw = raw.strip()
    # Match data-vidref="<id>"
    m = re.search(r'data-vidref=["\']([^"\']+)["\']', raw)
    if m:
        return m.group(1).strip()
    # Maybe they pasted just the bare ID (no HTML) — accept if it looks like one
    if re.fullmatch(r'[a-zA-Z0-9_\-]{5,20}', raw):
        return raw
    return None

def main():
    if not os.path.exists(MAP_PATH):
        print(f"[ERROR] {MAP_PATH} not found. Run this script from backend/model/")
        return

    with open(MAP_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Sort alphabetically by sign name
    entries = sorted(data.items(), key=lambda x: x[1]["sign"].lower())

    total    = len(entries)
    filled   = sum(1 for _, v in entries if v.get("asl_vidref"))
    skipped  = 0
    added    = 0

    print(f"\n{'─'*60}")
    print(f"  handTalk — ASL Video ID Filler")
    print(f"{'─'*60}")
    print(f"  Signs total : {total}")
    print(f"  Already filled: {filled}")
    print(f"  Remaining   : {total - filled}")
    print(f"{'─'*60}")
    print("  Paste the FULL embed code from signasl.org")
    print("  The ID is extracted automatically.")
    print("  Press Enter to skip. Ctrl+C to stop and save.")
    print(f"{'─'*60}\n")

    try:
        for k, v in entries:
            sign = v["sign"]

            # Skip already filled
            if v.get("asl_vidref"):
                continue

            print(f"  [{added + skipped + filled + 1}/{total}]  {sign.upper()}")
            print(f"  URL: https://www.signasl.org/sign/{sign.replace(' ', '-').replace('_', '-')}")

            while True:
                raw = input("  Paste embed (or Enter to skip): ").strip()

                if not raw:
                    skipped += 1
                    print("  → skipped\n")
                    break

                vid_id = extract_vidref(raw)
                if vid_id:
                    data[k]["asl_vidref"] = vid_id
                    added += 1
                    print(f"  ✓ saved: {vid_id}\n")
                    break
                else:
                    print("  ✗ Could not find data-vidref in that paste. Try again.\n")

    except KeyboardInterrupt:
        print("\n\n  Interrupted — saving progress…")

    # Save
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    filled_now = sum(1 for v in data.values() if v.get("asl_vidref"))
    print(f"\n{'─'*60}")
    print(f"  Saved to {MAP_PATH}")
    print(f"  Added this session : {added}")
    print(f"  Skipped this session: {skipped}")
    print(f"  Total filled now   : {filled_now} / {total}")
    print(f"{'─'*60}\n")

if __name__ == "__main__":
    main()