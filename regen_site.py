"""
Regenerate site-test/ from cached bundles + reports.json metadata.
Reads VerdictBundle objects from SQLite cache, reconstructs SiteReport,
and calls SitePublisher.publish().
"""
import sys, os, json, sqlite3, uuid
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent / "src"))

from truthbot.publish.site import SitePublisher, SiteReport
from truthbot.models import VerdictBundle

SITE_ROOT = Path(__file__).parent / "site-test"
CACHE_DB  = Path(__file__).parent / "truthbot_cache" / "bundles" / "cache.db"
REPORTS_JSON = SITE_ROOT / "data" / "reports.json"
CLAIMS_JSON  = SITE_ROOT / "data" / "claims.json"

def load_bundles_for_report(report_id: str) -> list[VerdictBundle]:
    claims_data = json.loads(CLAIMS_JSON.read_text(encoding="utf-8"))
    claim_ids = {c["id"] for c in claims_data if c["report_id"] == report_id}
    
    # Preserve original ordering from claims.json
    ordered_claim_ids = [c["id"] for c in claims_data if c["report_id"] == report_id]
    
    conn = sqlite3.connect(str(CACHE_DB))
    rows = list(conn.execute("SELECT key, value FROM Cache"))
    conn.close()
    
    bundles_by_claim: dict[str, VerdictBundle] = {}
    for _, v in rows:
        d = json.loads(v)
        claim = d.get("claim", {})
        cid = claim.get("id")
        if cid in claim_ids and cid not in bundles_by_claim:
            bundles_by_claim[cid] = VerdictBundle.model_validate(d)
    
    # Return in order
    result = []
    for cid in ordered_claim_ids:
        if cid in bundles_by_claim:
            result.append(bundles_by_claim[cid])
    return result

def main():
    reports = json.loads(REPORTS_JSON.read_text(encoding="utf-8"))
    
    publisher = SitePublisher(site_root=str(SITE_ROOT))
    
    for r in reports:
        report_id = r["id"]
        print(f"Regenerating report {report_id[:8]}...")
        
        bundles = load_bundles_for_report(report_id)
        if not bundles:
            print(f"  WARNING: no bundles found for {report_id[:8]}, skipping")
            continue
        
        date = datetime.strptime(r["date"], "%Y-%m-%d") if r.get("date") else None
        
        site_report = SiteReport(
            report_id=report_id,
            speaker=r.get("speaker", ""),
            role=r.get("role", ""),
            date=date,
            venue=r.get("venue", ""),
            transcript_source_url="",  # not stored in reports.json; will read from bundle
            bundles=bundles,
            generated_at=datetime.now(timezone.utc),
            # New decomposed speaker/speech fields
            source_of_claims=r.get("source_of_claims", ""),
            source_of_claims_professional_public_title=r.get("source_of_claims_professional_public_title", ""),
            event=r.get("event", ""),
            channel=r.get("channel", ""),
        )
        
        # Try to get transcript URL from the existing rendered HTML
        existing = SITE_ROOT / r.get("url", "")
        if existing.exists():
            html = existing.read_text(encoding="utf-8")
            import re
            m = re.search(r'href="([^"]+)" target="_blank" rel="noopener">Transcript source', html)
            if m:
                site_report.transcript_source_url = m.group(1)
        
        report_path = publisher.publish(site_report)
        print(f"  -> {report_path}")
    
    print("\nDone. Site regenerated at", SITE_ROOT)

if __name__ == "__main__":
    main()
