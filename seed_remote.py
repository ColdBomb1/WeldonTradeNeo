"""Run this on the remote server to seed the database with rules and config.

Usage:
    cd C:\WeldonTradeNeo
    python seed_remote.py
"""

import json
from datetime import datetime, timezone
from db import get_session, init_db
from models.ruleset import RuleSet

init_db()

# Load exported ruleset
with open("data/ruleset_export.json", "r", encoding="utf-8") as f:
    data = json.load(f)

session = get_session()
try:
    existing = session.query(RuleSet).filter(RuleSet.name == data["name"]).first()
    if existing:
        print(f"Updating existing ruleset: {existing.name} v{existing.version} -> v{data['version']}")
        existing.rules_text = data["rules_text"]
        existing.version = data["version"]
        existing.symbols = data["symbols"]
        existing.timeframes = data["timeframes"]
        existing.status = data["status"]
        existing.performance_metrics = data.get("performance_metrics")
        existing.updated_at = datetime.now(timezone.utc)
    else:
        print(f"Creating ruleset: {data['name']} v{data['version']}")
        session.add(RuleSet(
            name=data["name"],
            status=data["status"],
            rules_text=data["rules_text"],
            symbols=data["symbols"],
            timeframes=data["timeframes"],
            version=data["version"],
            performance_metrics=data.get("performance_metrics"),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))

    session.commit()
    print("Done.")
except Exception as e:
    session.rollback()
    print(f"Error: {e}")
finally:
    session.close()
