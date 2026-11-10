import json, os
from database import get_db
from models import Invoice

# 1. Fix SQL Ledger
try:
    with get_db() as db:
        inv = db.query(Invoice).filter_by(id='NPL-2026-0006').first()
        if inv:
            inv.client_email = 'janakievskigogo@gmail.com'
            inv.email = 'janakievskigogo@gmail.com'
            db.commit()
            print("[SUCCESS] SQL Ledger email fixed for NPL-2026-0006!")
        else:
            print("[WARNING] NPL-2026-0006 not found in SQL.")
except Exception as e:
    print(f"[SQL Error] {e}")

# 2. Fix JSON Store
p = 'data/invoices.json'
if os.path.exists(p):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get('id') == 'NPL-2026-0006':
                    item['client_email'] = 'janakievskigogo@gmail.com'
                    item['email'] = 'janakievskigogo@gmail.com'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("[SUCCESS] JSON Store email fixed for NPL-2026-0006!")
    except Exception as e:
        print(f"[JSON Error] {e}")
