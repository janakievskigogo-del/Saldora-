import json, os
from database import get_db
from models import Invoice

# 1. Update SQL Database using context manager
try:
    with get_db() as db:
        inv = db.query(Invoice).filter_by(invoice_id='NPL-2026-0005').first()
        if inv:
            if hasattr(inv, 'client_email'): setattr(inv, 'client_email', 'janakievskigogo@gmail.com')
            if hasattr(inv, 'email'): setattr(inv, 'email', 'janakievskigogo@gmail.com')
            db.commit()
            print("[SUCCESS] SQL Ledger updated!")
        else:
            print("[WARNING] Invoice NPL-2026-0005 not found in SQL database.")
except Exception as e:
    print(f"[SQL Error] {e}")

# 2. Update JSON Store
p = 'data/invoices.json'
if os.path.exists(p):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and (item.get('invoice_id') == 'NPL-2026-0005' or item.get('id') == 'NPL-2026-0005'):
                    item['client_email'] = 'janakievskigogo@gmail.com'
                    item['email'] = 'janakievskigogo@gmail.com'
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and (v.get('invoice_id') == 'NPL-2026-0005' or k == 'NPL-2026-0005'):
                    v['client_email'] = 'janakievskigogo@gmail.com'
                    v['email'] = 'janakievskigogo@gmail.com'

        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("[SUCCESS] JSON Store updated!")
    except Exception as e:
        print(f"[JSON Error] {e}")
