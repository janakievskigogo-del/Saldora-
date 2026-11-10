import json, os
from database import get_db
from models import Invoice

with get_db() as db:
    inv = db.query(Invoice).filter_by(id='NPL-2026-0006').first()
    if inv:
        for attr in ['client_email', 'email', 'recipient_email']:
            if hasattr(inv, attr):
                setattr(inv, attr, 'janakievskigogo@gmail.com')
        if hasattr(inv, 'client') and inv.client:
            if isinstance(inv.client, str):
                inv.client = 'janakievskigogo@gmail.com'
            else:
                for attr in ['email', 'client_email']:
                    if hasattr(inv.client, attr):
                        setattr(inv.client, attr, 'janakievskigogo@gmail.com')
        db.commit()
        print("[SUCCESS] SQL fully patched for NPL-2026-0006!")

p = 'data/invoices.json'
if os.path.exists(p):
    data = json.load(open(p, encoding='utf-8'))
    items = data if isinstance(data, list) else (data.values() if isinstance(data, dict) else [])
    for item in items:
        if isinstance(item, dict) and (item.get('id') == 'NPL-2026-0006' or item.get('invoice_id') == 'NPL-2026-0006'):
            item['client_email'] = 'janakievskigogo@gmail.com'
            item['email'] = 'janakievskigogo@gmail.com'
            if isinstance(item.get('client'), dict):
                item['client']['email'] = 'janakievskigogo@gmail.com'
            else:
                item['client'] = 'janakievskigogo@gmail.com'
    json.dump(data, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print("[SUCCESS] JSON fully patched for NPL-2026-0006!")
