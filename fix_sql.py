from database import get_db
from models import Invoice

try:
    with get_db() as db:
        inv = db.query(Invoice).filter_by(id='NPL-2026-0005').first()
        if inv:
            if hasattr(inv, 'client_email'): setattr(inv, 'client_email', 'janakievskigogo@gmail.com')
            if hasattr(inv, 'email'): setattr(inv, 'email', 'janakievskigogo@gmail.com')
            db.commit()
            print("[SUCCESS] SQL Ledger updated! We are good to go.")
        else:
            print("[WARNING] NPL-2026-0005 not found.")
except Exception as e:
    print(f"[SQL Error] {e}")
