"""
Utility helpers: fingerprinting, caching, hash, exif, file checks,
and bookmarks/report export helpers.
"""
import json
import csv
import io
import time
import abc
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import imagehash
import piexif
try:
    import magic
except Exception:
    magic = None
try:
    # fpdf2
    from fpdf import FPDF
except Exception:
    FPDF = None  # type: ignore

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

def is_image_file(p: Path):
    try:
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".gif",".tiff",".webp"}:
            return True
        if magic is not None:
            m = magic.from_file(str(p), mime=True)
            return bool(m and m.startswith("image/"))
        # fallback: try PIL open
        try:
            Image.open(p)
            return True
        except Exception:
            return False
    except Exception:
        return False

def fingerprint(p: Path):
    st = p.stat()
    return f"{st.st_mtime_ns}-{st.st_size}"

def cache_file_for(fp: str):
    return CACHE_DIR / f"{fp}.json"

def load_cache(fp: str):
    p = cache_file_for(fp)
    if p.exists():
        try:
            return json.load(open(p, "r"))
        except Exception:
            return None
    return None

def save_cache(fp: str, obj: dict):
    p = cache_file_for(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f)

def read_exif(path: Path):
    try:
        ex = piexif.load(str(path))
        dtb = ex["0th"].get(piexif.ImageIFD.DateTime, b"")
        dt = None
        if dtb:
            try:
                import time
                dt = time.strptime(dtb.decode(), "%Y:%m:%d %H:%M:%S")
            except Exception:
                dt = None
        return {"Make": ex["0th"].get(piexif.ImageIFD.Make, b"").decode(errors="ignore"), "Model": ex["0th"].get(piexif.ImageIFD.Model, b"").decode(errors="ignore"), "DateTime": dt}
    except Exception:
        return {}

# Backwards-compat: expose compute_perceptual_hashes here for callers expecting it
def compute_perceptual_hashes(path: Path) -> Dict[str, Any]:
    try:
        # Prefer dedicated hasher module if available
        from app.hashers.perceptual import compute_perceptual_hashes as _compute
        return _compute(path)
    except Exception:
        # Fallback inline computation
        try:
            im = Image.open(path).convert("RGB")
            return {
                "phash": str(imagehash.phash(im)),
                "ahash": str(imagehash.average_hash(im)),
                "dhash": str(imagehash.dhash(im)),
            }
        except Exception:
            return {"phash": None, "ahash": None, "dhash": None}

# ---------------- Bookmarks helpers ----------------
def load_bookmarks(case_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load bookmarks mapping from image path -> {tags: List[str], added_ts: float}.
    Returns empty dict if not found or invalid.
    """
    bookmarks_file = case_dir / "bookmarks.json"
    if bookmarks_file.exists():
        try:
            data = json.load(open(bookmarks_file, "r"))
            # normalize
            norm: Dict[str, Dict[str, Any]] = {}
            for k, v in (data or {}).items():
                tags = v.get("tags") if isinstance(v, dict) else []
                if not isinstance(tags, list):
                    tags = []
                notes = v.get("notes", "") if isinstance(v, dict) else ""
                added = v.get("added_ts") if isinstance(v, dict) else None
                if not isinstance(added, (int, float)):
                    added = time.time()
                norm[k] = {
                    "tags": [str(t).strip() for t in tags if str(t).strip()],
                    "notes": str(notes),
                    "added_ts": float(added)
                }
            return norm
        except Exception:
            return {}
    return {}

def save_bookmarks(bookmarks: Dict[str, Dict[str, Any]], case_dir: Path):
    try:
        bookmarks_file = case_dir / "bookmarks.json"
        bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(bookmarks_file, "w") as f:
            json.dump(bookmarks, f, indent=2)
    except Exception:
        pass

def _gather_metadata_for_path(path: Path) -> Dict[str, Any]:
    """Collect metadata for reports. Falls back to on-demand EXIF/hash if cache is missing."""
    from datetime import datetime
    info: Dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
    }
    # Try to get hashes from manifest if available (Streamlit context)
    try:
        import streamlit as st
        if 'manifest' in st.session_state and str(path) in st.session_state.manifest:
            info.update(st.session_state.manifest[str(path)].get("hashes", {}))
    except (ImportError, AttributeError):
        pass # Not in a streamlit context

    try:
        fp = fingerprint(path)
        cached = load_cache(fp) or {}
        md = cached.get("metadata", {}) if isinstance(cached, dict) else {}
    except Exception:
        md = {}

    if not md:
        # compute on-demand if cache is missing
        md = {"exif": read_exif(path)}

    exif = md.get("exif", {}) or {}
    dt = exif.get("DateTime")
    if isinstance(dt, tuple) or isinstance(dt, list):
        dt_str = str(dt)
    elif dt is None:
        dt_str = ""
    else:
        try:
            dt_str = time.strftime("%Y-%m-%d %H:%M:%S", dt)
        except Exception:
            dt_str = str(dt)

    info.update({
        "make": exif.get("Make", ""),
        "model": exif.get("Model", ""),
        "datetime": dt_str,
        "caption": (md.get("caption") or ""),
    })
    return info

def generate_bookmarks_csv(bookmarks: Dict[str, Dict[str, Any]]) -> bytes:
    """Create CSV bytes for current bookmarks with metadata and tags."""
    if not bookmarks:
        return b""

    output = io.StringIO(newline="")

    # Dynamically determine headers from the first bookmark's metadata
    first_path = next(iter(bookmarks.keys()))
    first_row = _gather_metadata_for_path(Path(first_path))

    base_fields = ["path", "filename", "make", "model", "datetime", "caption"]
    hash_fields = sorted(list(first_row.get("hashes", {}).keys()))
    bookmark_fields = ["tags", "notes", "bookmarked_at"]

    fields = base_fields + hash_fields + bookmark_fields
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    for path_str, meta in bookmarks.items():
        p = Path(path_str)
        row = _gather_metadata_for_path(p)

        # Flatten hashes
        row.update(row.pop("hashes", {}))

        # Add bookmark-specific data
        tags: List[str] = [t for t in (meta.get("tags") or []) if t]
        row["tags"] = "; ".join(tags)
        row["notes"] = meta.get("notes", "")
        row["bookmarked_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(meta.get("added_ts")))

        # Ensure all fields are present
        final_row = {f: row.get(f, "") for f in fields}
        writer.writerow(final_row)

    return output.getvalue().encode("utf-8")

def generate_bookmarks_pdf(bookmarks: Dict[str, Dict[str, Any]]) -> bytes:
    """Create a simple PDF report for bookmarks. Requires fpdf2. Returns bytes."""
    if not FPDF:
        # Fallback: return a single-page placeholder PDF-like bytes
        return b"PDF generation not available. Install fpdf2."
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "FORCEPS Bookmarks Report", ln=1)
    pdf.set_font("Helvetica", size=10)
    gen_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 6, f"Generated: {gen_ts}", ln=1)
    pdf.ln(2)

    for path_str, meta in bookmarks.items():
        p = Path(path_str)
        rec = _gather_metadata_for_path(p)
        tags: List[str] = [t for t in (meta.get("tags") or []) if t]

        # Header line with filename
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, rec["filename"], ln=1)
        pdf.set_font("Helvetica", size=9)

        # Try to show a small thumbnail (fits width ~60mm)
        try:
            if p.exists():
                pdf.image(str(p), w=60)
        except Exception:
            pass

        # Metadata block
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Path", ln=1)
        pdf.set_font("Helvetica", size=8)
        pdf.multi_cell(0, 5, rec['path'])

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Hashes", ln=1)
        pdf.set_font("Helvetica", size=8)
        for hash_name, hash_value in rec.get("hashes", {}).items():
            pdf.multi_cell(0, 5, f"{hash_name.upper()}: {hash_value}")

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Metadata", ln=1)
        pdf.set_font("Helvetica", size=8)
        pdf.multi_cell(0, 5, f"Make/Model: {rec['make']} / {rec['model']}")
        pdf.multi_cell(0, 5, f"Date/Time: {rec['datetime']}")

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Tags", ln=1)
        pdf.set_font("Helvetica", size=8)
        pdf.multi_cell(0, 5, '; '.join(tags) if tags else "N/A")

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Notes", ln=1)
        pdf.set_font("Helvetica", size=8)
        pdf.multi_cell(0, 5, meta.get("notes") or "N/A")

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "AI Caption", ln=1)
        pdf.set_font("Helvetica", size=8)
        pdf.multi_cell(0, 5, rec.get("caption") or "N/A")

        pdf.ln(2)

    out = pdf.output(dest="S").encode("latin1")
    return out

def hamming_distance(hex_str1: str, hex_str2: str) -> int:
    """Calculate the Hamming distance between two hex strings."""
    try:
        # Convert hex strings to integers
        h1 = int(hex_str1, 16)
        h2 = int(hex_str2, 16)
        # XOR the integers and count the number of set bits (1s)
        return bin(h1 ^ h2).count('1')
    except (ValueError, TypeError):
        # Return a large distance if hashes are invalid or not strings
        return 256 # Larger than any possible 64-bit hash distance
