from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from typing import List, Optional, Union

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

try:
    from text_cleaner import clean_the_text
except Exception:
    def clean_the_text(t: str) -> str:
        return (t or "").strip()
try:
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except Exception: 
        Retry = None
    _URL_OK = True
except Exception:
    _URL_OK = False

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "model" / "clickbait_model.joblib"
FALLBACK_MODEL_PATH = PROJECT_DIR / "model" / "click_model.joblib"
MIN_MODEL_SIZE_BYTES = 10 * 1024  # 10KB
REPORTS_DIR = PROJECT_DIR / "reports"
CONTENT_DUMP_PATH = REPORTS_DIR / "last_url_content.txt"
PREVIEW_CHARS = 10000

# --- Logger Kurulumu ---
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(name)s - %(message)s', '%H:%M:%S'))
    LOGGER.addHandler(handler)
LOGGER.propagate = False

def configure_logging(level: str | None = None) -> None:
    lvl = (level or os.getenv("APP_LOG_LEVEL", "INFO")).upper().strip()
    chosen = getattr(logging, lvl, logging.INFO)
    LOGGER.setLevel(chosen)

configure_logging()

if _URL_OK:
    LOGGER.info("URL desteği aktif (requests + bs4).")
else:
    LOGGER.warning("URL desteği pasif: 'requests' veya 'beautifulsoup4' eksik.")
LOGGER.info("Modül yüklendi ve hazır.")

TRIGGERS = (
    "şok", "şoke", "flaş", "acil", "göreceksiniz", "inanamayacaksınız", "bakın", "izle", "video",
    "galeri", "tıklayın", "detaylar", "son dakika", "müthiş", "büyük sürpriz", "resmen"
)

NEG_TRIGGERS = (
    "yalan", "yalanlandı", "doğrulandı", "resmi açıklama", "açıklama", 
    "doğru değil", "resmi", "iddia değil", "bakanlık açıkladı", "bakanlık açıkladı"
)

def normalize_text(text: str) -> str:
    return (text or "").strip()

def _heuristic_predict(text: str) -> dict:
    t = normalize_text(text).lower()
    
    length = len(t)
    exclam = t.count("!")
    quest = t.count("?")
    
    trig_hits = sum(1 for w in TRIGGERS if w in t)
    neg_hits = sum(1 for w in NEG_TRIGGERS if w in t)
    score = 0.0
    score += min(trig_hits / 3.0, 1.0) * 0.6
    score += (1.0 if exclam > 0 else 0.0) * 0.15
    score += (1.0 if quest > 0 else 0.0) * 0.1
    score += (1.0 if length > 120 else 0.0) * 0.15
    
    if neg_hits > 0:
        score -= min(neg_hits * 0.2, 0.6)
        
    score = max(0.0, min(score, 1.0))
    
    label_name = "CLICKBAIT" if score >= 0.5 else "NORMAL"
    
    return {"label_name": label_name, "probability_clickbait": round(score, 4)}

# --- Hibrit Skor Birleştirme ---
def _combine_model_and_heuristic(model_prob: float | None, heur_prob: float | None) -> dict:
    if model_prob is None and heur_prob is None:
        return {"combined": None, "label": None, "confidence": 0.0, "disagreement": False}
    if model_prob is None:
        p = heur_prob
        label = "CLICKBAIT" if p >= 0.5 else "NORMAL"
        return {"combined": round(p,4), "label": label, "confidence": 0.5, "disagreement": False}
    if heur_prob is None:
        p = model_prob
        label = "CLICKBAIT" if p >= 0.5 else "NORMAL"
        return {"combined": round(p,4), "label": label, "confidence": 0.5, "disagreement": False}
    diff = abs(model_prob - heur_prob)
    combined = 0.7 * model_prob + 0.3 * heur_prob
    label = "CLICKBAIT" if combined >= 0.5 else "NORMAL"
    confidence = round(1.0 - diff, 4)
    return {
        "combined": round(combined,4),
        "label": label,
        "confidence": confidence,
        "disagreement": bool(diff > 0.35),
    }

HEADERS_DEFAULT = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_1) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
}

def _create_session() -> "requests.Session":
    if not _URL_OK:
        raise RuntimeError("URL desteği devre dışı (requests/bs4 mevcut değil).")
    s = requests.Session()
    if 'Retry' in globals() and Retry is not None:
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504, 403),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    return s

def _build_candidate_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    if not netloc and parsed.path:
        netloc = parsed.path
        path = ""
    netloc_www = netloc if (not netloc or netloc.startswith("www.")) else "www." + netloc

    def _make(scheme_: str, netloc_: str, path_: str, query_: str = query) -> str:
        return urlunparse((scheme_ or "https", netloc_ or netloc, path_ or path, "", query_ or "", ""))

    base_candidates = [
        _make(parsed.scheme or "https", netloc, path),
        _make("https", netloc, path),
        _make("https", netloc_www, path),
    ]

    amp_candidates: list[str] = []
    if path:
        if not path.rstrip("/").endswith("/amp"):
            amp_path = path.rstrip("/") + "/amp"
            amp_candidates += [
                _make("https", netloc, amp_path),
                _make("https", netloc_www, amp_path),
            ]
        amp_candidates += [
            _make("https", netloc, path, "amp=1"),
            _make("https", netloc_www, path, "amp=1"),
            _make("https", netloc, path, "outputType=amp"),
            _make("https", netloc_www, path, "outputType=amp"),
        ]

    seen = set()
    out: list[str] = []
    for u in [url] + base_candidates + amp_candidates:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _extract_text_from_html(html: Union[str, bytes]) -> str:
    soup = BeautifulSoup(html, "html.parser")
    candidates = [
        *soup.select("article"),
        *soup.select("main"),
        *soup.select("div[itemprop='articleBody']"),
        *soup.select("div[class*='content']"),
    ]
    for c in candidates or [soup]:
        ps = c.find_all(["p", "h1", "h2", "li"]) if c else []
        if ps:
            text_parts = [p.get_text(" ", strip=True) for p in ps]
            text = "\n".join(tp for tp in text_parts if tp)
            if len(text) >= 60:
                return text

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json as _json
            data = _json.loads(script.get_text(strip=True))
            items = data if isinstance(data, list) else [data]
            for it in items:
                if isinstance(it, dict):
                    body = it.get("articleBody")
                    if isinstance(body, str) and len(body) >= 60:
                        return body
        except Exception:
            continue

    og = soup.find("meta", attrs={"property": "og:description"})
    if og and og.get("content") and len(og["content"]) >= 60:
        return og["content"].strip()

    return soup.get_text(" ", strip=True)

def _is_valid_url_basic(url: str) -> bool:
    try:
        p = urlparse(url)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False

def read_text_from_url(url: str, timeout: int = 15, max_attempts: int = 3) -> str:
    if not _URL_OK:
        raise RuntimeError("URL desteği devre dışı (requests/bs4 mevcut değil).")
    if not _is_valid_url_basic(url):
        raise RuntimeError(f"Geçersiz URL formatı: {url!r}")
    if max_attempts <= 0:
        max_attempts = 1
    session = _create_session()
    last_err: Exception | None = None
    best_text: str = ""
    candidates = _build_candidate_urls(url)
    if len(candidates) > max_attempts:
        original_len = len(candidates)
        candidates = candidates[:max_attempts]
        LOGGER.debug("Aday URL listesi kısaltıldı | orijinal=%d -> kullanılan=%d", original_len, len(candidates))
    LOGGER.info("URL içerik indirme başlatıldı | aday_sayısı=%d | max_attempts=%d", len(candidates), max_attempts)
    for cand in candidates:
        try:
            r = session.get(
                cand,
                headers=HEADERS_DEFAULT,
                timeout=timeout,
                allow_redirects=True,
            )
            if r.status_code < 200 or r.status_code >= 300:
                last_err = RuntimeError(f"HTTP {r.status_code} - {cand}")
                LOGGER.warning("İstek başarısız | status=%s | url=%s", r.status_code, cand)
                continue
            text = _extract_text_from_html(r.content)
            if len(text) >= 60:
                LOGGER.info("URL içerik başarıyla alındı | seçilen=%s | uzunluk=%d", cand, len(text))
                return text
            if len(text) > len(best_text):
                best_text = text
            last_err = RuntimeError(f"Çok kısa içerik: {cand}")
        except Exception as e:
            last_err = e
            LOGGER.warning("İstek hatası | url=%s | hata=%s: %s", cand, type(e).__name__, e)
            continue

    if best_text and len(best_text) >= 10:
        LOGGER.info("Kısmi kısa içerik (fallback) döndürüldü len=%d", len(best_text))
        return best_text
    if last_err:
        LOGGER.error("Tüm aday URL denemeleri başarısız | deneme_sayısı=%d | son_hata=%s", len(candidates), last_err)
        raise last_err
    raise RuntimeError("URL'den içerik alınamadı (neden bilinmiyor)")

def predict_with_pipeline(pipe: Pipeline, texts: List[str]) -> List[dict]:
    X = list(texts)
    preds = pipe.predict(X)
    probs = None
    used = "none"
    try:
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X)[:, 1]
            used = "predict_proba"
    except Exception:
        probs = None
    if probs is None:
        try:
            if hasattr(pipe, "decision_function"):
                dec = pipe.decision_function(X)
                if getattr(dec, "ndim", 1) == 2 and dec.shape[1] > 1:
                    dec = dec[:, -1]
                dec = np.asarray(dec, dtype=float)
                probs = 1.0 / (1.0 + np.exp(-dec))
                used = "decision_function-sigmoid"
        except Exception:
            probs = None
    LOGGER.info("Model olasılık yöntemi: %s", used)
    out = []
    for i, p in enumerate(preds):
        prob = float(probs[i]) if probs is not None else None
        out.append({
            "label": int(p),
            "label_name": "CLICKBAIT" if int(p) == 1 else "NORMAL",
            "probability_clickbait": None if prob is None else round(prob, 4),
            "text": X[i],
        })
    return out

def _resolve_model_path(user_path: Optional[str]) -> Path:
    candidates = []
    if user_path:
        p = Path(user_path).expanduser()
        candidates.extend([p, PROJECT_DIR / p])
    candidates.extend([DEFAULT_MODEL_PATH, FALLBACK_MODEL_PATH])
    for c in candidates:
        try:
            if c.exists() and c.stat().st_size >= MIN_MODEL_SIZE_BYTES:
                return c
        except Exception:
            continue
    return DEFAULT_MODEL_PATH

_MODEL_CACHE: dict[str, Pipeline] = {}

def train_clickbait_model(
    csv_path: Optional[str] = None,
    text_col: str = "text",
    label_col: str = "label",
    model_out: Optional[str] = None,
    model_type: str = "logreg",
    do_grid: bool = False,
) -> Optional[Path]:
    try:
        from URL_Clickbait_detector_train import (
            train as _train_impl,
            DEFAULT_CSV as _DEF_CSV,
            DEFAULT_MODEL_PATH as _DEF_MODEL_PATH,
        )
    except Exception as e: 
        LOGGER.warning("Eğitim modülü import edilemedi: %s", e)
        return None

    csv_p = Path(csv_path) if csv_path else _DEF_CSV
    model_p = Path(model_out) if model_out else _DEF_MODEL_PATH
    try:
        LOGGER.info("Model eğitimi başlıyor | csv=%s -> model=%s", csv_p, model_p)
        _train_impl(
            csv_path=csv_p,
            text_col=text_col,
            label_col=label_col,
            model_out=model_p,
            model_type=model_type,
            do_grid=do_grid,
        )
        if model_p.exists() and model_p.stat().st_size >= MIN_MODEL_SIZE_BYTES:
            LOGGER.info("Model eğitimi tamamlandı | model=%s", model_p)
            return model_p
        LOGGER.warning("Model çıktısı bulunamadı veya çok küçük | path=%s", model_p)
    except Exception as e:
        LOGGER.exception("Model eğitimi başarısız: %s", e)
    return None

def _load_pipeline_cached(model_path: Path) -> Optional[Pipeline]:
    key = str(model_path.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        pipe: Pipeline = joblib.load(model_path)
        _MODEL_CACHE[key] = pipe
        try:
            tfidf = pipe.named_steps.get("tfidf")
            clf = pipe.named_steps.get("clf")
            vocab_size = getattr(tfidf, "vocabulary_", None)
            LOGGER.info(
                "Model yüklendi | sınıflar=%s | vocab_size=%s",
                getattr(clf, "classes_", None),
                (len(vocab_size) if vocab_size else "?"),
            )
        except Exception:
            pass
        return pipe
    except Exception as e:
        LOGGER.warning("Model yüklenemedi | path=%s | hata=%s", model_path, e)
        return None

def get_or_train_pipeline(model_path: Path, auto_train: bool = False) -> Optional[Pipeline]:
    """Önce yükle; yoksa ve auto_train True ise eğit."""
    pipe = _load_pipeline_cached(model_path)
    if pipe is not None:
        return pipe
    if auto_train:
        LOGGER.info("Model bulunamadı, auto_train etkin -> eğitim denenecek.")
        trained_path = train_clickbait_model(model_out=str(model_path))
        if trained_path:
            return _load_pipeline_cached(trained_path)
    return None

def predict_from_url(
    url: str,
    model_in: Optional[str] = None,
    auto_train: bool = False,
    use_heuristic_alongside: bool = True,
    return_combined: bool = True,
    max_attempts: int = 3,
) -> dict:
    content = read_text_from_url(url, max_attempts=max_attempts)
    model_path = _resolve_model_path(model_in)
    pipe = get_or_train_pipeline(model_path, auto_train=auto_train)
    LOGGER.info("Tahmin başlatıldı | url=%s | model=%s", url, model_path.name)
    model_score: float | None = None
    model_label: str | None = None
    if pipe is not None:
        out = predict_with_pipeline(pipe, [content])[0]
        model_score = out.get("probability_clickbait")
        model_label = out.get("label_name")
    LOGGER.info("Model tahmini tamamlandı | label=%s | score=%s", model_label, model_score)

    heur_score: float | None = None
    heur_label: str | None = None
    heur_dict: dict | None = None
    if (pipe is None) or use_heuristic_alongside:
        heur_dict = _heuristic_predict(content)
        heur_score = heur_dict.get("probability_clickbait")  # type: ignore
        heur_label = heur_dict.get("label_name")  # type: ignore
        if pipe is None:
            LOGGER.info("Heuristik tahmin (yalnız) | label=%s | score=%s", heur_label, heur_score)
        else:
            LOGGER.info("Heuristik ek skor | label=%s | score=%s", heur_label, heur_score)

    combined_info = None
    final_label: str | None = None
    final_score: float | None = None
    if return_combined and (model_score is not None or heur_score is not None):
        combined_info = _combine_model_and_heuristic(model_score, heur_score)
        final_label = combined_info.get("label")
        final_score = combined_info.get("combined")
    else:
        # Geriye dönük: varsa model yoksa heuristik
        if model_score is not None:
            final_label = model_label
            final_score = model_score
        else:
            final_label = heur_label
            final_score = heur_score

    result = {
        "label": final_label,
        "score": final_score,
        "text_len": len(content),
        "source": url,
        "content": content,
        "model_score": model_score,
        "model_label": model_label,
        "heuristic_score": heur_score,
        "heuristic_label": heur_label,
    }
    if combined_info:
        result.update({
            "combined_score": combined_info.get("combined"),
            "combined_label": combined_info.get("label"),
            "confidence": combined_info.get("confidence"),
            "disagreement": combined_info.get("disagreement"),
        })
    if pipe is None:
        result["reason"] = "heuristic_only"
    return result

def predict_from_url_message(url: str) -> str:
    """Arayüz için özet mesaj döndürür + temiz içerik önizlemesi yazdırır."""
    res = predict_from_url(url)
    score = res.get("score")
    score_txt = "?" if score is None else f"{score:.4f}"
    raw = (res.get("content") or "").strip()
    cleaned = clean_the_text(raw)
    preview = cleaned[:PREVIEW_CHARS] + ("..." if len(cleaned) > PREVIEW_CHARS else "")
    print("\n[CLEANED CONTENT PREVIEW]\n" + preview)
    LOGGER.info("Özet üretildi | label=%s | score=%s | len=%s", res.get('label'), score_txt, res.get('text_len'))
    return f"Label: {res.get('label')} | Score: {score_txt} | Len: {res.get('text_len')}"

def run_url_only_cli(model_in: Optional[str] = None) -> int:
    print("URL-only mod. Bir URL girin (boş bırakılırsa çıkış):")
    try:
        url = input("URL: ").strip()
    except EOFError:
        return 1
    if not url:
        print("URL verilmedi. Çıkılıyor.")
        return 1
    try:
        res = predict_from_url(url, model_in=model_in)
        label = res.get('label')
        score = res.get('score')
        length = res.get('text_len')
        content = res.get('content') or ""
        print(f"Label: {label} | Score: {score} | Length: {length}")
        if content:
            preview = content[:PREVIEW_CHARS] + ("..." if len(content) > PREVIEW_CHARS else "")
            print("\n--- İçerik Önizlemesi ---\n" + preview + "\n-------------------------")
            try:
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                CONTENT_DUMP_PATH.write_text(content, encoding="utf-8")
                print(f"\n[Tam içerik kaydedildi]: {CONTENT_DUMP_PATH}")
            except Exception as dump_err:
                LOGGER.warning("İçerik dosyaya kaydedilemedi | hata=%s", dump_err)
        return 0
    except Exception as e:
        LOGGER.exception("CLI URL işleme hatası: %s", e)
        print(f"Hata: {e}")
        return 2

def main():
    try:
        code = run_url_only_cli(None)
        raise SystemExit(code)
    except SystemExit as e:
        raise e
    except Exception as e:
        LOGGER.exception("Beklenmeyen hata: %s", e)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
